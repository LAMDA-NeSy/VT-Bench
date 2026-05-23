import json
import os
import sys
import time
import mimetypes
import string
import re
import traceback
from collections import deque, defaultdict
from tqdm import tqdm
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from PIL import Image

# -----------------------------------------------------------
# 1. 频率限制器 (维持高性能逻辑)
# -----------------------------------------------------------

class GeminiRateLimiter:
    def __init__(self, rpm_limit: int = 0, tpm_limit: int = 0, safety: float = 0.90):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.safety = safety
        self.req_window = deque()
        self.tok_window = deque()

    def _prune(self):
        now = time.time()
        while self.req_window and now - self.req_window[0] > 60.0:
            self.req_window.popleft()
        while self.tok_window and now - self.tok_window[0][0] > 60.0:
            self.tok_window.popleft()

    def throttle(self, estimated_tokens: int = 0):
        while True:
            self._prune()
            if self.rpm_limit > 0 and len(self.req_window) + 1 > int(self.rpm_limit * self.safety):
                time.sleep(max(60.0 - (time.time() - self.req_window[0]) + 0.1, 0.5))
                continue
            if self.tpm_limit > 0:
                used_tpm = sum(t for _, t in self.tok_window)
                if used_tpm + estimated_tokens > int(self.tpm_limit * self.safety):
                    time.sleep(max(60.0 - (time.time() - self.tok_window[0][0]) + 0.1, 0.5))
                    continue
            break

    def record(self, used_tokens: int | None = None, fallback_tokens: int = 0):
        self._prune()
        now = time.time()
        self.req_window.append(now)
        tok = used_tokens if used_tokens is not None else fallback_tokens
        if self.tpm_limit > 0 and tok:
            self.tok_window.append((now, int(tok)))

# -----------------------------------------------------------
# 2. Gemini 推理引擎类
# -----------------------------------------------------------

class GeminiEngine:
    def __init__(self, api_key, model_name="gemini-2.0-flash", rpm=0, tpm=0):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.limiter = GeminiRateLimiter(rpm_limit=rpm, tpm_limit=tpm)

    def _get_image_part(self, path):
        if not os.path.exists(path):
            print(f"[警告] 图片不存在: {path}")
            return None
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or "image/jpeg"
        with open(path, "rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type=mime_type)

    def convert_messages(self, raw_messages):
        messages = []
        for msg in raw_messages:
            role = "model" if msg["role"] == "assistant" else "user"
            parts = []
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item["text"]))
                elif item.get("type") == "image":
                    img_part = self._get_image_part(item["image_url"])
                    if img_part: parts.append(img_part)
            if parts:
                messages.append(types.Content(role=role, parts=parts))
        return messages

    def predict(self, raw_messages, estimated_tokens=1500):
        # 1. 频率控制
        self.limiter.throttle(estimated_tokens)
        
        # 2. 准备输入
        contents = self.convert_messages(raw_messages)
        
        # 3. 带重试的生成
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                
                # 4. 记账
                used = response.usage_metadata.total_token_count if response.usage_metadata else None
                self.limiter.record(used_tokens=used, fallback_tokens=estimated_tokens)
                
                # 5. 解析输出
                if not response.candidates:
                    return "", 0
                
                text = response.candidates[0].content.parts[0].text or ""
                return text.strip(), (used or 0)

            except google_exceptions.ResourceExhausted:
                wait_time = 2 ** (attempt + 1)
                print(f"\n[429] 频率限制触发，等待 {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"API 异常: {e}")
                raise e
        return "", 0

# -----------------------------------------------------------
# 3. 评测接口函数
# -----------------------------------------------------------

def run_gemini_mmqa_evaluation(
    input_file, 
    output_file, 
    api_key, 
    model_name="gemini-2.0-flash",
    checkpoint_file="checkpoint_gemini.json",
    rpm=15, 
    tpm=1000000
):
    """
    Gemini 评测主要函数接口
    """
    Image.MAX_IMAGE_PIXELS = None
    engine = GeminiEngine(api_key, model_name, rpm, tpm)

    # 加载进度与 Checkpoint
    done_qids = _get_done_qids(output_file)
    checkpoint = _load_json(checkpoint_file) or {}
    start_line = checkpoint.get("line_number", 0)

    print(f"--- 启动 Gemini 评测: {model_name} ---")
    print(f"--- 跳过已完成: {len(done_qids)} 条 ---")

    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(input_file, "r", encoding="utf-8") as f_in:
        # 为了 tqdm 计数，我们先拿到总行数
        all_lines = f_in.readlines()

    for i, line in enumerate(tqdm(all_lines, desc="Gemini 评测进度"), start=1):
        if i <= start_line: continue
        
        try:
            sample = json.loads(line)
            qid = sample["qid"]
            if qid in done_qids: continue

            # 推理
            res_text, cost = engine.predict(sample["model_input"])
            
            # 评估 (EM 逻辑)
            gt_answers = [ans["answer"] for ans in sample["answers"]]
            pred_norm = _normalize_text(res_text)
            gt_norms = [_normalize_text(gt) for gt in gt_answers]
            is_correct = pred_norm in gt_norms

            # 保存结果
            result_entry = {
                "qid": qid,
                "question_type": sample["metadata"].get("type", "Unknown"),
                "prediction_raw": res_text,
                "is_correct": is_correct,
                "cost": cost
            }
            _append_jsonl(output_file, result_entry)
            
            # 更新 Checkpoint
            _save_json(checkpoint_file, {"last_processed_qid": qid, "line_number": i})
            done_qids.add(qid)

        except Exception as e:
            print(f"\n[错误] 处理 QID {sample.get('qid')} 时中断: {e}")
            traceback.print_exc()
            break # 严重错误建议先停下来检查

    # 生成最终报告
    _print_final_report(output_file)

# -----------------------------------------------------------
# 4. 辅助工具函数 (内部使用)
# -----------------------------------------------------------

def _normalize_text(s):
    if not s: return ""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(s.split())

def _get_done_qids(path):
    qids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try: qids.add(json.loads(line)["qid"])
                except: continue
    return qids

def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return None

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f)

def _append_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def _print_final_report(results_path):
    """从结果文件中重新统计并打印完整报告"""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_cost = 0
    
    if not os.path.exists(results_path): return

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj["question_type"]
            stats[t]["total"] += 1
            if obj["is_correct"]: stats[t]["correct"] += 1
            total_cost += obj.get("cost", 0)

    print("\n" + "="*50)
    print(f"{'Question Type':<35} | {'Accuracy'}")
    print("-" * 50)
    grand_total, grand_correct = 0, 0
    for qt in sorted(stats.keys()):
        s = stats[qt]
        acc = (s["correct"] / s["total"]) * 100
        print(f"{qt[:35]:<35} | {acc:6.2f}% ({s['correct']}/{s['total']})")
        grand_total += s["total"]
        grand_correct += s["correct"]
    
    overall_acc = (grand_correct / grand_total) * 100 if grand_total > 0 else 0
    print("-" * 50)
    print(f"{'OVERALL ACCURACY':<35} | {overall_acc:6.2f}%")
    print(f"Total Token Cost: {total_cost}")
    print("="*50)

# -----------------------------------------------------------
# 5. 启动
# -----------------------------------------------------------

if __name__ == "__main__":
    # 使用时填入你的 Key
    MY_GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
    
    run_gemini_mmqa_evaluation(
        input_file="vl_inference_patched.jsonl",
        output_file="evaluation_gemini_results.jsonl",
        api_key=MY_GEMINI_KEY,
        model_name="gemini-2.0-flash-exp", # 推荐
        rpm=10, # 根据你的 API 账号层级调整
        tpm=1000000
    )