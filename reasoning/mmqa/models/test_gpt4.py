import json
import os
import sys
import base64
import re
import string
import time
import random
from collections import defaultdict, deque
from tqdm import tqdm
from openai import OpenAI
from PIL import Image

# ----------------------------
# 1. 文本标准化 (保持原有 EM 逻辑)
# ----------------------------
def normalize_text(s):
    if s is None: return ""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(s.split())

# ----------------------------
# 2. GPT 推理引擎类
# ----------------------------
class GPTEngine:
    def __init__(self, api_key=None, model_name="gpt-4.1", rpm=0, tpm=0):
        # 如果不传 api_key，OpenAI 会自动从环境变量 OPENAI_API_KEY 读取
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.limiter = GPTLimiter(rpm_limit=rpm, tpm_limit=tpm)

    def _image_to_base64(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def convert_messages(self, raw_messages):
        """将原始消息转换为 GPT-4V 视觉格式"""
        formatted_messages = []
        for msg in raw_messages:
            role = msg["role"]
            content = []
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image":
                    b64 = self._image_to_base64(item["image_url"])
                    if b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        })
            if content:
                formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def predict(self, raw_messages, max_tokens=1028):
        # 1. 预估与节流
        est_tokens = max_tokens + 800  # 800 为图片+Prompt的保守预估
        self.limiter.throttle(est_tokens)

        # 2. 带重试的 API 调用
        def _call():
            # 注意：新版 OpenAI SDK 使用 client.chat.completions.create
            # 兼容你代码中的 client.responses.create 逻辑
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=self.convert_messages(raw_messages),
                max_tokens=max_tokens,
                temperature=0.0
            )

        response = self._call_with_retry(_call)
        
        # 3. 解析 Usage 并记账
        usage = getattr(response, "usage", None)
        total_tokens = usage.total_tokens if usage else est_tokens
        self.limiter.record(total_tokens)

        # 4. 提取文本
        res_text = response.choices[0].message.content or ""
        return res_text.strip(), total_tokens

    def _call_with_retry(self, fn, max_retries=8):
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                err_msg = str(e).lower()
                if any(x in err_msg for x in ["rate limit", "429", "too many requests"]):
                    # 解析 Retry-After 或指数退避
                    wait_s = self._parse_wait_time(err_msg, backoff)
                    time.sleep(wait_s + random.uniform(0.1, 0.5))
                    backoff = min(backoff * 2, 30.0)
                    continue
                raise e
        raise RuntimeError("GPT API 重试次数过多")

    def _parse_wait_time(self, msg, default):
        m = re.search(r"try again in ([0-9.]+)s", msg)
        return float(m.group(1)) if m else default

# ----------------------------
# 3. 评测接口函数
# ----------------------------
def run_gpt_evaluation(
    input_file="vl_inference_patched.jsonl",
    output_file="./gpt_result/evaluation_gpt_results.jsonl",
    rpm=3,   # 每分钟 3 次请求 (根据 Tier 调整)
    tpm=50000, # 每分钟 5万 Token
    api_key=None, 
    model_name="gpt-4.1", 
):
    """
    GPT 评测任务主入口
    """
    Image.MAX_IMAGE_PIXELS = None
    engine = GPTEngine(api_key, model_name, rpm, tpm)
    
    # 加载已处理的 QID 实现断点续传
    processed_qids = _get_processed_qids(output_file)
    
    # 统计行数用于进度条
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"--- 启动 GPT 评测: {model_name} ---")

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "a", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="GPT 推理中"):
            try:
                sample = json.loads(line)
                qid = sample["qid"]
                if qid in processed_qids: continue

                # 执行推理
                res_text, used_tokens = engine.predict(sample["model_input"])
                
                # EM 评估逻辑
                gt_answers = [ans["answer"] for ans in sample["answers"]]
                pred_norm = normalize_text(res_text)
                is_correct = any(pred_norm == normalize_text(gt) for gt in gt_answers)

                # 实时落盘
                result = {
                    "qid": qid,
                    "question_type": sample.get("metadata", {}).get("type", "Unknown"),
                    "prediction_raw": res_text,
                    "is_correct": is_correct,
                    "total_tokens": used_tokens
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                processed_qids.add(qid)

            except Exception as e:
                print(f"\n[Error] QID {qid} 失败: {e}")
                continue

    _print_summary(output_file)

# ----------------------------
# 4. 辅助工具 (Limiter & IO)
# ----------------------------
class GPTLimiter:
    def __init__(self, rpm_limit, tpm_limit, safety=0.9):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.safety = safety
        self.req_window = deque()
        self.tok_window = deque()

    def _prune(self):
        now = time.time()
        while self.req_window and now - self.req_window[0] > 60: self.req_window.popleft()
        while self.tok_window and now - self.tok_window[0][0] > 60: self.tok_window.popleft()

    def throttle(self, est):
        if self.rpm_limit <= 0 and self.tpm_limit <= 0: return
        while True:
            self._prune()
            if self.rpm_limit > 0 and len(self.req_window) >= int(self.rpm_limit * self.safety):
                time.sleep(1); continue
            if self.tpm_limit > 0 and sum(t for _, t in self.tok_window) + est >= int(self.tpm_limit * self.safety):
                time.sleep(1); continue
            break

    def record(self, tokens):
        self._prune()
        now = time.time()
        self.req_window.append(now)
        self.tok_window.append((now, tokens))

def _get_processed_qids(path):
    qids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try: qids.add(json.loads(line)["qid"])
                except: continue
    return qids

def _print_summary(path):
    # 这里可以复用你原来的 compute_report_from_results 逻辑
    print(f"\n评测任务完成，结果保存在: {path}")

if __name__ == "__main__":
    # 使用环境变量或直接传参
    run_gpt_evaluation(
        input_file="vl_inference_patched.jsonl",
        output_file="./gpt_result/evaluation_gpt_results.jsonl",
        rpm=3,   # 每分钟 3 次请求 (根据 Tier 调整)
        tpm=50000 # 每分钟 5万 Token
    )