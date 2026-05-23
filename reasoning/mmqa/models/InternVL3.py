import json
import os
import sys
import string
import re
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# -----------------------------------------------------------
# 1. 文本处理工具 (保持逻辑不变)
# -----------------------------------------------------------

def normalize_text(s):
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_prediction_text(pred_text):
    s = pred_text.strip()
    prefixes_to_split = [
        "the final answer is:", "the answer is:", "my answer is:",
        "answer is:", "answer:", "ans:", "the answer is",
        "is:", "was:", "are:", "were:",
    ]
    s_lower = s.lower()
    last_prefix_pos, prefix_len = -1, 0
    for prefix in prefixes_to_split:
        pos = s_lower.rfind(prefix)
        if pos > last_prefix_pos:
            last_prefix_pos, prefix_len = pos, len(prefix)

    if last_prefix_pos != -1:
        s = s[last_prefix_pos + prefix_len:].strip()
    
    s = s.split("\n")[0].strip()
    if s.startswith('"') and s.endswith('"') and len(s) > 2:
        s = s[1:-1].strip()
    return s.strip()

# -----------------------------------------------------------
# 2. 推理引擎类 (封装模型加载与单次推理)
# -----------------------------------------------------------

class InternVLInferenceEngine:
    def __init__(self, model_id="OpenGVLab/InternVL3-8B-hf", device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"--- 正在加载模型: {model_id} 到 {self.device} ---")
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # 处理 pad_token
        self.pad_id = self._setup_pad_token()
        print(f"模型加载完成，pad_token_id: {self.pad_id}")

    def _setup_pad_token(self):
        pad_id = getattr(self.processor, "pad_token_id", None)
        if pad_id is None and hasattr(self.processor, "tokenizer"):
            pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.pad_token_id = pad_id
        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = pad_id
        return pad_id

    def convert_messages(self, raw_messages):
        """将原始 MMQA 消息格式转为模型输入格式"""
        new_msgs = []
        for msg in raw_messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            new_content = []
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        img_path = item.get("image_url") or item.get("image") or item.get("image_path")
                        if img_path:
                            img = Image.open(img_path).convert("RGB")
                            new_content.append({"type": "image", "image": img})
                    elif item.get("type") == "text":
                        new_content.append({"type": "text", "text": item.get("text", "")})
            else:
                new_content.append({"type": "text", "text": str(content)})
            new_msgs.append({"role": role, "content": new_content})
        return new_msgs

    @torch.no_grad()
    def predict(self, raw_messages, max_new_tokens=1024):
        """核心推理函数"""
        messages = self.convert_messages(raw_messages)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.pad_id,
        )

        input_len = inputs['input_ids'].shape[1]
        response_text = self.processor.decode(gen_ids[0, input_len:], skip_special_tokens=True).strip()
        return response_text

# -----------------------------------------------------------
# 3. 评测接口函数 (实现业务逻辑)
# -----------------------------------------------------------

def run_internvl(input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="result/mmqa_internvl3_8b_results.jsonl", 
        model_id="OpenGVLab/InternVL3-8B-hf", device=None):
    """
    外部调用的主要函数接口
    """
    Image.MAX_IMAGE_PIXELS = None
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件: {input_file}")

    # 初始化引擎
    engine = InternVLInferenceEngine(model_id=model_id, device=device)

    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="评估进度"):
            try:
                sample = json.loads(line)
                qid = sample['qid']
                question_type = sample['metadata'].get('type', 'Unknown')
                gt_answers = [ans['answer'] for ans in sample['answers']]

                # 执行推理
                response_raw = engine.predict(sample['model_input'])
                
                # 清洗与归一化
                cleaned_response = clean_prediction_text(response_raw)
                pred_normalized = normalize_text(cleaned_response)
                gt_normalized_list = [normalize_text(gt) for gt in gt_answers]
                
                is_correct = (pred_normalized in gt_normalized_list)

                # 统计
                overall_total += 1
                type_stats[question_type]["total"] += 1
                if is_correct:
                    overall_correct += 1
                    type_stats[question_type]["correct"] += 1

                # 写入结果
                result_entry = {
                    "qid": qid,
                    "question_type": question_type,
                    "prediction_raw": response_raw,
                    "prediction_cleaned": cleaned_response,
                    "is_correct": is_correct
                }
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"\n[错误] QID {sample.get('qid', 'N/A')}: {e}", file=sys.stderr)

    # 打印最终报告
    print_report(type_stats, overall_correct, overall_total)
    return {"overall_acc": overall_correct / overall_total if overall_total > 0 else 0}

def print_report(type_stats, overall_correct, overall_total):
    print("\n" + "="*50)
    print(f"{'Question Type':<40} | {'Accuracy':<10}")
    print("-" * 55)
    for q_type in sorted(type_stats.keys()):
        stats = type_stats[q_type]
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{q_type[:40]:<40} | {acc:6.2f}% ({stats['correct']}/{stats['total']})")
    
    overall_acc = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print("-" * 55)
    print(f"{'OVERALL EM ACCURACY':<40} | {overall_acc:6.2f}%")
    print("="*50)

# -----------------------------------------------------------
# 4. 使用示例
# -----------------------------------------------------------

if __name__ == "__main__":
    # 你只需要调用这一个函数即可
    run_internvl(
        input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="result/mmqa_internvl3_8b_results.jsonl"
    )