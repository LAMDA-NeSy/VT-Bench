import json
import os
import sys
import string
import re
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# -----------------------------------------------------------
# 1. 文本处理工具
# -----------------------------------------------------------

def normalize_text(s):
    _ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
    s = str(s).strip().lower()
    s = re.sub(r'(\d)\s*:\s*(\d+)\s*p\s*\.?\s*m\.?', r'\1:\2pm', s)
    s = re.sub(r'(\d)\s*:\s*(\d+)\s*a\s*\.?\s*m\.?', r'\1:\2am', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'(\d+)\s+(am|pm)\b', r'\1\2', s)
    s = _ARTICLES.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_prediction_text(pred_text):
    s = pred_text.strip()
    prefixes_to_split = [
        "the final answer is:", "the answer is:", "my answer is:", 
        "the answer is", "is:", "was:", "are:", "were:"
    ]
    s_lower = s.lower()
    last_prefix_pos, prefix_len = -1, 0
    for prefix in prefixes_to_split:
        pos = s_lower.rfind(prefix)
        if pos > last_prefix_pos:
            last_prefix_pos, prefix_len = pos, len(prefix)
    
    if last_prefix_pos != -1:
        s = s[last_prefix_pos + prefix_len:].strip()
    
    s = s.split('\n')[0].strip()
    s = re.split(r'[\.\!\?]\s+', s, maxsplit=1)[0].strip()
    if s.startswith('"') and s.endswith('"') and len(s) > 2:
        s = s[1:-1]
    return s.strip()

# -----------------------------------------------------------
# 2. Pixtral 推理引擎类
# -----------------------------------------------------------

class PixtralEngine:
    def __init__(self, model_id="mistral-community/pixtral-12b"):
        print(f"--- 正在加载 Pixtral 模型: {model_id} ---")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            low_cpu_mem_usage=True,
            # 如果显存足够，可以开启 torch_dtype=torch.bfloat16
        )
        self.model.eval()
        self._setup_pad_token()

    def _setup_pad_token(self):
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

    def _process_images_and_chat(self, raw_messages, max_size=(448, 448)):
        """将原始消息转为 Pixtral 格式并处理图片"""
        images_list = []
        chat = []
        
        for msg in raw_messages:
            role = msg["role"]
            new_content = []
            for item in msg["content"]:
                if item["type"] == "image":
                    path = item.get("image_url") or item.get("image")
                    
                    if os.path.exists(path):
                        try:
                            img = Image.open(path).convert("RGB")
                            img.thumbnail(max_size, Image.Resampling.LANCZOS)
                            images_list.append(img)
                            new_content.append({"type": "image"})
                        except Exception as e:
                            print(f"图片加载失败: {path} -> {e}")
                elif item["type"] == "text":
                    if item["text"].strip():
                        new_content.append({"type": "text", "content": item["text"]})
            chat.append({"role": role, "content": new_content})
        return chat, images_list

    @torch.no_grad()
    def predict(self, raw_messages, max_new_tokens=1024):
        chat, images = self._process_images_and_chat(raw_messages)
        
        # 使用模板生成 prompt
        prompt = self.processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        
        inputs_args = {"text": prompt, "return_tensors": "pt", "padding": True}
        if images:
            inputs_args["images"] = images

        inputs = self.processor(**inputs_args).to(self.model.device)
        
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            do_sample=False
        )

        # 解码并切掉 prompt
        input_len = inputs["input_ids"].shape[1]
        response_ids = gen_ids[0, input_len:]
        response_text = self.processor.decode(response_ids, skip_special_tokens=True).strip()
        
        return response_text

# -----------------------------------------------------------
# 3. 评测接口函数
# -----------------------------------------------------------

def run_pixtral_evaluation(
        input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="result/mmqa_pixtral_results.jsonl", 
        model_id="mistral-community/pixtral-12b"):
    """
    Pixtral 评测主要入口
    """
    Image.MAX_IMAGE_PIXELS = None
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    engine = PixtralEngine(model_id=model_id)
    
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct, overall_total = 0, 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Pixtral 评测进度"):
            try:
                sample = json.loads(line)
                qid = sample['qid']
                q_type = sample['metadata'].get('type', 'Unknown')
                gt_answers = [ans['answer'] for ans in sample['answers']]

                # 推理
                raw_response = engine.predict(sample['model_input'])
                
                # 清洗与归一化
                cleaned_res = clean_prediction_text(raw_response)
                pred_norm = normalize_text(cleaned_res)
                gt_norms = [normalize_text(gt) for gt in gt_answers]
                
                is_correct = (pred_norm in gt_norms)

                # 统计
                overall_total += 1
                type_stats[q_type]["total"] += 1
                if is_correct:
                    overall_correct += 1
                    type_stats[q_type]["correct"] += 1

                # 实时保存
                result = {
                    "qid": qid,
                    "question_type": q_type,
                    "prediction_raw": raw_response,
                    "prediction_cleaned": cleaned_res,
                    "is_correct": is_correct
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"QID {sample.get('qid')} 发生异常: {e}")

    _print_summary(type_stats, overall_correct, overall_total)

def _print_summary(type_stats, overall_correct, overall_total):
    print("\n" + "="*60)
    print(f"{'Question Type':<45} | {'Accuracy'}")
    print("-" * 60)
    for qt in sorted(type_stats.keys()):
        s = type_stats[qt]
        acc = (s["correct"] / s["total"]) * 100 if s["total"] > 0 else 0
        print(f"{qt[:45]:<45} | {acc:6.2f}% ({s['correct']}/{s['total']})")
    
    overall_acc = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print("-" * 60)
    print(f"{'OVERALL EM ACCURACY':<45} | {overall_acc:6.2f}%")
    print("="*60)

# -----------------------------------------------------------
# 4. 运行示例
# -----------------------------------------------------------

if __name__ == "__main__":
    run_pixtral_evaluation(
        input_file="/lamda12/jiazy/09jiazy/MMQA/mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="/lamda12/jiazy/09jiazy/MMQA/result/mmqa_pixtral_results.jsonl"
    )