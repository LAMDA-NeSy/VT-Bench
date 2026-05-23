import json
import os
import sys
import string
import re
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

# -----------------------------------------------------------
# 1. Text Normalization & Cleaning Utilities
# -----------------------------------------------------------

def normalize_text(s):
    """Standard EM normalization: lowercase, remove punctuation and articles."""
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_prediction_text(pred_text):
    """
    Extracts answer from GLM-4.1V thinking format.
    1. Priority: Extract content within <answer>...</answer> tags.
    2. Fallback: Prefix cleaning and sentence truncation.
    """
    s = pred_text.strip()

    # --- Step 1: Extract from <answer> tags ---
    match = re.search(r'<answer>(.*?)</answer>', s, re.DOTALL)
    if match:
        s = match.group(1).strip()
        # GLM specific token cleaning
        s = s.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
        if s.startswith('"') and s.endswith('"') and len(s) > 2:
            s = s[1:-1]
        return s.strip()

    # --- Step 2: Fallback logic for non-tagged output ---
    prefixes = [
        "the final answer is:", "the answer is:", "my answer is:",
        "the answer is", "is:", "was:", "are:", "were:"
    ]
    s_lower = s.lower()
    last_pos, prefix_len = -1, 0
    for p in prefixes:
        pos = s_lower.rfind(p)
        if pos > last_pos:
            last_pos, prefix_len = pos, len(p)
            
    if last_pos != -1:
        s = s[last_pos + prefix_len:].strip()

    # Take first line and truncate at first punctuation
    s = s.split('\n')[0].strip()
    s = re.split(r'[\.\!\?]\s+', s, maxsplit=1)[0].strip()
    
    if s.startswith('"') and s.endswith('"') and len(s) > 2:
        s = s[1:-1]

    return s.strip()

# -----------------------------------------------------------
# 2. GLM Inference Engine Class
# -----------------------------------------------------------

class GLMThinkingEngine:
    def __init__(self, model_id="zai-org/GLM-4.1V-9B-Thinking", cache_dir=None):
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['TRANSFORMERS_CACHE'] = cache_dir

        print(f"--- Loading GLM Model: {model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Setup Pad Token
        self.pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id or 0
        self.processor.tokenizer.pad_token_id = self.pad_id
        self.model.config.pad_token_id = self.pad_id

    def _convert_messages(self, raw_messages):
        """Converts MMQA raw messages to GLM format (PIL Image conversion)."""
        new_msgs = []
        for msg in raw_messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            new_content = []
            for item in content:
                if item.get("type") == "image":
                    path = item.get("image_url") or item.get("image") or item.get("image_path")
                    if isinstance(path, str) and path:
                        try:
                            img = Image.open(path).convert("RGB")
                            new_content.append({"type": "image", "image": img})
                        except Exception as e:
                            print(f"[WARN] Failed to load image: {path} -> {e}", file=sys.stderr)
                    elif isinstance(path, Image.Image):
                        new_content.append({"type": "image", "image": path})
                elif item.get("type") == "text":
                    if item.get("text"):
                        new_content.append({"type": "text", "text": item["text"]})
            new_msgs.append({"role": role, "content": new_content})
        return new_msgs

    @torch.no_grad()
    def predict_thinking(self, raw_messages, short_max=1024, long_max=1024):
        """
        Inference with self-correction/retry logic.
        If the model fails to output </answer> in 256 tokens, it retries with 1024.
        """
        msgs = self._convert_messages(raw_messages)
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, 
            return_tensors="pt", return_dict=True
        ).to(self.device)
        
        input_len = inputs["input_ids"].shape[1]

        # First Attempt (Short context)
        out1 = self.model.generate(**inputs, max_new_tokens=short_max, pad_token_id=self.pad_id)
        text = self.processor.decode(out1[0, input_len:], skip_special_tokens=True).strip()

        # Second Attempt (Long context if tag is missing)
        if "</answer>" not in text:
            out2 = self.model.generate(**inputs, max_new_tokens=long_max, pad_token_id=self.pad_id)
            text = self.processor.decode(out2[0, input_len:], skip_special_tokens=True).strip()
            
        return text

# -----------------------------------------------------------
# 3. Main Evaluation Interface
# -----------------------------------------------------------

def run_glm_evaluation(input_file="mmqa_for_qwen_vl_inference_patched.jsonl", 
                       output_file="mmqa_eval_glm_pass.jsonl", 
                       model_id="zai-org/GLM-4.1V-9B-Thinking", 
                       cache_dir=None):
    """
    Main interface for MMQA evaluation using GLM-4.1V.
    """
    Image.MAX_IMAGE_PIXELS = None
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    engine = GLMThinkingEngine(model_id, cache_dir)
    
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct, total_count = 0, 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Evaluating GLM"):
            try:
                sample = json.loads(line)
                qid, qtype = sample["qid"], sample["metadata"].get("type", "Unknown")
                ground_truths = [a["answer"] for a in sample["answers"]]

                # Thinking Inference
                raw_prediction = engine.predict_thinking(sample["model_input"])
                
                # Cleanup & Metrics
                if "</answer>" in raw_prediction:
                    cleaned = clean_prediction_text(raw_prediction)
                    pred_norm = normalize_text(cleaned)
                    gt_norms = [normalize_text(x) for x in ground_truths]
                    is_correct = pred_norm in gt_norms
                else:
                    # Failure case: tag never produced
                    cleaned, pred_norm, is_correct = "", "", False

                # Update Stats
                total_count += 1
                type_stats[qtype]["total"] += 1
                if is_correct:
                    total_correct += 1
                    type_stats[qtype]["correct"] += 1

                # Save Results
                result = {
                    "qid": qid,
                    "question_type": qtype,
                    "prediction_raw": raw_prediction,
                    "prediction_cleaned": cleaned,
                    "is_correct": is_correct,
                    "ground_truths": ground_truths
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"[ERROR] QID {sample.get('qid', 'NA')}: {e}", file=sys.stderr)

    _print_report(type_stats, total_correct, total_count)

def _print_report(type_stats, correct, total):
    print("\n" + "="*55)
    print(f"{'Question Type':<40} | {'Accuracy'}")
    print("-" * 55)
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        acc = (s["correct"] / s["total"]) * 100 if s["total"] > 0 else 0
        print(f"{t[:40]:<40} | {acc:6.2f}% ({s['correct']}/{s['total']})")
    
    overall_acc = (correct / total) * 100 if total > 0 else 0
    print("-" * 55)
    print(f"{'OVERALL EM ACCURACY':<40} | {overall_acc:6.2f}%")
    print("="*55)

if __name__ == "__main__":
    run_glm_evaluation(
        input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="mmqa_eval_glm_pass.jsonl",
        cache_dir='/mnt/hdd/jiazy/GLM-4.1V-9B-Thinking'
    )