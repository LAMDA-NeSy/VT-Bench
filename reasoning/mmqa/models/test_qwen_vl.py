import json
import os
import sys
import string
import re
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# -----------------------------------------------------------
# 1. Logging and Utility Classes
# -----------------------------------------------------------

class Tee:
    """Redirects output to both console and a log file."""
    def __init__(self, console, file_obj):
        self.console = console
        self.file = file_obj

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()

def redirect_output_to_log(log_path="evaluation.log"):
    """Wraps stdout and stderr to record all logs to a file."""
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    return log_f

# -----------------------------------------------------------
# 2. Text Normalization & Cleaning Utilities
# -----------------------------------------------------------

def normalize_text(s):
    """Standard EM normalization: lowercase, remove punctuation and articles."""
    s = str(s).strip().lower()
    # Normalize time formats (e.g., 7:05 p.m. -> 7:05pm)
    s = re.sub(r'(\d)\s*:\s*(\d+)\s*p\s*\.?\s*m\.?', r'\1:\2pm', s)
    s = re.sub(r'(\d)\s*:\s*(\d+)\s*a\s*\.?\s*m\.?', r'\1:\2am', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'(\d+)\s+(am|pm)\b', r'\1\2', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_prediction_text(pred_text):
    """
    Robust cleaning for thinking models:
    1. Strip <think>...</think> blocks.
    2. Extract content following specific answer prefixes.
    3. Truncate to the first sentence/line.
    """
    s = pred_text.strip()
    
    # Remove thought process blocks if present
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()

    # Prefix cleaning
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
        
    # Take the first line and first sentence
    s = s.split('\n')[0].strip()
    s = re.split(r'[\.\!\?]\s+', s, maxsplit=1)[0].strip()
    
    # Remove wrapping quotes
    if s.startswith('"') and s.endswith('"') and len(s) > 2:
        s = s[1:-1]
        
    return s.strip()

# -----------------------------------------------------------
# 3. Qwen3-VL Inference Engine
# -----------------------------------------------------------

class Qwen3ThinkingEngine:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Thinking"):
        print(f"--- Loading Qwen3-VL Thinking Model: {model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

        # Ensure pad token is set correctly
        pad_id = self.processor.tokenizer.pad_token_id 
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id or 0
        self.processor.tokenizer.pad_token_id = pad_id
        self.model.config.pad_token_id = pad_id

    def _convert_image_urls(self, messages):
        """Converts local file paths in messages to PIL Image objects."""
        new_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)
                continue
            content = msg.get("content", [])
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image_url" in item:
                    try:
                        img = Image.open(item["image_url"]).convert("RGB")
                        new_content.append({"type": "image", "image": img})
                    except Exception as e:
                        print(f"[WARN] Failed to open image {item['image_url']}: {e}")
                else:
                    new_content.append(item)
            new_msg = dict(msg)
            new_msg["content"] = new_content
            new_messages.append(new_msg)
        return new_messages

    @torch.no_grad()
    def predict(self, raw_messages, max_new_tokens=128):
        """Processes messages and returns the raw textual response."""
        messages = self._convert_image_urls(raw_messages)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id 
        )

        input_len = inputs['input_ids'].shape[1]
        response_ids = gen_ids[0, input_len:]
        response_text = self.processor.decode(response_ids, skip_special_tokens=True).strip()
        return response_text

# -----------------------------------------------------------
# 4. Main Evaluation Interface
# -----------------------------------------------------------

def run_qwen3_evaluation(
        input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="evaluation_thinking_results.jsonl", 
        model_id="Qwen/Qwen3-VL-8B-Thinking", 
        log_path="test.log"):
    """
    Main interface to run the Qwen3-VL Thinking evaluation.
    Supports checkpointing (skips processed QIDs).
    """
    Image.MAX_IMAGE_PIXELS = None
    log_f = redirect_output_to_log(log_path)
    
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Check existing results for resuming
        done_qids = _load_existing_qids(output_file)
        print(f"Resuming evaluation. Found {len(done_qids)} completed samples.")

        engine = Qwen3ThinkingEngine(model_id)

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'a', encoding='utf-8') as f_out:

            for line in tqdm(f_in, desc="Processing Qwen3"):
                try:
                    sample = json.loads(line)
                    qid = sample['qid']
                    if qid in done_qids:
                        continue
                        
                    q_type = sample['metadata'].get('type', 'Unknown')
                    # Filter for specific logic if needed
                    if q_type != "Intersect(ImageListQ,TextQ)": 
                        continue

                    # Perform Inference
                    raw_res = engine.predict(sample['model_input'])
                    
                    # Evaluation logic
                    cleaned = clean_prediction_text(raw_res)
                    pred_norm = normalize_text(cleaned)
                    gt_answers = [ans['answer'] for ans in sample['answers']]
                    gt_norms = [normalize_text(gt) for gt in gt_answers]
                    is_correct = pred_norm in gt_norms

                    # Save Result
                    result = {
                        "qid": qid,
                        "question_type": q_type,
                        "prediction_raw": raw_res,
                        "prediction_cleaned": cleaned,
                        "is_correct": is_correct,
                        "ground_truths": gt_answers
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                    done_qids.add(qid)

                except Exception as e:
                    print(f"[ERROR] Exception at QID {sample.get('qid', 'N/A')}: {e}")

        # Generate Final Report from the result file
        _print_final_report(output_file)

    finally:
        log_f.close()

# -----------------------------------------------------------
# 5. Helper Functions
# -----------------------------------------------------------

def _load_existing_qids(path):
    qids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    qids.add(json.loads(line)["qid"])
                except:
                    continue
    return qids

def _print_final_report(results_path):
    print("\n--- Final Evaluation Report ---")
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_c, total_t = 0, 0

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj["question_type"]
                stats[t]["total"] += 1
                if obj["is_correct"]: stats[t]["correct"] += 1
            except: continue

    for q_type in sorted(stats.keys()):
        s = stats[q_type]
        acc = (s["correct"] / s["total"]) * 100
        print(f"{q_type:<45}: Acc: {acc:6.2f}% ({s['correct']}/{s['total']})")
        total_c += s["correct"]
        total_t += s["total"]

    overall_acc = (total_c / total_t) * 100 if total_t > 0 else 0
    print("-" * 60)
    print(f"{'Overall EM Accuracy':<45}: Acc: {overall_acc:6.2f}% ({total_c}/{total_t})")

if __name__ == "__main__":
    run_qwen3_evaluation(
        input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="evaluation_thinking_results.jsonl"
    )