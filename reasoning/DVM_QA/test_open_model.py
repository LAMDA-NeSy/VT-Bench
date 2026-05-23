import os
import sys
import json
import re
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoModelForImageTextToText,
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration,
    Glm4vForConditionalGeneration,
)

# Optional: Vision parsing for Qwen3-VL
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# Global Context Constraints
MAX_CONTEXT_LEN = 1024
MAX_NEW_TOKENS_CAP = 1024

CURRENT_FILE_DIR = Path(__file__).resolve().parent
# ============================================================
# 1. Parsing & Logic Utilities (Kept from original)
# ============================================================

def parse_predicted_row_id(text: str, n_rows=None):
    if not isinstance(text, str): text = str(text)
    json_objs = re.findall(r"\{[\s\S]*?\}", text)
    for obj in reversed(json_objs):
        try:
            data = json.loads(obj)
            if isinstance(data, dict) and "answer" in data:
                ans = data["answer"]
                if isinstance(ans, list) and len(ans) > 0: ans = ans[0]
                cand = str(int(ans)) if isinstance(ans, (int, float)) else re.search(r"\d+", str(ans)).group(0) if re.search(r"\d+", str(ans)) else None
                if cand and (n_rows is None or 1 <= int(cand) <= int(n_rows)): return cand
        except: continue
    strong_patterns = [r"answer\s*[:=]\s*\"?(?P<id>\d+)\"?", r"row\s*id\s*[:=#]?\s*\"?(?P<id>\d+)\"?"]
    for pat in strong_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            cand = m.group("id")
            if n_rows is None or 1 <= int(cand) <= int(n_rows): return cand
    return None

def parse_predicted_attr_answer(text: str):
    if not isinstance(text, str): text = str(text)
    m = re.search(r'"answer"\s*:\s*(?:\[(?P<inner>.*?)\]|"(?P<val>[^"]+)"|(?P<raw>[^\n\r,}]*))', text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        res = m.group("inner") or m.group("val") or m.group("raw")
        return res.strip().strip('"[] ')
    m = re.search(r"answer\s*[:=]\s*(?P<val>.+)", text, flags=re.IGNORECASE)
    return m.group("val").strip().strip("`*_ ") if m else None

def is_numeric_like_answer_correct(gt, pred, rel_tol=0.0, abs_tol=1e-6):
    def extract(s):
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(s))
        return float(nums[0]) if nums else None
    g_n, p_n = extract(gt), extract(pred)
    if g_n is not None and p_n is not None:
        diff = abs(g_n - p_n)
        return diff <= abs_tol or (diff / max(abs(g_n), abs(p_n), 1e-8)) <= rel_tol
    return str(gt).strip().lower() == str(pred).strip().lower()

def truncate_model_inputs_inplace(inputs, max_len):
    if not isinstance(inputs, dict) or "input_ids" not in inputs: return inputs
    cur_len = inputs["input_ids"].shape[1]
    if cur_len <= max_len: return inputs
    start = cur_len - max_len
    for k in ["input_ids", "attention_mask", "position_ids"]:
        if k in inputs and inputs[k] is not None: inputs[k] = inputs[k][:, start:]
    return inputs

# ============================================================
# 2. Model Loading & Core Evaluation
# ============================================================

def load_model_and_processor(model_id: str):
    print(f"--- Loading Model: {model_id} ---")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    lower_id = model_id.lower()
    
    if "qwen3-vl" in lower_id: cls = AutoModelForVision2Seq
    elif "internvl3" in lower_id: cls = AutoModelForImageTextToText
    elif "llama-3.2" in lower_id: cls = MllamaForConditionalGeneration
    elif "glm-4.1v" in lower_id: cls = Glm4vForConditionalGeneration
    elif "pixtral" in lower_id: cls = LlavaForConditionalGeneration
    else: cls = AutoModelForVision2Seq

    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    return model, processor

def evaluate_single_file(model, processor, input_path, img_dir, out_file, task_type):
    n_rows_def = int(re.search(r"_(\d+)rows", os.path.basename(input_path)).group(1)) if re.search(r"_(\d+)rows", input_path) else None
    is_qwen = "qwen3-vl" in getattr(model, "name_or_path", "").lower()
    is_pixtral = "pixtral" in getattr(model, "name_or_path", "").lower()

    total, correct = 0, 0
    with open(input_path, "r") as f_in, open(out_file, "w") as f_out:
        for line in tqdm(f_in, desc=f"Task: {task_type}"):
            sample = json.loads(line)
            img_path = os.path.join(img_dir, sample["image"])
            if not os.path.exists(img_path): continue

            # Construct Inputs
            if is_pixtral:
                prompt = processor.apply_chat_template([{"role":"user","content":[{"type":"text","content":sample["prompt"]},{"type":"image"}]}], tokenize=False, add_generation_prompt=True)
                inputs = processor(text=prompt, images=Image.open(img_path).convert("RGB"), return_tensors="pt").to("cuda")
            elif is_qwen and process_vision_info:
                msgs = [{"role":"user","content":[{"type":"image","image":img_path},{"type":"text","text":sample["prompt"]}]}]
                prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                i_in, v_in = process_vision_info(msgs)
                inputs = processor(text=[prompt], images=i_in, videos=v_in, return_tensors="pt", padding=True).to("cuda")
            else:
                msgs = [{"role":"user","content":[{"type":"image","image":Image.open(img_path).convert("RGB")},{"type":"text","text":sample["prompt"]}]}]
                inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to("cuda")

            inputs = truncate_model_inputs_inplace(inputs, MAX_CONTEXT_LEN)
            in_len = inputs["input_ids"].shape[1]
            
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=min(MAX_NEW_TOKENS_CAP, MAX_CONTEXT_LEN - in_len))
            
            resp = processor.decode(out[0, in_len:], skip_special_tokens=True).strip()
            
            # Parsing
            if task_type == "loc":
                pred = parse_predicted_row_id(resp, n_rows=sample.get("meta",{}).get("n_rows", n_rows_def))
                matched = (pred == sample["ground_truth"]["answer"][0])
            else:
                pred = parse_predicted_attr_answer(resp)
                tol = 0.0001 if task_type == "mean" else 0.0
                matched = is_numeric_like_answer_correct(sample["ground_truth"]["answer"][0], pred, rel_tol=tol)

            total += 1
            if matched: correct += 1
            f_out.write(json.dumps({"id":sample["id"], "is_correct":matched, "pred":pred, "raw":resp}, ensure_ascii=False)+"\n")

    return {"total": total, "correct": correct, "accuracy": correct/total if total > 0 else 0}

# ============================================================
# 3. Functional Interface (API)
# ============================================================

def evaluate_dvm_benchmark(model_id: str, task_types: list = None, output_dir: str = "."):
    """
    Main entry point to evaluate the DVM benchmark.
    :param model_id: HuggingFace model path or ID.
    :param task_types: List of tasks to run (e.g., ['loc', 'attr', 'count', 'mean']). If None, runs all.
    :param output_dir: Path to save results. Defaults to current directory.
    """
    DATA_DIR = str(CURRENT_FILE_DIR / "stage1")
    IMAGE_BASE_DIR = str(CURRENT_FILE_DIR)
    
    if task_types is None:
        task_types = ["loc", "attr", "count", "mean"]

    # Initialize results folder
    model_name_clean = model_id.split("/")[-1]
    base_output = os.path.join(output_dir, f"eval_results_{model_name_clean}")
    os.makedirs(base_output, exist_ok=True)

    # Load Model once
    model, processor = load_model_and_processor(model_id)
    
    # Setup Padding
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    processor.tokenizer.pad_token_id = pad_id
    model.config.pad_token_id = pad_id

    final_summary = {}

    for task in task_types:
        task_upper = task.upper()
        task_path = os.path.join(base_output, task_upper)
        os.makedirs(task_path, exist_ok=True)
        
        files = sorted(Path(DATA_DIR).glob(f"dvm_*_{task.lower()}_*rows.jsonl"))
        if not files: continue
        
        task_results = []
        for f in files:
            out_name = os.path.join(task_path, f.name.replace(".jsonl", ".results.jsonl"))
            res = evaluate_single_file(model, processor, str(f), IMAGE_BASE_DIR, out_name, task.lower())
            task_results.append(res)
        
        # Aggregate
        total_q = sum(r["total"] for r in task_results)
        total_c = sum(r["correct"] for r in task_results)
        acc = total_c / total_q if total_q > 0 else 0
        
        final_summary[task_upper] = {"accuracy": acc, "total": total_q}
        print(f">>> Task {task_upper} completed. Accuracy: {acc*100:.2f}%")

    # Save final JSON summary
    with open(os.path.join(base_output, "overall_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\nAll evaluations finished. Results saved to: {os.path.abspath(base_output)}")
    return final_summary

if __name__ == "__main__":
    # Example usage
    # python script.py --model_id Qwen/Qwen3-VL-8B-Instruct --tasks loc attr
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", default=None, help="e.g. loc attr count mean")
    args = parser.parse_args()
    
    evaluate_dvm_benchmark(args.model_id, task_types=args.tasks)


# use
# from your_script_name import evaluate_dvm_benchmark
# results = evaluate_dvm_benchmark("Qwen/Qwen3-VL-8B-Instruct", task_types=["loc"])