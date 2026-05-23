import os
import sys
import json
import re
import time
import argparse
import base64
from collections import defaultdict, deque
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import (
    AutoModelForImageTextToText,
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration,
    Glm4vForConditionalGeneration,
)
import argparse
import json
import os
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)

# 某些特定模型可能需要显式导入
try:
    from transformers import Glm4vForConditionalGeneration
except ImportError:
    Glm4vForConditionalGeneration = None

# =========================
# 路径处理：当前文件夹的上一个文件夹
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

# =========================
# 全局配置
# =========================
CONTEXT_LEN = 1024

def _set_context_len(processor, model, context_len: int = CONTEXT_LEN):
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.model_max_length = context_len
        if hasattr(tok, "max_len_single_sentence"):
            tok.max_len_single_sentence = context_len
    if hasattr(processor, "model_max_length"):
        try: processor.model_max_length = context_len
        except: pass
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        try: model.config.max_position_embeddings = context_len
        except: pass
    if hasattr(model, "generation_config") and model.generation_config is not None:
        try: model.generation_config.max_length = context_len
        except: pass

def _ensure_pad_token(processor, model) -> int:
    pad_id = getattr(processor, "pad_token_id", None)
    tok = getattr(processor, "tokenizer", None)
    if pad_id is None and tok is not None:
        pad_id = getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", None))
    if pad_id is None and hasattr(model, "config"):
        pad_id = getattr(model.config, "eos_token_id", 0)
    if pad_id is None: pad_id = 0
    if tok is not None: tok.pad_token_id = pad_id
    if hasattr(processor, "pad_token_id"): processor.pad_token_id = pad_id
    if hasattr(model, "config"): model.config.pad_token_id = pad_id
    return pad_id

def _move_to_device(batch, device):
    if hasattr(batch, "to"): return batch.to(device)
    if isinstance(batch, torch.Tensor): return {"input_ids": batch}
    return batch

def _ensure_attention_mask(batch: Dict[str, Any], model) -> Dict[str, Any]:
    if "attention_mask" in batch: return batch
    input_ids = batch.get("input_ids")
    if input_ids is not None and torch.is_tensor(input_ids):
        pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
        attn = torch.ones_like(input_ids) if pad_id is None else (input_ids != pad_id).long()
        batch["attention_mask"] = attn.to(input_ids.device)
    return batch
from PIL import Image

# --- API Library Imports ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- Global Rate Limiting Configuration ---
TPM_LIMIT = int(os.getenv("OPENAI_TPM_LIMIT", "20000"))
SAFETY = 0.70
window = deque()

# 获取当前脚本文件所在的绝对路径目录
CURRENT_FILE_DIR = Path(__file__).resolve().parent

# ============================================================
# 1. Utility Functions (Parsing & Rate Limiting)
# ============================================================

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(gt))
    g_num = float(nums[0]) if nums else None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(pred))
    p_num = float(nums[0]) if nums else None
    if g_num is not None and p_num is not None:
        diff = abs(g_num - p_num)
        return diff <= abs_tol or (diff / max(abs(g_num), abs(p_num), 1e-8)) <= rel_tol
    return str(gt).strip().lower() == str(pred).strip().lower()

def throttle_openai(estimated_tokens):
    now = time.time()
    while window and now - window[0][0] > 60: window.popleft()
    used = sum(t for _, t in window)
    if used + estimated_tokens > TPM_LIMIT * SAFETY:
        wait = 60 - (now - window[0][0]) + 0.1
        time.sleep(max(wait, 1.0))

# ============================================================
# 2. Model Loader & Evaluation Engine
# ============================================================

def load_model_and_processor(model_id: str):
    print(f"--- Initializing: {model_id} ---")
    
    # 1. OpenAI
    if any(k in model_id.lower() for k in ["gpt", "o1"]):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY not found.")
        return OpenAI(api_key=api_key), None

    # 2. Gemini (New SDK)
    if "gemini" in model_id.lower():
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
        return genai.Client(api_key=api_key), None

    # 3. Local HuggingFace
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if "qwen3-vl" in model_id.lower(): cls = AutoModelForVision2Seq
    elif "internvl3" in model_id.lower(): cls = AutoModelForImageTextToText
    elif "llama-3.2" in model_id.lower(): cls = MllamaForConditionalGeneration
    elif "pixtral" in model_id.lower(): cls = LlavaForConditionalGeneration
    else: cls = AutoModelForVision2Seq

    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    return model, processor

def evaluate_single_file(model, processor, input_path, img_dir, out_file, model_id, task_type):
    # Resume Logic
    processed_ids = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for line in f:
                try: processed_ids.add(json.loads(line)["id"])
                except: continue

    is_api = processor is None
    is_gemini = is_api and "gemini" in model_id.lower()
    is_openai = is_api and not is_gemini
    
    n_rows_def = int(re.search(r"_(\d+)rows", input_path).group(1)) if re.search(r"_(\d+)rows", input_path) else None
    
    total, correct = 0, 0
    with open(input_path, "r") as f_in, open(out_file, "a") as f_out:
        lines = f_in.readlines()
        for line in tqdm(lines, desc=f"Evaluating {task_type.upper()}"):
            sample = json.loads(line)
            if sample["id"] in processed_ids: continue

            img_path = os.path.join(img_dir, sample["image"])
            if not os.path.exists(img_path): continue

            resp_text = ""
            # --- API Logic ---
            if is_openai:
                throttle_openai(1000)
                base64_img = encode_image_to_base64(img_path)
                resp = model.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": [{"type": "text", "text": sample["prompt"]}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]}],
                    temperature=0.0
                )
                resp_text = resp.choices[0].message.content
                window.append((time.time(), resp.usage.total_tokens))

            elif is_gemini:
                resp = model.models.generate_content(model=model_id, contents=[sample["prompt"], Image.open(img_path)])
                resp_text = resp.text if resp.text else "[Safety Blocked]"

            # --- Local HF Logic ---
            else:
                inputs = processor.apply_chat_template([{"role":"user","content":[{"type":"image","image":Image.open(img_path).convert("RGB")},{"type":"text","text":sample["prompt"]}]}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=128)
                resp_text = processor.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # --- Parsing ---
            if task_type == "loc":
                pred = parse_predicted_row_id(resp_text, n_rows=sample.get("meta",{}).get("n_rows", n_rows_def))
                matched = (pred == sample["ground_truth"]["answer"][0])
            else:
                pred = parse_predicted_attr_answer(resp_text)
                tol = 0.0001 if task_type == "mean" else 0.0
                matched = is_numeric_like_answer_correct(sample["ground_truth"]["answer"][0], pred, rel_tol=tol)

            total += 1
            if matched: correct += 1
            f_out.write(json.dumps({"id": sample["id"], "is_correct": matched, "pred": pred, "raw": resp_text}, ensure_ascii=False)+"\n")
            f_out.flush()

    return {"total": total, "correct": correct}

# ============================================================
# 3. Main Functional Interface (API)
# ============================================================

def run_dvm_benchmark_api(model_id: str, tasks: list = None, output_root: str = "."):
    """
    Programmatic interface for the DVM benchmark across API and Local models.
    :param model_id: Path to HF model OR OpenAI/Gemini model name string.
    :param tasks: List of tasks to run (e.g. ['loc', 'attr']). Runs all if None.
    :param output_root: Where to save results. Defaults to current directory.
    """
    DATA_DIR = str(CURRENT_FILE_DIR / "stage1")
    IMAGE_BASE_DIR = str(CURRENT_FILE_DIR)
    
    if tasks is None: tasks = ["loc", "attr", "count", "mean"]
    
    # Create output structure
    clean_name = model_id.split("/")[-1].replace(":", "_")
    output_dir = os.path.join(output_root, f"eval_api_{clean_name}")
    os.makedirs(output_dir, exist_ok=True)

    model, processor = load_model_and_processor(model_id)
    if processor: # Set padding for local models
        model.config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0

    overall_stats = {}
    for task in tasks:
        task_out = os.path.join(output_dir, task.upper())
        os.makedirs(task_out, exist_ok=True)
        
        files = sorted(Path(DATA_DIR).glob(f"dvm_*_{task.lower()}_*rows.jsonl"))
        if not files: continue
        
        task_correct, task_total = 0, 0
        for f_path in files:
            res_file = os.path.join(task_out, f_path.name.replace(".jsonl", ".results.jsonl"))
            stats = evaluate_single_file(model, processor, str(f_path), IMAGE_BASE_DIR, res_file, model_id, task.lower())
            task_total += stats["total"]
            task_correct += stats["correct"]
        
        acc = task_correct / task_total if task_total > 0 else 0
        overall_stats[task] = {"accuracy": acc, "total_samples": task_total}
        print(f"Task {task.upper()} Accuracy: {acc*100:.2f}%")

    with open(os.path.join(output_dir, "final_summary.json"), "w") as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"Evaluation complete. Results: {os.path.abspath(output_dir)}")
    return overall_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="e.g. 'gpt-4o' or 'Qwen/Qwen2-VL-7B-Instruct'")
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()
    
    run_dvm_benchmark_api(args.model_id, tasks=args.tasks)