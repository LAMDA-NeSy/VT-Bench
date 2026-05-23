import os
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, deque
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from google.api_core import retry

# ==========================================
# 0. Global Path Configuration
# ==========================================
# /mnt/hdd/jiazy/ehrxqa -> Parent of current directory
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
# /mnt/hdd/jiazy/ehrxqa/stage2 -> Current directory
CURRENT_DIR = os.getcwd()

# ==========================================
# 1. Rate Limiter (RPM & TPM control)
# ==========================================
class GeminiRateLimiter:
    """
    Simple sliding window rate limiter to manage Requests Per Minute (RPM) 
    and Tokens Per Minute (TPM) for Gemini API.
    """
    def __init__(self, max_rpm, max_tpm):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.history = deque()
        self.IMG_TOKEN_COST = 260 # Approximate cost per image for Gemini 1.5/2.0

    def estimate_tokens(self, text, num_images):
        text_tokens = len(text) // 4
        image_tokens = num_images * self.IMG_TOKEN_COST
        return text_tokens + image_tokens

    def wait_if_needed(self, estimated_tokens):
        while True:
            now = time.time()
            # Remove records older than 60 seconds
            while self.history and self.history[0][0] < now - 60:
                self.history.popleft()

            current_rpm = len(self.history)
            current_tpm = sum(req[1] for req in self.history)

            rpm_exceeded = current_rpm >= self.max_rpm
            tpm_exceeded = (current_tpm + estimated_tokens) > self.max_tpm

            if not rpm_exceeded and not tpm_exceeded:
                break

            # Calculate sleep time based on the oldest request in the window
            if self.history:
                oldest_time = self.history[0][0]
                wait_time = 60 - (now - oldest_time) + 0.5
            else:
                wait_time = 1

            if wait_time > 0:
                print(f"[LIMIT] Reached limits (RPM: {current_rpm}/{self.max_rpm}, TPM: {current_tpm}/{self.max_tpm}). Sleeping {wait_time:.2f}s...")
                time.sleep(wait_time)

    def update(self, actual_tokens):
        self.history.append((time.time(), int(actual_tokens)))

# ==========================================
# 2. Helper Functions (Normalization & Logic)
# ==========================================
def configure_gemini(api_key, model_name):
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("[ERROR] API Key is missing. Set GOOGLE_API_KEY env var.")

    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.0,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    }

    print(f"[INFO] Initializing Gemini model: {model_name}")
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )
    return model

def normalize_single_value(val):
    """Normalize predicted or gold standard values for uniform comparison."""
    if isinstance(val, (int, float)):
        if float(val).is_integer():
            return int(val)
        return round(float(val), 6)
    if isinstance(val, str):
        v_str = val.strip().lower()
        if v_str in ["yes", "true", "correct"]:
            return 1
        if v_str in ["no", "false", "wrong", "incorrect"]:
            return 0
        try:
            f_val = float(v_str)
            if f_val.is_integer():
                return int(f_val)
            return round(f_val, 6)
        except ValueError:
            return v_str
    return str(val) if isinstance(val, (list, dict, set, tuple)) else val

def compare_answers(gold_list, pred_list):
    """Compare normalized sets of answers."""
    if not isinstance(gold_list, list): gold_list = [gold_list]
    if not isinstance(pred_list, list): pred_list = [pred_list]
    
    g_norm = [normalize_single_value(x) for x in gold_list]
    p_norm = [normalize_single_value(x) for x in pred_list]
    
    try:
        g_set = sorted(list(set(g_norm)), key=str)
        p_set = sorted(list(set(p_norm)), key=str)
    except Exception:
        g_set, p_set = str(g_norm), str(p_norm)
    return g_set == p_set

def load_study_to_image_map(tb_cxr_csv_paths):
    dfs = []
    for p in tb_cxr_csv_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs: return {}
    
    df = pd.concat(dfs, ignore_index=True)
    study_to_images = defaultdict(list)
    for _, row in df.iterrows():
        study_to_images[int(row["study_id"])].append((int(row["subject_id"]), str(row["image_id"])))
    print(f"[INFO] Loaded image map for {len(study_to_images)} studies.")
    return study_to_images

def get_image_paths(study_ids, study_to_images, mimic_cxr_root, max_images_total=5):
    paths = []
    unique_sids = sorted(list(set(study_ids)))
    for raw_sid in unique_sids:
        try:
            sid = int(raw_sid)
            if sid not in study_to_images: continue
            for sub_id, img_id in study_to_images[sid]:
                # Mimic-CXR folder structure: files/pXX/pXXXXXX/sYYYYYY/image.jpg
                p = os.path.join(mimic_cxr_root, "files", f"p{str(sub_id)[:2]}", f"p{sub_id}", f"s{sid}", f"{img_id}.jpg")
                if os.path.exists(p):
                    paths.append(p)
        except: continue
    return paths[:max_images_total]

def load_processed_ids(output_file):
    """Checks for existing records to support resuming evaluation after interruption."""
    processed_ids = set()
    if not os.path.exists(output_file):
        return processed_ids
    
    print(f"[RESUME] Found existing output file: {output_file}")
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and s.startswith("{"):
                    obj = json.loads(s)
                    if "id" in obj: processed_ids.add(obj["id"])
        print(f"[RESUME] Loaded {len(processed_ids)} processed IDs.")
    except Exception as e:
        print(f"[WARN] Failed to load resume data: {e}")
    return processed_ids

# ==========================================
# 3. Main Function Interface
# ==========================================
def run_gemini_stage2_eval():
    """
    Main evaluation interface. No arguments required.
    All settings and paths are defined internally using relative path logic.
    """
    # Internal Config
    CONFIG = {
        "model_name": "gemini-1.5-flash",
        "input_prompts": os.path.join(CURRENT_DIR, "split_prompts.json"),
        "original_test_json": os.path.join(BASE_DIR, "dataset/mimic_iv_cxr/test.json"),
        "mimic_cxr_root": os.path.join(BASE_DIR, "physionet.org/files/mimic-cxr-jpg/2.0.0"),
        "db_root": os.path.join(BASE_DIR, "database/mimic_iv_cxr"),
        "output_file": os.path.join(CURRENT_DIR, "step2_gemini_flash.jsonl"),
        "max_rpm": 10,
        "max_tpm": 500000,
        "max_context_images": 10,
        "max_image_size": 1024,
        "api_key": None # Fallback to GOOGLE_API_KEY environment variable
    }

    limiter = GeminiRateLimiter(max_rpm=CONFIG["max_rpm"], max_tpm=CONFIG["max_tpm"])

    # Load Database and Image Map
    tb_paths = [
        os.path.join(CONFIG["db_root"], "train", "tb_cxr.csv"),
        os.path.join(CONFIG["db_root"], "test", "tb_cxr.csv")
    ]
    study_to_images = load_study_to_image_map(tb_paths)

    # Load Inputs
    with open(CONFIG["input_prompts"], "r") as f:
        prompts_data = json.load(f)
    with open(CONFIG["original_test_json"], "r") as f:
        id2gt = {item["id"]: item.get("answer", []) for item in json.load(f)}

    # Initialize Gemini
    model = configure_gemini(CONFIG["api_key"], CONFIG["model_name"])

    # Handle Resume Logic
    processed_ids = load_processed_ids(CONFIG["output_file"])
    eval_data = [item for item in prompts_data if item["id"] not in processed_ids]
    
    print(f"[INFO] Evaluating {len(eval_data)} samples. (Skipped {len(processed_ids)} already done).")
    if not eval_data:
        print("[INFO] All samples processed. Exiting.")
        return

    stats = {"total": 0, "errors": 0, "correct": 0}
    
    # Open file for appending
    with open(CONFIG["output_file"], "a", encoding="utf-8") as out_fp:
        try:
            for item in tqdm(eval_data, desc="Gemini Eval"):
                stats["total"] += 1
                sid = item["id"]
                gold_ans = id2gt.get(sid, [])
                
                res = {
                    "id": sid,
                    "is_correct": False,
                    "status": "success",
                    "pred_raw": []
                }

                # Image Gathering
                image_paths = []
                if item.get("type") == "vqa":
                    vqa_calls = item.get("vqa_meta", {}).get("vqa_calls", [])
                    all_sids = set()
                    for call in vqa_calls:
                        all_sids.update(call.get("study_ids_for_images", []))
                    
                    if len(all_sids) > CONFIG["max_context_images"]:
                        res["status"] = "overflow"
                        out_fp.write(json.dumps(res) + "\n")
                        continue
                    image_paths = get_image_paths(list(all_sids), study_to_images, CONFIG["mimic_cxr_root"])

                # Multimodal Prompt Construction
                content_parts = [item["prompt"]]
                loaded_imgs = []
                for p in image_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        if CONFIG["max_image_size"] > 0:
                            img.thumbnail((CONFIG["max_image_size"], CONFIG["max_image_size"]), Image.Resampling.LANCZOS)
                        content_parts.append(img)
                        loaded_imgs.append(img)
                    except: pass

                # Inference with Rate Limiting
                try:
                    estimated = limiter.estimate_tokens(item["prompt"], len(loaded_imgs))
                    limiter.wait_if_needed(estimated)

                    response = model.generate_content(
                        content_parts,
                        request_options={"retry": retry.Retry(predicate=retry.if_transient_error)}
                    )

                    # Safe Text Extraction
                    output_text = ""
                    cands = getattr(response, "candidates", None) or []
                    if cands:
                        parts = getattr(cands[0].content, "parts", None) or []
                        output_text = "".join([getattr(p, "text", "") for p in parts]).strip()
                    
                    res["raw_output"] = output_text
                except Exception as e:
                    print(f"[ERR] Inference failed for {sid}: {e}")
                    res.update({"status": "api_error", "error": str(e)})
                    out_fp.write(json.dumps(res) + "\n")
                    stats["errors"] += 1
                    continue

                # Parsing and Comparison
                pred_ans_raw = []
                clean_text = output_text.replace("```json", "").replace("```", "").strip()
                try:
                    # Attempt JSON parsing
                    pred_obj = json.loads(clean_text)
                    if isinstance(pred_obj, dict): pred_ans_raw = pred_obj.get("answer", [])
                    elif isinstance(pred_obj, list): pred_ans_raw = pred_obj
                except:
                    # Fallback to Regex extraction of list formats
                    match = re.search(r"(\[[^\]]+\])", clean_text)
                    if match:
                        try: pred_ans_raw = json.loads(match.group(1).replace("'", '"'))
                        except: pass

                res["pred_raw"] = pred_ans_raw
                res["is_correct"] = compare_answers(gold_ans, pred_ans_raw)
                if res["is_correct"]: stats["correct"] += 1
                
                # Immediate write to ensure persistence
                out_fp.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_fp.flush()

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Evaluation paused. Progress saved.")
    
    # Final Summary
    print(f"\n{'='*50}\n EVALUATION SUMMARY\n{'='*50}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Total Processed: {stats['total']} | Errors: {stats['errors']}")
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"Accuracy: {acc:.4f} ({stats['correct']}/{stats['total']})")
    print("=" * 50)

if __name__ == "__main__":
    run_gemini_stage2_eval()