import os
import json
import re
import time
import base64
import mimetypes
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, deque
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

# ==========================================
# 0. Path Configuration
# ==========================================
# /mnt/hdd/jiazy/ehrxqa -> Parent of current directory
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
# /mnt/hdd/jiazy/ehrxqa/stage2 -> Current directory
CURRENT_DIR = os.getcwd()

# ==========================================
# 1. Rate Limiter (RPM & TPM control)
# ==========================================
class OpenAIRateLimiter:
    """
    Sliding window rate limiter to manage Requests Per Minute (RPM) 
    and Tokens Per Minute (TPM) for OpenAI API.
    """
    def __init__(self, max_rpm, max_tpm):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.history = deque()
        # Approximate cost for high-res images in GPT-4o
        self.IMG_TOKEN_COST = 765 

    def estimate_tokens(self, text, num_images):
        """Roughly estimate Input Tokens for pre-judging."""
        text_tokens = len(text) // 4
        image_tokens = num_images * self.IMG_TOKEN_COST
        return text_tokens + image_tokens

    def wait_if_needed(self, estimated_tokens):
        """Block and sleep if current RPM or TPM exceeds limits."""
        while True:
            now = time.time()
            # 1. Clear records older than 60 seconds
            while self.history and self.history[0][0] < now - 60:
                self.history.popleft()

            # 2. Calculate current usage
            current_rpm = len(self.history)
            current_tpm = sum(req[1] for req in self.history)

            # 3. Check thresholds
            rpm_exceeded = current_rpm >= self.max_rpm
            tpm_exceeded = (current_tpm + estimated_tokens) > self.max_tpm

            if not rpm_exceeded and not tpm_exceeded:
                break 

            # 4. Calculate wait time based on the oldest request
            if self.history:
                oldest_time = self.history[0][0]
                wait_time = 60 - (now - oldest_time) + 0.5
            else:
                wait_time = 1

            if wait_time > 0:
                print(f"[LIMIT] Limits reached (RPM: {current_rpm}/{self.max_rpm}, TPM: {current_tpm}/{self.max_tpm}). Sleeping {wait_time:.2f}s...")
                time.sleep(wait_time)

    def update(self, actual_tokens):
        """Record the actual token consumption after a successful request."""
        self.history.append((time.time(), actual_tokens))

# ==========================================
# 2. Helper Functions (Normalization & Utils)
# ==========================================
def configure_openai(api_key: str | None):
    """Initialize OpenAI Client."""
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("[ERROR] OpenAI API Key is missing.")
    return OpenAI(api_key=api_key)

def image_path_to_data_url(path: str) -> str:
    """Convert local image path to base64 data URL for API payload."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def normalize_single_value(val):
    if isinstance(val, (int, float)):
        if float(val).is_integer():
            return int(val)
        return round(float(val), 6)
    if isinstance(val, str):
        v_str = val.strip().lower()
        if v_str in ["yes", "true", "correct"]: return 1
        if v_str in ["no", "false", "wrong", "incorrect"]: return 0
        try:
            f_val = float(v_str)
            if f_val.is_integer(): return int(f_val)
            return round(f_val, 6)
        except ValueError: return v_str
    return str(val) if isinstance(val, (list, dict, set, tuple)) else val

def compare_answers(gold_list, pred_list):
    if not isinstance(gold_list, list): gold_list = [gold_list]
    if not isinstance(pred_list, list): pred_list = [pred_list]
    g_norm = [normalize_single_value(x) for x in gold_list]
    p_norm = [normalize_single_value(x) for x in pred_list]
    try:
        g_set = sorted(list(set(g_norm)), key=str)
        p_set = sorted(list(set(p_norm)), key=str)
    except:
        g_set, p_set = str(g_norm), str(p_norm)
    return g_set == p_set

def load_processed_ids_from_jsonl(jsonl_path: str) -> set:
    processed = set()
    if not os.path.exists(jsonl_path): return processed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "id" in obj: processed.add(obj["id"])
            except: continue
    return processed

def load_study_to_image_map(tb_cxr_csv_paths):
    dfs = []
    for p in tb_cxr_csv_paths:
        if os.path.exists(p): dfs.append(pd.read_csv(p))
    if not dfs: return {}
    df = pd.concat(dfs, ignore_index=True)
    study_to_images = defaultdict(list)
    for _, row in df.iterrows():
        study_to_images[int(row["study_id"])].append((int(row["subject_id"]), str(row["image_id"])))
    print(f"[INFO] Loaded image map for {len(study_to_images)} studies.")
    return study_to_images

# ==========================================
# 3. Main Evaluation Function
# ==========================================
def run_openai_stage2_eval():
    """
    Main evaluation interface. Accepts no arguments.
    Paths and settings are determined by internal logic.
    """
    # Internal Config
    CONFIG = {
        "model_name": "gpt-4o", # Map your 'gpt-4.1' here
        "api_key": None, # Will read from environment variable
        "input_prompts": os.path.join(CURRENT_DIR, "split_prompts.json"),
        "original_test_json": os.path.join(BASE_DIR, "dataset/mimic_iv_cxr/test.json"),
        "mimic_cxr_root": os.path.join(BASE_DIR, "physionet.org/files/mimic-cxr-jpg/2.0.0"),
        "db_root": os.path.join(BASE_DIR, "database/mimic_iv_cxr"),
        "output_file": os.path.join(CURRENT_DIR, "step2_gpt.jsonl"),
        "max_rpm": 500,
        "max_tpm": 300000,
        "max_context_images": 10,
        "fsync": True
    }

    limiter = OpenAIRateLimiter(max_rpm=CONFIG["max_rpm"], max_tpm=CONFIG["max_tpm"])
    
    # Load Image Mappings
    tb_paths = [
        os.path.join(CONFIG["db_root"], "train", "tb_cxr.csv"),
        os.path.join(CONFIG["db_root"], "test", "tb_cxr.csv")
    ]
    study_to_images = load_study_to_image_map(tb_paths)

    # Load Prompts and Ground Truth
    with open(CONFIG["input_prompts"], "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    with open(CONFIG["original_test_json"], "r", encoding="utf-8") as f:
        id2gt = {item["id"]: item.get("answer", []) for item in json.load(f)}

    client = configure_openai(CONFIG["api_key"])
    
    # Resume Logic
    processed_ids = load_processed_ids_from_jsonl(CONFIG["output_file"])
    eval_data = [item for item in prompts_data if item["id"] not in processed_ids]
    print(f"[INFO] Evaluating {len(eval_data)} samples. Skipped {len(processed_ids)}.")

    if not eval_data:
        print("[INFO] No samples left to process.")
        return

    stats = {"total": 0, "correct": 0, "errors": 0}

    # Main Processing Loop
    for item in tqdm(eval_data, desc="OpenAI Eval"):
        stats["total"] += 1
        sid = item["id"]
        prompt_text = item["prompt"]
        gold_ans = id2gt.get(sid, [])
        
        current_res = {
            "id": sid,
            "status": "success",
            "is_correct": False,
            "pred_raw": []
        }

        # Handle Visual Input
        image_urls = []
        if item.get("type") == "vqa":
            vqa_calls = item.get("vqa_meta", {}).get("vqa_calls", [])
            all_sids = set()
            for call in vqa_calls:
                all_sids.update(call.get("study_ids_for_images", []))
            
            if len(all_sids) > CONFIG["max_context_images"]:
                current_res["status"] = "overflow"
                with open(CONFIG["output_file"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(current_res) + "\n")
                continue

            unique_sids = sorted(list(all_sids))
            for raw_sid in unique_sids:
                if int(raw_sid) in study_to_images:
                    for sub, img in study_to_images[int(raw_sid)]:
                        p = os.path.join(CONFIG["mimic_cxr_root"], "files", f"p{str(sub)[:2]}", f"p{sub}", f"s{raw_sid}", f"{img}.jpg")
                        if os.path.exists(p):
                            image_urls.append(image_path_to_data_url(p))

        # API Request
        try:
            # Prepare Multi-modal Payload
            messages_content = [{"type": "text", "text": prompt_text}]
            for url in image_urls[:5]: # Cap images per request for safety
                messages_content.append({"type": "image_url", "image_url": {"url": url}})

            estimated = limiter.estimate_tokens(prompt_text, len(image_urls))
            limiter.wait_if_needed(estimated)

            response = client.chat.completions.create(
                model=CONFIG["model_name"],
                messages=[{"role": "user", "content": messages_content}],
                temperature=0.0,
                max_tokens=2048
            )

            output_text = response.choices[0].message.content.strip()
            current_res["raw_output"] = output_text
            limiter.update(response.usage.total_tokens)

            # Parsing Answer (JSON or Regex)
            pred_ans_raw = []
            clean_text = output_text.replace("```json", "").replace("```", "").strip()
            try:
                pred_obj = json.loads(clean_text)
                pred_ans_raw = pred_obj.get("answer", []) if isinstance(pred_obj, dict) else pred_obj
            except:
                match = re.search(r"\[.*?\]", clean_text, re.DOTALL)
                if match:
                    try: pred_ans_raw = json.loads(match.group(0).replace("'", '"'))
                    except: pass

            current_res["pred_raw"] = pred_ans_raw
            current_res["is_correct"] = compare_answers(gold_ans, pred_ans_raw)
            if current_res["is_correct"]: stats["correct"] += 1

        except Exception as e:
            print(f"[ERR] Sample {sid} failed: {e}")
            current_res.update({"status": "error", "error": str(e)})
            stats["errors"] += 1

        # Persistence
        with open(CONFIG["output_file"], "a", encoding="utf-8") as f:
            f.write(json.dumps(current_res, ensure_ascii=False) + "\n")
            if CONFIG["fsync"]: os.fsync(f.fileno())

    print(f"\nEvaluation Complete. Accuracy: {stats['correct']/stats['total']:.4f}")

if __name__ == "__main__":
    run_openai_stage2_eval()