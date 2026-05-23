#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EHRXQA + Multi-Modal Evaluation Script (Hugging Face Version)
Integrated Factory for: InternVL, Qwen-VL, Llama-Vision, and GLM-4V.
"""

import os
import json
import re
import gc
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

# Core Transformers components
from transformers import (
    AutoConfig,
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoModelForVision2Seq,
    AutoModel,
    Qwen2VLForConditionalGeneration
)

# Optional model-specific classes
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import Glm4vForConditionalGeneration
except ImportError:
    Glm4vForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

# ==========================================
# 0. Global Path Configuration
# ==========================================
# Identifies /mnt/hdd/jiazy/ehrxqa (the parent of current directory)
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

# ==========================================
# 1. Model Factory: Loader
# ==========================================

def load_model_and_processor(model_id: str):
    """
    Identifies the model architecture from the HF model_id and returns the 
    instantiated model, processor, and an internal architecture label.
    """
    path_lower = model_id.lower()
    print(f"[INFO] Detecting architecture for: {model_id}")

    # --- InternVL Architecture ---
    if "internvl" in path_lower:
        print("[INFO] Loading as InternVL...")
        model_class = AutoModelForImageTextToText if AutoModelForImageTextToText else AutoModelForVision2Seq
        model = model_class.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Handle Pad Token
        pad_id = 0
        if hasattr(processor, "tokenizer"):
            if processor.tokenizer.pad_token_id is not None: pad_id = processor.tokenizer.pad_token_id
            elif processor.tokenizer.eos_token_id is not None: pad_id = processor.tokenizer.eos_token_id
            processor.tokenizer.pad_token_id = pad_id
        if hasattr(model, "config"): model.config.pad_token_id = pad_id
        return model, processor, "internvl"

    # --- Llama 3.2 Vision ---
    elif "llama" in path_lower and "vision" in path_lower:
        print("[INFO] Loading as Llama-Vision...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, "llama"

    # --- GLM-4V ---
    elif "glm" in path_lower:
        print("[INFO] Loading as GLM architecture...")
        model_cls = Glm4vForConditionalGeneration or AutoModelForImageTextToText or AutoModelForCausalLM
        try:
            model = model_cls.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
            ).eval()
        except:
            model = AutoModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
            ).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor, "glm"

    # --- Qwen / Pixtral ---
    elif "qwen" in path_lower or "pixtral" in path_lower:
        if "qwen3" in path_lower and Qwen3VLForConditionalGeneration:
            model_cls = Qwen3VLForConditionalGeneration
        elif "qwen" in path_lower:
            model_cls = Qwen2VLForConditionalGeneration
        else:
            model_cls = AutoModelForVision2Seq # Pixtral
        
        print(f"[INFO] Loading as {model_cls.__name__}...")
        model = model_cls.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Fix pad_token issues for Pixtral/Mistral variants
        if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            model.config.pad_token_id = processor.tokenizer.pad_token_id
            
        return model, processor, "qwen"

    # --- Generic Fallback ---
    else:
        print("[WARN] Unknown architecture, falling back to AutoModelForVision2Seq...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor, "auto"

# ==========================================
# 2. Answer Comparison Logic
# ==========================================

def normalize_single_value(val):
    if isinstance(val, (int, float)):
        return int(val) if float(val).is_integer() else round(float(val), 6)
    if isinstance(val, str):
        v_str = val.strip().lower()
        if v_str in ["yes", "true", "correct"]: return 1
        if v_str in ["no", "false", "wrong", "incorrect"]: return 0
        try:
            f_val = float(v_str)
            return int(f_val) if f_val.is_integer() else round(f_val, 6)
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

# ==========================================
# 3. Main Evaluation Interface
# ==========================================

def evaluate_hf_vllm(model_id: str):
    """
    Receives a full HF model_id, runs evaluation, and saves results to BASE_DIR/stage2.
    """
    # Dynamic Configuration based on BASE_DIR
    CONFIG = {
        "prompts_json": os.path.join(BASE_DIR, "stage2", "split_prompts.json"),
        "test_json": os.path.join(BASE_DIR, "dataset", "mimic_iv_cxr", "test.json"),
        "image_root": os.path.join(BASE_DIR, "physionet.org", "files", "mimic-cxr-jpg", "2.0.0"),
        "db_root": os.path.join(BASE_DIR, "database", "mimic_iv_cxr"),
        "output_file": os.path.join(BASE_DIR, "stage2", f"results_{model_id.split('/')[-1]}.json"),
        "max_context_images": 5,
        "max_new_tokens": 1024,
        "max_image_size": 512
    }

    # Handle Thinking models (Chain-of-Thought)
    if "thinking" in model_id.lower():
        print("[CONFIG] Thinking model detected. Scaling max_new_tokens to 2048.")
        CONFIG["max_new_tokens"] = 2048

    # 1. Loading Mapping Data
    tb_paths = [
        os.path.join(CONFIG["db_root"], "train", "tb_cxr.csv"),
        os.path.join(CONFIG["db_root"], "test", "tb_cxr.csv")
    ]
    
    # Load Study -> Image Map
    dfs = [pd.read_csv(p) for p in tb_paths if os.path.exists(p)]
    study_to_images = defaultdict(list)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        for _, row in df_all.iterrows():
            study_to_images[int(row["study_id"])].append((int(row["subject_id"]), str(row["image_id"])))

    # Load Prompts and Ground Truth
    with open(CONFIG["prompts_json"], "r") as f:
        eval_data = json.load(f)
    with open(CONFIG["test_json"], "r") as f:
        id2gt = {item["id"]: item.get("answer", []) for item in json.load(f)}

    # 2. Load Model & Processor
    model, processor, arch_type = load_model_and_processor(model_id)

    stats = {"total": 0, "correct": 0, "errors": 0}
    results_log = []

    # 3. Inference Loop
    for item in tqdm(eval_data, desc=f"Evaluating {arch_type.upper()}"):
        stats["total"] += 1
        sid = item["id"]
        prompt_text = item["prompt"]
        gold_ans = id2gt.get(sid, [])
        
        current_res = {
            "id": sid, "question": item["question"], 
            "gold_raw": gold_ans, "is_correct": False, "status": "success"
        }

        # --- A. Collect and Process Images ---
        loaded_images = []
        if item.get("type") == "vqa":
            all_sids = set()
            for call in item.get("vqa_meta", {}).get("vqa_calls", []):
                all_sids.update(call.get("study_ids_for_images", []))
            
            # Filter and Load images
            for target_sid in sorted(list(all_sids))[:CONFIG["max_context_images"]]:
                if target_sid not in study_to_images: continue
                for sub_id, img_id in study_to_images[target_sid]:
                    img_path = os.path.join(CONFIG["image_root"], "files", f"p{str(sub_id)[:2]}", f"p{sub_id}", f"s{target_sid}", f"{img_id}.jpg")
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert("RGB")
                            img.thumbnail((CONFIG["max_image_size"], CONFIG["max_image_size"]), Image.Resampling.LANCZOS)
                            loaded_images.append(img)
                        except: pass

        # --- B. Inference Logic by Architecture ---
        try:
            content = []
            for img in loaded_images: content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

            with torch.no_grad():
                if arch_type == 'internvl':
                    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
                    gen_ids = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], do_sample=False)
                    output_text = processor.decode(gen_ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                elif arch_type == 'llama':
                    txt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=txt, images=loaded_images if loaded_images else None, return_tensors="pt").to(model.device)
                    gen_ids = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], do_sample=False)
                    output_text = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

                elif arch_type in ['qwen', 'glm', 'auto']:
                    txt = processor.apply_chat_template(messages, tokenize=False if arch_type != 'glm' else True, add_generation_prompt=True)
                    inputs = processor(text=[txt] if arch_type != 'glm' else txt, images=loaded_images if loaded_images else None, return_tensors="pt").to(model.device)
                    gen_ids = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], do_sample=False)
                    # Handle slice for GLM vs Qwen
                    input_len = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else inputs['input_ids'].shape[1]
                    output_text = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0]

            output_text = output_text.strip()
            current_res["raw_output"] = output_text

            # --- C. Parsing and Comparison ---
            pred_ans_raw = []
            # Extract list from JSON block or Regex
            match = re.search(r"(\[[^\]]+\])", output_text)
            if match:
                try: pred_ans_raw = json.loads(match.group(1).replace("'", '"'))
                except: pass
            
            current_res["pred_raw"] = pred_ans_raw
            current_res["is_correct"] = compare_answers(gold_ans, pred_ans_raw)
            if current_res["is_correct"]: stats["correct"] += 1

        except Exception as e:
            stats["errors"] += 1
            current_res.update({"status": "error", "error": str(e)})

        results_log.append(current_res)

        # Periodic Cleanup
        del content, messages
        torch.cuda.empty_cache()
        gc.collect()

    # Final Report
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"\nEvaluation Report for {model_id}")
    print(f"Accuracy: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage:
    # evaluate_hf_vllm("OpenGVLab/InternVL2-8B")
    pass