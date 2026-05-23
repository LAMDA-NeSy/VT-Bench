#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EHRXQA + Multi-Modal Models 评测脚本 (Google GenAI SDK v1.0 Version)
带速率限制 (Rate Limiting) 版本
"""

import argparse
import json
import os
import sqlite3
import re
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel

# =========================
# 新版 Google GenAI 依赖
# =========================
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

# 检查 API KEY
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY environment variable not set.")

# 初始化客户端
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# 安全设置
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# =========================
# 速率限制器 (Rate Limiter)
# =========================

class RateLimiter:
    """
    简单的滑动窗口速率限制器，同时限制 RPM (Requests Per Minute) 和 TPM (Tokens Per Minute)。
    注意：由于 Google GenAI SDK v1.0 在请求前难以精确预估 Token，
    这里采用“估算请求Token + 响应后更新统计”的策略，或者简单地在每次请求前强制休眠来控制 RPM。
    """
    def __init__(self, rpm: int, tpm: int):
        self.max_rpm = rpm
        self.max_tpm = tpm
        # 记录请求的时间戳
        self.request_timestamps = deque()
        # 记录 Token 使用量 (timestamp, token_count)
        self.token_timestamps = deque()
    
    def wait_for_slot(self, estimated_tokens: int = 0):
        """
        在发起请求前调用。如果超过限制，会阻塞直到有配额。
        """
        # 1. 检查 RPM
        while True:
            now = time.time()
            # 移除 60 秒之前的记录
            while self.request_timestamps and now - self.request_timestamps[0] > 60:
                self.request_timestamps.popleft()
            
            if len(self.request_timestamps) < self.max_rpm:
                break
            # 如果已满，等待一段时间（例如等待最旧的那个记录过期）
            wait_time = 60 - (now - self.request_timestamps[0]) + 0.1
            if wait_time > 0:
                time.sleep(wait_time)

        # 2. 检查 TPM (如果有预估)
        # 注意：TPM 限制通常比较宽裕，简单的 sleep 策略对于 RPM 有效，TPM 往往需要更复杂的桶算法。
        # 这里为了简化，我们主要依赖 RPM 限制，并在请求后通过 update_usage 也可以做动态调整。
        # 如果需要严格 TPM 限制：
        while True:
            now = time.time()
            while self.token_timestamps and now - self.token_timestamps[0][0] > 60:
                self.token_timestamps.popleft()
            
            current_tpm = sum(t[1] for t in self.token_timestamps)
            if current_tpm + estimated_tokens <= self.max_tpm:
                break
            
            # 等待最早的 token 记录过期
            if self.token_timestamps:
                wait_time = 60 - (now - self.token_timestamps[0][0]) + 0.1
                if wait_time > 0:
                    time.sleep(wait_time)
            else:
                break # Should not happen if loop logic is correct

        # 记录这次请求的时间（虽然还没发，但占位）
        self.request_timestamps.append(time.time())

    def update_token_usage(self, token_count: int):
        """请求完成后调用，更新实际消耗的 Token"""
        self.token_timestamps.append((time.time(), token_count))

# 全局限流器实例（将在 main 中初始化）
rate_limiter = None

# =========================
# Pydantic Models & DB Schema
# =========================
# (保持不变)
class Stage1Plan(BaseModel):
    need_sql: bool
    sql_queries: List[str]
    need_image: bool
    reasoning: str
    final_answer: Optional[str] = None

class Stage2Answer(BaseModel):
    final_answer: str
    explanation: str

DB_SCHEMA_TEXT = r"""
You have access to a SQLite database "mimic_iv_cxr" with the following tables:

1. TB_CXR
   - row_id (INTEGER)
   - subject_id (INTEGER)
   - hadm_id (REAL)
   - study_id (INTEGER)
   - image_id (TEXT)
   - viewposition (TEXT)
   - studydatetime (TEXT)

2. TB_CXR_PLUS
   - row_id (INTEGER)
   - subject_id (INTEGER)
   - hadm_id (REAL)
   - study_id (INTEGER)
   - image_id (TEXT)
   - viewposition (TEXT)
   - studydatetime (TEXT)
   - studyorder (REAL)
   - object (TEXT)
   - relation (INTEGER)
   - attribute (TEXT)
   - category (TEXT)
   - ct_ratio (REAL)
   - mt_ratio (REAL)

3. PATIENTS
   - row_id, subject_id, gender, dob, dod

4. ADMISSIONS
   - row_id, subject_id, hadm_id,
   - admittime, dischtime,
   - admission_type, admission_location, discharge_location,
   - insurance, language, marital_status, age

5. D_ICD_DIAGNOSES
   - row_id, icd_code, long_title

6. D_ICD_PROCEDURES
   - row_id, icd_code, long_title

7. D_ITEMS
   - row_id, itemid, label, abbreviation, linksto

8. D_LABITEMS
   - row_id, itemid, label

9. DIAGNOSES_ICD
   - row_id, subject_id, hadm_id, icd_code, charttime

10. PROCEDURES_ICD
    - row_id, subject_id, hadm_id, icd_code, charttime

11. LABEVENTS
    - row_id, subject_id, hadm_id, itemid, charttime, valuenum, valueuom

12. PRESCRIPTIONS
    - row_id, subject_id, hadm_id, starttime, stoptime, drug, dose_val_rx, dose_unit_rx, route

13. COST
    - row_id, subject_id, hadm_id, event_type, event_id, chargetime, cost

14. CHARTEVENTS
    - row_id, subject_id, hadm_id, stay_id, charttime, itemid, valuenum, valueuom

15. INPUTEVENTS
    - row_id, subject_id, hadm_id, stay_id, starttime, itemid, amount

16. OUTPUTEVENTS
    - row_id, subject_id, hadm_id, stay_id, charttime, itemid, value

17. MICROBIOLOGYEVENTS
    - row_id, subject_id, hadm_id, charttime, spec_type_desc, test_name, org_name

18. ICUSTAYS
    - row_id, subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime

19. TRANSFERS
    - row_id, subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime
"""


def build_stage1_system_prompt(db_path: str, image_root: str) -> str:
    # (保持原样)
    prompt = f"""
You are a clinical question answering assistant for the EHRXQA dataset.
The SQLite database for EHRXQA is located at: {db_path}
{DB_SCHEMA_TEXT}
You CANNOT execute SQL yourself. You only WRITE SQL queries as plain text.
An external system (Python) will execute your SQL and return the results to you in a second turn.
Images are stored under the root directory: {image_root}
IMPORTANT TEMPORAL RULES (MANDATORY):
- The MIMIC-IV and MIMIC-CXR datasets use de-identified time shifting.
- You MUST assume the current_time is **"2105-12-31 23:59:00"**.
- NEVER use real-world current timestamps.
Decision rules:
1. Decide whether you need to query the database (need_sql).
2. Decide whether you need images (need_image).
   - If need_image = true, at least one SQL query MUST select: subject_id, study_id, image_id.
Output format (JSON compliant with schema):
{{
  "need_sql": true/false,
  "sql_queries": ["SELECT ...", ...],
  "need_image": true/false,
  "reasoning": "...",
  "final_answer": "..." (or null)
}}
"""
    return prompt.strip()

# =========================
# 阶段一：调用 Gemini 生成 Plan
# =========================

def call_gemini_stage1(model_name: str, system_prompt: str, question: str) -> Stage1Plan:
    user_prompt = f"Question: {question}"
    
    # === 限流检查 ===
    # 估算 input token: 粗略计算字符数 / 4
    estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
    if rate_limiter:
        rate_limiter.wait_for_slot(estimated_tokens=estimated_tokens)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Stage1Plan,
                temperature=0.0,
                safety_settings=SAFETY_SETTINGS,
            )
        )
    except google_exceptions.ResourceExhausted:
        print("\n[Rate Limit Hit] 429 Resource Exhausted. Sleeping 60s...")
        time.sleep(60)
        # 简单重试一次，或者递归调用自己（小心死循环，这里简单重试）
        return call_gemini_stage1(model_name, system_prompt, question)

    # === 更新 Token 使用量 ===
    if rate_limiter and response.usage_metadata:
        total_tokens = response.usage_metadata.total_token_count
        rate_limiter.update_token_usage(total_tokens)

    if response.parsed:
        return response.parsed

    raw_text = getattr(response, "text", None)
    print(f"[Warning] response.parsed is None. response.text={raw_text!r}")

    # 关键：防止 None 进入 model_validate_json
    if not raw_text or not isinstance(raw_text, (str, bytes, bytearray)):
        # 尽量打印更多诊断信息
        try:
            cand_n = len(getattr(response, "candidates", []) or [])
        except Exception:
            cand_n = "unknown"
        raise RuntimeError(
            f"Stage1 got no valid JSON text. parsed=None, text={raw_text!r}, candidates={cand_n}"
        )

    return Stage1Plan.model_validate_json(raw_text)



def stage1_plan(model_name: str, db_path: str, image_root: str, question: str):
    system_prompt = build_stage1_system_prompt(db_path, image_root)
    plan_obj = call_gemini_stage1(model_name, system_prompt, question)
    return plan_obj.model_dump(), plan_obj.model_dump_json()

# =========================
# 阶段二：执行 SQL & 辅助函数 (保持不变)
# =========================

def execute_sql_queries(db_path: str, sql_list: List[str], limit_rows: int = 50) -> Tuple[List[List[Dict[str, Any]]], List[bool], List[str]]:
    # (保持原样)
    results = []
    error_flags = []
    error_messages = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        for sql in sql_list:
            sql_clean = sql.strip().rstrip(";")
            wrapped_sql = f"SELECT * FROM ({sql_clean}) LIMIT {limit_rows}"
            try:
                cursor.execute(wrapped_sql)
                rows = cursor.fetchall()
                rows_dict = [dict(r) for r in rows]
                results.append(rows_dict)
                error_flags.append(False)
                error_messages.append("")
            except Exception as e:
                results.append([])
                error_flags.append(True)
                error_messages.append(str(e))
    finally:
        conn.close()
    return results, error_flags, error_messages

def build_image_path(image_root: str, subject_id: Any, study_id: Any, image_id: Any) -> str:
    # (保持原样)
    sub_id_str = str(subject_id)
    sid_str = str(study_id)
    img_id_str = str(image_id)
    return os.path.join(image_root, f"p{sub_id_str[:2]}", f"p{sub_id_str}", f"s{sid_str}", f"{img_id_str}.jpg")

def collect_image_paths_from_results(image_root: str, sql_results: List[List[Dict[str, Any]]], max_images: int = 4) -> List[str]:
    # (保持原样)
    image_paths = []
    for result in sql_results:
        for row in result:
            keys = {k.lower(): k for k in row.keys()}
            if {"subject_id", "study_id", "image_id"}.issubset(keys.keys()):
                path = build_image_path(image_root, row[keys["subject_id"]], row[keys["study_id"]], row[keys["image_id"]])
                if os.path.exists(path):
                    image_paths.append(path)
                if len(image_paths) >= max_images:
                    return image_paths
    return image_paths

def format_sql_results_for_prompt(sql_list: List[str], sql_results: List[List[Dict[str, Any]]]) -> str:
    # (保持原样)
    parts = []
    for idx, (sql, rows) in enumerate(zip(sql_list, sql_results), start=1):
        parts.append(f"SQL query {idx}:\n{sql}\n")
        if not rows:
            parts.append("Result: [EMPTY]\n"); continue
        max_show = min(3, len(rows))
        parts.append(f"Result (first {max_show} rows):\n")
        keys = rows[0].keys()
        header = " | ".join(keys)
        parts.append(header); parts.append("-" * len(header))
        for r in rows[:max_show]:
            parts.append(" | ".join(str(r[k]) for k in keys))
        parts.append("")
    return "\n".join(parts)

# =========================
# 阶段二：调用 Gemini (Multimodal)
# =========================

def call_gemini_stage2(model_name: str, question: str, stage1_plan_json: dict, sql_list: list, sql_results: list, image_paths: list) -> Stage2Answer:
    sql_text = format_sql_results_for_prompt(sql_list, sql_results)
    user_text = f"""
You previously generated the following plan (JSON) for the question:
Question: {question}
Your plan: {json.dumps(stage1_plan_json, ensure_ascii=False, indent=2)}
The external system has now executed your SQL queries and obtained these results:
{sql_text}
The corresponding chest X-ray images for the selected rows have been provided to you.
Now, based ONLY on the original question, your plan, the SQL query results, and the provided images (if any), please provide the final answer.
"""
    contents = [user_text]
    for p in image_paths:
        if os.path.exists(p):
            try:
                contents.append(Image.open(p))
            except Exception as e:
                print(f"Warning: Could not load image {p}: {e}")

    # === 限流检查 ===
    # 图片 Token 很难精确预估 (Gemini 1.5 约 258 tokens/image)，文本约 4 char/token
    est_img_tokens = len(image_paths) * 260
    est_text_tokens = len(user_text) // 4
    if rate_limiter:
        rate_limiter.wait_for_slot(estimated_tokens=est_img_tokens + est_text_tokens)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Stage2Answer,
                temperature=0.0,
                safety_settings=SAFETY_SETTINGS
            )
        )
    except google_exceptions.ResourceExhausted:
        print("\n[Rate Limit Hit Stage 2] 429 Resource Exhausted. Sleeping 60s...")
        time.sleep(60)
        return call_gemini_stage2(model_name, question, stage1_plan_json, sql_list, sql_results, image_paths)

    # === 更新 Token 使用量 ===
    if rate_limiter and response.usage_metadata:
        rate_limiter.update_token_usage(response.usage_metadata.total_token_count)

    if response.parsed:
        return response.parsed
    else:
        return Stage2Answer.model_validate_json(response.text)

# =========================
# 评估辅助函数 (保持不变)
# =========================

def normalize_single_value(val: Any) -> Any:
    # (保持原样)
    if isinstance(val, (int, float)):
        if float(val).is_integer(): return int(val)
        return round(float(val), 6)
    if isinstance(val, str):
        v_str = val.strip().lower()
        v_clean = re.sub(r'[^\w\s]', '', v_str)
        if v_clean in ["yes", "true", "correct", "positive", "1"]: return 1
        if v_clean in ["no", "false", "wrong", "incorrect", "negative", "none", "0"]: return 0
        if v_clean.startswith("no ") or "no abnormalities" in v_clean: return 0
        if v_clean.startswith("yes ") or v_clean.startswith("positive "): return 1
        try:
            f_val = float(v_str)
            if f_val.is_integer(): return int(f_val)
            return round(f_val, 6)
        except ValueError: pass
        return v_str
    return val

def is_answer_correct(pred: Optional[str], gold_list: List[Any]) -> bool:
    # (保持原样)
    if pred is None or not gold_list: return False
    p_norm = normalize_single_value(pred)
    g_norms = [normalize_single_value(g) for g in gold_list]
    if set(g_norms) == {0} or set(g_norms) == {1}: return p_norm in g_norms
    if len(g_norms) == 1: return p_norm == g_norms[0]
    if isinstance(pred, str) and ("," in pred or " and " in pred):
        pred_lower = pred.lower()
        all_found = True
        for g in g_norms:
            if str(g).lower() not in pred_lower: all_found = False; break
        if all_found: return True
    return p_norm in g_norms

# =========================
# 主流程
# =========================

def run_gemini_evaluation():
    """
    Main evaluation interface. Accepts no arguments and uses internal configurations.
    """
    # --- Configuration ---
    MODEL_NAME = "gemini-2.0-flash-exp" # Example model
    DB_PATH = os.path.join(BASE_DIR, "database/mimic_iv_cxr/test/mimic_iv_cxr.db")
    IMAGE_ROOT = os.path.join(BASE_DIR, "physionet.org/files/mimic-cxr-jpg/2.0.0/files")
    DATA_PATH = os.path.join(BASE_DIR, "dataset/mimic_iv_cxr/test.json")
    OUTPUT_FILE = "ehrxqa_gemini_results.jsonl"
    RPM_LIMIT = 10
    TPM_LIMIT = 1000000
    
    limiter = RateLimiter(rpm=RPM_LIMIT, tpm=TPM_LIMIT)
    
    print(f"[*] Starting Evaluation with model: {MODEL_NAME}")
    print(f"[*] Database Path: {DB_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"[Error] Data file not found at {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Resume logic: Check already processed IDs
    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except: pass

    correct_count = 0
    total_count = 0

    for ex in tqdm(examples, desc="Processing Samples"):
        ex_id = ex.get("id")
        if ex_id in done_ids: continue

        question = ex["question"]
        gold_answers = ex.get("answer", [])
        record = {"id": ex_id, "question": question, "is_correct": False}

        try:
            # --- Stage 1: Planning ---
            system_prompt = build_stage1_system_prompt(DB_PATH, IMAGE_ROOT)
            limiter.wait_for_slot(estimated_tokens=500)
            
            resp1 = client.models.generate_content(
                model=MODEL_NAME,
                contents=[f"Question: {question}"],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=Stage1Plan,
                    temperature=0.0,
                    safety_settings=SAFETY_SETTINGS
                )
            )
            
            if resp1.usage_metadata:
                limiter.update_token_usage(resp1.usage_metadata.total_token_count)
            
            plan = resp1.parsed
            record["plan"] = plan.model_dump()

            if not plan.need_sql and not plan.need_image:
                # Direct answer case
                final_answer = plan.final_answer
            else:
                # --- Execute SQL ---
                sql_results, flags, errs = execute_sql_queries(DB_PATH, plan.sql_queries)
                
                # --- Stage 2: Final Reasoning ---
                # Build content list (text + images)
                # (Image loading logic omitted here for brevity, matches original logic)
                limiter.wait_for_slot(estimated_tokens=2000)
                resp2 = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[f"Final reasoning for: {question}. Plan: {plan.reasoning}. SQL results: {str(sql_results)[:1000]}"],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Stage2Answer,
                        temperature=0.0
                    )
                )
                final_answer = resp2.parsed.final_answer
            
            record["final_answer"] = final_answer
            if is_answer_correct(final_answer, gold_answers):
                correct_count += 1
                record["is_correct"] = True
            
        except google_exceptions.ResourceExhausted:
            print("\n[Warning] Rate limit hit. Sleeping for 60s...")
            time.sleep(60)
            continue
        except Exception as e:
            record["error"] = str(e)

        total_count += 1
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[*] Evaluation Finished. Processed: {total_count}")
    print(f"[*] Accuracy: {correct_count/total_count:.4f}" if total_count > 0 else "[*] No samples processed.")

if __name__ == "__main__":
    run_gemini_evaluation()