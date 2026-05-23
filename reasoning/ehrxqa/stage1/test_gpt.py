#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EHRXQA + Multi-Modal Models (Qwen3-VL, Pixtral, InternVL, Llama3V, GLM-4V) 评测脚本

功能：
1. 在 prompt 中要求 final_answer 必须简洁，只包含答案本身。
2. 统计：
   - 总体准确率（基于 final_answer vs gold answer）
   - 阶段 1 错误率（阶段1输出格式错误 + SQL 语法错误 + 子表全为空）
   - 阶段 2 错误率（阶段2输出格式错误 + 回答错误）
   - SQL 错误率（只看 SQL 语法错误 + 子表全为空）
   - 总错误率（1 - Accuracy）
3. 将每个样本的全过程信息记录到 JSON 文件中。
4. [新增] 支持 GLM-4V 模型：增大 max_tokens，并在 <answer> 标签中抽取答案。
"""

import argparse
import json
import os
import sqlite3
import re
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, Glm4vForConditionalGeneration

from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
client = OpenAI()

class Stage1Plan(BaseModel):
    need_sql: bool
    sql_queries: List[str]
    need_image: bool
    reasoning: str
    final_answer: Optional[str]  # 允许 null

class Stage2Answer(BaseModel):
    final_answer: str
    explanation: str


def _ensure_pad_token(processor, model) -> int:
    """
    Ensure pad_token_id is set on tokenizer/processor/model to avoid generation warnings.
    """
    pad_id = getattr(processor, "pad_token_id", None)
    tok = getattr(processor, "tokenizer", None)

    if pad_id is None and tok is not None:
        pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None and tok is not None:
        pad_id = getattr(tok, "eos_token_id", None)
    if pad_id is None and hasattr(model, "config"):
        pad_id = getattr(model.config, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0

    if tok is not None:
        tok.pad_token_id = pad_id
    if hasattr(processor, "pad_token_id"):
        processor.pad_token_id = pad_id
    if hasattr(model, "config"):
        model.config.pad_token_id = pad_id
    return pad_id


def _move_to_device(batch, device):
    """
    Safely move inputs to target device.
    """
    if hasattr(batch, "to"):
        batch = batch.to(device)
    if isinstance(batch, torch.Tensor):
        return {"input_ids": batch}
    if hasattr(batch, "items"):
        return batch
    try:
        return dict(batch)
    except Exception:
        raise TypeError(f"Unsupported input type for model.generate: {type(batch)}")


def _ensure_attention_mask(batch: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Ensure attention_mask exists to avoid HF warnings.
    """
    if "attention_mask" in batch:
        return batch
    input_ids = batch.get("input_ids")
    if input_ids is None or not torch.is_tensor(input_ids):
        return batch
    pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_id is None:
        attn = torch.ones_like(input_ids)
    else:
        attn = (input_ids != pad_id).long()
    batch["attention_mask"] = attn.to(input_ids.device)
    return batch

print("Torch cuda available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# =========================
# 配置：DB Schema 描述
# =========================

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


# =========================
# System Prompt（阶段一）
# =========================

def build_stage1_system_prompt(db_path: str, image_root: str) -> str:
    prompt = f"""
You are a clinical question answering assistant for the EHRXQA dataset.

The SQLite database for EHRXQA is located at:
  {db_path}

{DB_SCHEMA_TEXT}

You CANNOT execute SQL yourself. You only WRITE SQL queries as plain text.
An external system (Python) will execute your SQL and return the results to you in a second turn.

You may also request chest X-ray images. Images are stored externally.
To locate an image, the external system needs: subject_id, study_id, image_id.

Images are stored under the root directory:
  {image_root}

The external system will construct image paths like:
  {image_root}/{{pXX}}/{{pXXXXXX}}/{{sYYYYYY}}/{{image_id}}.jpg
where:
  - pXX is 'p' + the first two characters of subject_id string,
  - pXXXXXX is 'p' + the full subject_id string,
  - sYYYYYY is 's' + the study_id string.

- If you need any image, your SQL MUST select the columns: subject_id, study_id, image_id.
- The external system will use these columns to load the images and provide them to you.

IMPORTANT TEMPORAL RULES (MANDATORY):
- The MIMIC-IV and MIMIC-CXR datasets use de-identified time shifting.
- You MUST assume the current_time is **"2105-12-31 23:59:00"**.
- When reasoning about time (e.g., checking if an event happened recently), ALWAYS treat this as the current time.
- NEVER use real-world current timestamps or SQL functions such as CURRENT_TIMESTAMP, NOW(), DATE('now'), or similar.
- Never reference system clock or actual present-day dates.
- ONLY use the literal string "2105-12-31 23:59:00" as the current time in time comparisons.

EXAMPLES (MANDATORY):
- If you need to check if charttime is before "now", use:
    charttime < "2105-12-31 23:59:00"
- If you need to check if something happened within 24 hours, compute:
    charttime >= DATETIME("2105-12-31 23:59:00", "-24 hours")


IMPORTANT:
- In the original EHRXQA paper, there is a pseudo function func_vqa(...).
- You MUST NOT use func_vqa in your SQL. Instead, treat it as a request to perform visual reasoning:
  you should request the corresponding chest X-ray images and answer by looking at the images.
- For example, questions involving "technical assessments", "findings", "still present" etc.
  that correspond to such visual QA should be handled by:
    1) querying TB_CXR (and optionally TB_CXR_PLUS only for locating the studies),
    2) requesting images via subject_id, study_id, image_id,
    3) and then using the images in the second stage to answer.

Decision rules:

1. Decide whether you need to query the database (need_sql):
   - If the question can be answered purely from common clinical knowledge or the text itself,
     set need_sql = false and directly answer.
   - If the question requires specific patient / study information (study_id, lab values,
     diagnoses, procedures, images, etc.), set need_sql = true and write SQL.

2. Decide whether you need images (need_image):
   - If you must visually inspect chest X-ray images (e.g., technical assessments, findings,
     comparison between two studies, etc.), set need_image = true.
   - In that case, at least one SQL query MUST have a SELECT clause including:
       subject_id, study_id, image_id
   - You should use TB_CXR (and possibly TB_CXR_PLUS) to locate the relevant image_ids,
     but you should NOT simply read the final visual answer directly from TB_CXR_PLUS
     if the question is intended as a visual QA (e.g., technical assessment comparison).
   - Instead, you should rely on the images provided in the second stage.

Output format (MANDATORY):

You MUST output a single JSON object with exactly the following keys:

{{
  "need_sql": true or false,
  "sql_queries": [
    "SQL query 1 here",
    "SQL query 2 here if needed"
  ],
  "need_image": true or false,
  "reasoning": "Brief explanation of why you decided to use or not use SQL and images.",
  "final_answer": "If need_sql=false and need_image=false, directly answer here with ONLY the answer content, no extra phrases. Otherwise set this to null."
}}

- If need_sql = false, set sql_queries to an empty list [].
- If need_image = true, at least one SQL query must select subject_id, study_id, image_id.
- Do NOT execute SQL.
- Do NOT use func_vqa or any other custom SQL functions.
- Do NOT add extra fields beyond the ones listed above.

VERY IMPORTANT ABOUT ANSWERS:
- Whenever you fill "final_answer", you MUST output ONLY the answer string itself,
  without any leading phrases such as:
    "The answer is", "The technical assessments still present are", etc.
- Example:
    Correct: "low lung volumes"
    Incorrect: "The technical assessments still present are: low lung volumes."
"""
    return prompt.strip()


# =========================
# Glm 专用：答案抽取辅助函数
# =========================
def extract_glm_content(text: str) -> str:
    """
    针对 glm 模型提取 <answer>...</answer> 中的内容。
    同时处理 <|begin_of_box|><|end_of_box|> 这种空占位符。
    """
    # 1. 尝试正则匹配 <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # 2. 去除特定的 box token
        content = content.replace("<|begin_of_box|><|end_of_box|>", "").strip()
        return content
    # 如果没匹配到标签，可能模型没有输出标签，直接返回原文本（交由后续 json 尝试解析）
    return text.strip()


# =========================
# 阶段一：调用模型，生成 plan(JSON)
# =========================

def call_gpt41_stage1(
    system_prompt: str,
    question: str,
    model: str = "gpt-4.1",
    max_output_tokens: int = 1024,
) -> Stage1Plan:
    resp = client.responses.parse(
        model=model,
        instructions=system_prompt,
        input=[{
            "role": "user",
            "content": f"Question: {question}"
        }],
        text_format=Stage1Plan,
        max_output_tokens=max_output_tokens,
    )
    return resp.output_parsed




def safe_json_extract(text: str) -> Dict[str, Any]:
    """
    通用 JSON 提取，寻找第一个 { 和最后一个 }
    """
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"Cannot find JSON object in model output:\n{text}")
    json_str = text[first:last+1]
    return json.loads(json_str)


def stage1_plan(
    db_path: str,
    image_root: str,
    question: str,
    model: str = "gpt-4.1",
    max_output_tokens: int = 1024,
):
    system_prompt = build_stage1_system_prompt(db_path, image_root)
    plan_obj = call_gpt41_stage1(
        system_prompt=system_prompt,
        question=question,
        model=model,
        max_output_tokens=max_output_tokens,
    )
    return plan_obj.model_dump(), plan_obj.model_dump_json()


import base64
import mimetypes

def image_path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# =========================
# 阶段二：执行 SQL + 加载图像
# =========================

def execute_sql_queries(
    db_path: str,
    sql_list: List[str],
    limit_rows: int = 50
) -> Tuple[List[List[Dict[str, Any]]], List[bool], List[str]]:
    results: List[List[Dict[str, Any]]] = []
    error_flags: List[bool] = []
    error_messages: List[str] = []

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
    sub_id_str = str(subject_id)
    sid_str = str(study_id)
    img_id_str = str(image_id)

    path = os.path.join(
        image_root,
        f"p{sub_id_str[:2]}",
        f"p{sub_id_str}",
        f"s{sid_str}",
        f"{img_id_str}.jpg",
    )
    return path


def collect_image_paths_from_results(image_root: str,
                                     sql_results: List[List[Dict[str, Any]]],
                                     max_images: int = 4) -> List[str]:
    image_paths = []
    for result in sql_results:
        for row in result:
            if {"subject_id", "study_id", "image_id"}.issubset(row.keys()):
                path = build_image_path(image_root, row["subject_id"], row["study_id"], row["image_id"])
                if os.path.exists(path):
                    image_paths.append(path)
                if len(image_paths) >= max_images:
                    return image_paths
    return image_paths


def format_sql_results_for_prompt(sql_list: List[str],
                                  sql_results: List[List[Dict[str, Any]]]) -> str:
    parts = []
    for idx, (sql, rows) in enumerate(zip(sql_list, sql_results), start=1):
        parts.append(f"SQL query {idx}:\n{sql}\n")
        if not rows:
            parts.append("Result: [EMPTY]\n")
            continue
        max_show = min(3, len(rows))
        parts.append(f"Result (first {max_show} rows):\n")
        keys = rows[0].keys()
        header = " | ".join(keys)
        parts.append(header)
        parts.append("-" * len(header))
        for r in rows[:max_show]:
            row_str = " | ".join(str(r[k]) for k in keys)
            parts.append(row_str)
        parts.append("")
    return "\n".join(parts)


# =========================
# 阶段二：带图像 + 子表调用模型
# =========================

def call_gpt41_stage2(
    question: str,
    stage1_plan_json: dict,
    sql_list: list,
    sql_results: list,
    image_paths: list,
    model: str = "gpt-4.1",
    max_output_tokens: int = 2048,
) -> Stage2Answer:
    sql_text = format_sql_results_for_prompt(sql_list, sql_results)

    user_text = f"""
You previously generated the following plan (JSON) for the question:

Question:
{question}

Your plan:
{json.dumps(stage1_plan_json, ensure_ascii=False, indent=2)}

The external system has now executed your SQL queries and obtained these results:

{sql_text}

Now, based ONLY on:
1) the original question,
2) your plan, and
3) the SQL query results and images,

please provide the final answer.
""".strip()

    content_parts = [{"type": "input_text", "text": user_text}]

    for p in image_paths:
        if os.path.exists(p):
            content_parts.append({
                "type": "input_image",
                "image_url": image_path_to_data_url(p)
            })

    resp = client.responses.parse(
        model=model,
        input=[{
            "role": "user",
            "content": content_parts
        }],
        text_format=Stage2Answer,
        max_output_tokens=max_output_tokens,
    )
    return resp.output_parsed


def call_model_with_images(model, processor,
                           model_type: str,
                           question: str,
                           stage1_plan_json: Dict[str, Any],
                           sql_list: List[str],
                           sql_results: List[List[Dict[str, Any]]],
                           image_paths: List[str],
                           max_new_tokens: int = 512) -> str:
    # 【修改】：针对 glm4v 模型，增大 max_tokens
    if model_type == 'glm4v':
        max_new_tokens = 4096

    # 1. 准备 SQL 文本结果
    sql_text = format_sql_results_for_prompt(sql_list, sql_results)

    # 2. 构建 Prompt 文本
    user_text = f"""
You previously generated the following plan (JSON) for the question:

Question:
{question}

Your plan:
{json.dumps(stage1_plan_json, ensure_ascii=False, indent=2)}

The external system has now executed your SQL queries and obtained these results:

{sql_text}

The corresponding chest X-ray images for the selected rows have been provided to you (attached as images in this message).

Now, based ONLY on:
1) the original question,
2) your plan, and
3) the SQL query results and images,

please provide the final answer.

Output format (MANDATORY):
{{
  "final_answer": "your short answer here (ONLY the answer, no extra phrases).",
  "explanation": "brief explanation of how you used the query results and images to answer (this can be longer)."
}}

VERY IMPORTANT:
- The value of "final_answer" MUST be only the answer itself, with no leading phrases.
  For example, answer like:
    "low lung volumes"
  NOT like:
    "The technical assessments still present are: low lung volumes."
"""

    # 3. 加载图片
    pil_images = []
    for p in image_paths:
        abs_path = os.path.abspath(p)
        try:
            img = Image.open(abs_path).convert("RGB")
            # 【修改】：glm 视情况也可能支持高分辨率，这里暂不 resize glm
            if model_type not in ["pixtral", "glm4v"]:
                max_size = 512
                w, h = img.size
                scale = min(max_size / max(w, h), 1.0)
                if scale < 1.0:
                    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            pil_images.append(img)
        except Exception:
            continue

    # 4. 构建消息列表
    content = []
    
    if len(pil_images) == 0 and model_type=='pixtral':
        messages = [{"role": "user", "content": user_text}]
    else:
        # 标准多模态格式 (GLM-4V 等走这里)
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

    # 5. 调用 apply_chat_template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    )
    
    inputs = _move_to_device(inputs, model.device)
    inputs = _ensure_attention_mask(inputs, model)

    # 6. 生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return outputs[0]


def parse_stage2_answer(text: str, model_type: str = "") -> Dict[str, Any]:
    """
    【修改】：针对 glm4v 模型，从 <answer>...</answer> 中解析，并构造成脚本需要的 Dict 格式
    """
    if model_type == 'glm4v':
        extracted_content = extract_glm_content(text)
        # 尝试看看内容是否是 JSON，如果是 JSON 则 parse
        # 如果不是 JSON（例如是纯文本 "Yes"），则手动构造成 {"final_answer": "Yes"}
        try:
            return safe_json_extract(extracted_content)
        except ValueError:
            # 解析 JSON 失败，说明可能是纯文本答案
            return {"final_answer": extracted_content, "explanation": "Parsed from <answer> tag."}

    return safe_json_extract(text)


# =========================
# 文本归一化 & 匹配函数 (增强版)
# =========================

def normalize_single_value(val: Any) -> Any:
    # 1. 数字转 int/float
    if isinstance(val, (int, float)):
        if float(val).is_integer():
            return int(val)
        return round(float(val), 6)

    # 2. 字符串处理
    if isinstance(val, str):
        v_str = val.strip().lower()
        v_clean = re.sub(r'[^\w\s]', '', v_str) # 去标点
        
        # Boolean 映射
        if v_clean in ["yes", "true", "correct", "positive", "1"]:
            return 1
        if v_clean in ["no", "false", "wrong", "incorrect", "negative", "none", "0"]:
            return 0
            
        # 长句 Boolean 判别
        if v_clean.startswith("no ") or v_clean.startswith("not ") or "no abnormalities" in v_clean:
            return 0
        if v_clean.startswith("yes ") or v_clean.startswith("positive "):
            return 1
            
        # 数值处理
        try:
            f_val = float(v_str)
            if f_val.is_integer():
                return int(f_val)
            return round(f_val, 6)
        except ValueError:
            pass
            
        # 普通字符串：保留小写
        return v_str
        
    return val

def is_answer_correct(pred: Optional[str], gold_list: List[Any]) -> bool:
    if pred is None:
        return False
    if not gold_list:
        return False

    p_norm = normalize_single_value(pred)
    g_norms = [normalize_single_value(g) for g in gold_list]
    
    if set(g_norms) == {0} or set(g_norms) == {1}:
        return p_norm in g_norms
        
    if len(g_norms) == 1:
        return p_norm == g_norms[0]
        
    if isinstance(pred, str) and ("," in pred or " and " in pred):
        pred_lower = pred.lower()
        all_found = True
        for g in g_norms:
            if str(g).lower() not in pred_lower:
                all_found = False
                break
        if all_found:
            return True
            
    return p_norm in g_norms



def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
# =========================
# 主流程
# =========================

def run_openai_evaluation():
    """
    Main evaluation interface. 
    Uses default internal paths based on BASE_DIR and requires no parameters.
    """
    # --- Internal Configurations ---
    DB_PATH = os.path.join(BASE_DIR, "database/mimic_iv_cxr/test/mimic_iv_cxr.db")
    IMAGE_ROOT = os.path.join(BASE_DIR, "physionet.org/files/mimic-cxr-jpg/2.0.0/files")
    DATA_PATH = os.path.join(BASE_DIR, "dataset/mimic_iv_cxr/test.json")
    OUTPUT_JSONL = "ehrxqa_eval_results.jsonl"
    OPENAI_MODEL = "gpt-4o" # Example model
    
    # Initialize OpenAI Client
    client = OpenAI()

    print(f"[*] Starting Evaluation Process")
    print(f"[*] Base Directory: {BASE_DIR}")
    print(f"[*] Model: {OPENAI_MODEL}")

    # Load Dataset
    if not os.path.exists(DATA_PATH):
        print(f"[Error] Dataset not found at: {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)
    
    # Setup Output and Resume Logic
    done_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("id") is not None:
                        done_ids.add(obj["id"])
                except: continue
        print(f"[*] Resuming: {len(done_ids)} samples already processed.")

    # Metrics
    total_processed = 0
    correct_count = 0
    
    progress_bar = tqdm(examples, desc="Evaluating Samples")

    for ex in progress_bar:
        ex_id = ex.get("id")
        if ex_id in done_ids:
            continue

        question = ex["question"]
        gold_answers = ex.get("answer", [])
        record = {"id": ex_id, "question": question, "gold_answers": gold_answers, "is_correct": False}

        try:
            # --- STAGE 1: Planning ---
            # Instructions are essentially the build_stage1_system_prompt logic
            system_instr = f"You are a clinical assistant. DB is at {DB_PATH}. Follow the schema and return JSON."
            
            # Using structured output (parse)
            completion1 = client.beta.chat.completions.parse(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": f"Question: {question}"}
                ],
                response_format=Stage1Plan,
            )
            plan = completion1.choices[0].message.parsed
            record["stage1_plan"] = plan.model_dump()

            if not plan.need_sql and not plan.need_image:
                # Direct Answer Case
                final_answer = plan.final_answer
            else:
                # --- SQL Execution ---
                sql_results, flags, errs = execute_sql_queries(DB_PATH, plan.sql_queries)
                record["sql_results"] = sql_results
                
                # --- STAGE 2: Multimodal Reasoning ---
                # Formatting results for prompt
                sql_context = str(sql_results)[:2000] # Truncate for token safety
                
                completion2 = client.beta.chat.completions.parse(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "user", "content": f"Based on SQL results: {sql_context}, answer the question: {question}"}
                    ],
                    response_format=Stage2Answer,
                )
                final_answer = completion2.choices[0].message.parsed.final_answer
            
            record["final_answer"] = final_answer
            if is_answer_correct(final_answer, gold_answers):
                correct_count += 1
                record["is_correct"] = True

        except Exception as e:
            record["error"] = str(e)
            print(f"[Warning] Sample {ex_id} failed: {e}")

        # Persistent Logging
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        total_processed += 1
        progress_bar.set_postfix({"Accuracy": f"{correct_count / max(1, total_processed):.3f}"})

    print(f"\n[*] Finished. Accuracy: {correct_count / max(1, total_processed):.4f}")

if __name__ == "__main__":
    # Ensure you have your OPENAI_API_KEY environment variable set
    run_openai_evaluation()