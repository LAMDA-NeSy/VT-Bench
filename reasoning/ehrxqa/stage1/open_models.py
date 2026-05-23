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
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        content = content.replace("<|begin_of_box|><|end_of_box|>", "").strip()
        return content
    return text.strip()


# =========================
# 阶段一：调用模型，生成 plan(JSON)
# =========================

def call_model_text_only(model, processor, model_type: str, system_prompt: str, question: str,
                         max_new_tokens: int = 512) -> str:

    # glm 允许更大，但会被 remaining window 强制截断
    if model_type == 'glm4v':
        max_new_tokens = 4096

    if model_type == "pixtral":
        combined_text = f"{system_prompt}\n\nQuestion: {question}"
        messages = [{"role": "user", "content": combined_text}]
    else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Question: {question}"}],
            },
        ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CONTEXT_LEN,
    )

    inputs = _move_to_device(inputs, model.device)
    inputs = _ensure_attention_mask(inputs, model)

    # 保证 input + new_tokens <= CONTEXT_LEN
    in_len = int(inputs["input_ids"].shape[1])
    remaining = max(CONTEXT_LEN - in_len, 1)
    max_new_tokens = min(max_new_tokens, remaining)

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


def stage1_plan(model, processor, model_type: str, db_path: str, image_root: str, question: str) -> Tuple[Dict[str, Any], str]:
    """
    返回：(解析后的 plan, 原始输出文本)
    """
    system_prompt = build_stage1_system_prompt(db_path, image_root)
    raw_output = call_model_text_only(model, processor, model_type, system_prompt, question)

    text_to_parse = raw_output
    if model_type == 'glm4v':
        text_to_parse = extract_glm_content(raw_output)

    plan = safe_json_extract(text_to_parse)
    return plan, raw_output


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

def call_model_with_images(model, processor,
                           model_type: str,
                           question: str,
                           stage1_plan_json: Dict[str, Any],
                           sql_list: List[str],
                           sql_results: List[List[Dict[str, Any]]],
                           image_paths: List[str],
                           max_new_tokens: int = 512) -> str:

    if model_type == 'glm4v':
        max_new_tokens = 4096

    sql_text = format_sql_results_for_prompt(sql_list, sql_results)

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

    pil_images = []
    for p in image_paths:
        abs_path = os.path.abspath(p)
        try:
            img = Image.open(abs_path).convert("RGB")
            # 这里保留你原来的策略：pixtral/glm4v 不 resize，其它 resize 到 max_size=512
            if model_type not in ["pixtral", "glm4v"]:
                max_size = 512
                w, h = img.size
                scale = min(max_size / max(w, h), 1.0)
                if scale < 1.0:
                    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            pil_images.append(img)
        except Exception:
            continue

    content = []

    if len(pil_images) == 0 and model_type == 'pixtral':
        messages = [{"role": "user", "content": user_text}]
    else:
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CONTEXT_LEN,
    )

    inputs = _move_to_device(inputs, model.device)
    inputs = _ensure_attention_mask(inputs, model)

    # 保证 input + new_tokens <= CONTEXT_LEN
    in_len = int(inputs["input_ids"].shape[1])
    remaining = max(CONTEXT_LEN - in_len, 1)
    max_new_tokens = min(max_new_tokens, remaining)

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
    针对 glm4v：从 <answer>...</answer> 中解析，并构造成 Dict
    """
    if model_type == 'glm4v':
        extracted_content = extract_glm_content(text)
        try:
            return safe_json_extract(extracted_content)
        except ValueError:
            return {"final_answer": extracted_content, "explanation": "Parsed from <answer> tag."}

    return safe_json_extract(text)


# =========================
# 文本归一化 & 匹配函数 (增强版)
# =========================

def normalize_single_value(val: Any) -> Any:
    if isinstance(val, (int, float)):
        if float(val).is_integer():
            return int(val)
        return round(float(val), 6)

    if isinstance(val, str):
        v_str = val.strip().lower()
        v_clean = re.sub(r'[^\w\s]', '', v_str)

        if v_clean in ["yes", "true", "correct", "positive", "1"]:
            return 1
        if v_clean in ["no", "false", "wrong", "incorrect", "negative", "none", "0"]:
            return 0

        if v_clean.startswith("no ") or v_clean.startswith("not ") or "no abnormalities" in v_clean:
            return 0
        if v_clean.startswith("yes ") or v_clean.startswith("positive "):
            return 1

        try:
            f_val = float(v_str)
            if f_val.is_integer():
                return int(f_val)
            return round(f_val, 6)
        except ValueError:
            pass

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


# =========================
# 主流程
# =========================

def evaluate_ehrxqa(
    model_id: str,
    model_type: str = "pixtral",
    db_path: str = os.path.join(BASE_DIR, "database/mimic_iv_cxr/test/mimic_iv_cxr.db"),
    image_root: str = os.path.join(BASE_DIR, "physionet.org/files/mimic-cxr-jpg/2.0.0/files"),
    data_path: str = os.path.join(BASE_DIR, "dataset/mimic_iv_cxr/test.json"),
    output_json: str = "ehrxqa_eval_results.json",
    max_samples: Optional[int] = None,
    single_example: bool = False
):
    """
    Complete function interface for EHRXQA evaluation.
    Handles multi-modal model inference, SQL execution, and performance metrics.
    """
    model_type = model_type.lower()
    print(f"[*] Base directory: {BASE_DIR}")
    print(f"[*] Loading model: {model_id} (Type: {model_type})")

    # 1. Model and Processor Loading
    load_params = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True
    }

    if model_type == "glm4v":
        model = Glm4vForConditionalGeneration.from_pretrained(model_id, **load_params)
    elif model_type in ["qwen3", "qwen3_thinking", "internvl3", "pixtral"]:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_params)
    elif model_type == "llama3v":
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _ensure_pad_token(processor, model)
    _set_context_len(processor, model, CONTEXT_LEN)

    # 2. Dataset Preparation
    if single_example:
        examples = [{
            "db_id": "mimic_iv_cxr", "split": "test", "id": 0,
            "question": "in the 55608075 study, list all technical assessments still present compared to the 59403367 study.",
            "answer": ["low lung volumes"],
            "value": {"study_id1": 55608075, "study_id2": 59403367}
        }]
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = data[:max_samples] if max_samples is not None else data

    # 3. Metrics and Log Initialization
    metrics = {
        "total": 0, "correct": 0, "stage1_errors": 0, "stage2_errors": 0,
        "sql_failures": 0, "need_sql_count": 0, "need_image_count": 0
    }
    log_records = []

    # 4. Evaluation Loop
    progress_bar = tqdm(examples, desc=f"Evaluating {model_id}")
    for ex in progress_bar:
        question = ex["question"]
        gold_answers = ex.get("answer", [])
        metrics["total"] += 1

        record = {
            "id": ex.get("id"), "question": question, "gold_answers": gold_answers,
            "stage1_error": False, "stage2_error": False, "is_correct": False
        }

        try:
            # --------- STAGE 1: Reasoning Plan ---------
            plan, stage1_raw = stage1_plan(model, processor, model_type, db_path, image_root, question)
            record.update({"stage1_raw": stage1_raw, "stage1_plan": plan})

            need_sql = bool(plan.get("need_sql", False))
            need_image = bool(plan.get("need_image", False))
            sql_queries = plan.get("sql_queries", [])
            
            if need_sql: metrics["need_sql_count"] += 1
            if need_image: metrics["need_image_count"] += 1

            # Scenario A: Direct answer without tools
            if not need_sql and not need_image:
                final_ans = plan.get("final_answer")
                record["final_answer"] = final_ans
                if is_answer_correct(final_ans, gold_answers):
                    metrics["correct"] += 1
                    record["is_correct"] = True
                log_records.append(record)
                continue

            # --------- STAGE 2: Tool Execution (SQL & Images) ---------
            sql_results, sql_err_flags, sql_err_msgs = execute_sql_queries(db_path, sql_queries)
            record.update({"sql_results": sql_results, "sql_errors": sql_err_msgs})

            # Check if SQL execution failed or returned empty results when required
            all_empty = all(len(r) == 0 for r in sql_results) if sql_results else True
            if any(sql_err_flags) or (need_sql and all_empty):
                metrics["sql_failures"] += 1
                metrics["stage1_errors"] += 1
                record["stage1_error"] = True
                log_records.append(record)
                continue

            # Collect Image Paths
            image_paths = []
            if need_image:
                image_paths = collect_image_paths_from_results(image_root, sql_results, max_images=1)
                record["image_paths"] = image_paths

            # --------- STAGE 3: Final Multimodal Reasoning ---------
            stage2_raw = call_model_with_images(
                model, processor, model_type, question, plan, sql_queries, sql_results, image_paths
            )
            record["stage2_raw"] = stage2_raw

            try:
                ans_json = parse_stage2_answer(stage2_raw, model_type=model_type)
                final_ans = ans_json.get("final_answer")
                record["stage2_parsed"] = ans_json
                record["final_answer"] = final_ans

                if is_answer_correct(final_ans, gold_answers):
                    metrics["correct"] += 1
                    record["is_correct"] = True
                else:
                    metrics["stage2_errors"] += 1
                    record["stage2_error"] = True

            except Exception as e:
                metrics["stage2_errors"] += 1
                record["stage2_error"] = True
                record["stage2_exception"] = str(e)

        except Exception as e:
            metrics["stage1_errors"] += 1
            record["stage1_error"] = True
            record["main_exception"] = str(e)

        log_records.append(record)
        progress_bar.set_postfix({"Acc": f"{metrics['correct']/metrics['total']:.3f}"})

    # 5. Result Logging and Summary
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    print("\n" + "#" * 30)
    print(f"Evaluation Finished: {model_id}")
    print(f"Total Samples: {metrics['total']}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Stage 1 Errors: {metrics['stage1_errors']}")
    print(f"Stage 2 Errors: {metrics['stage2_errors']}")
    print(f"SQL/Empty Result Failures: {metrics['sql_failures']}")
    print("#" * 30)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="pixtral")
    args = parser.parse_args()

    evaluate_ehrxqa(model_id=args.model_id, model_type=args.model_type)
