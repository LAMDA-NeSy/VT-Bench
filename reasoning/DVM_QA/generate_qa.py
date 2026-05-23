import os
import re
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# =================================================================
# 1. Configuration & Constants
# =================================================================

CSV_PATH = "/home/jiazy/DVM_QA/Ad_table (extra).csv"
IMAGE_TABLE_CSV_PATH = "/home/jiazy/DVM_QA/Image_table.csv"
ROOT_IMG_DIR = "/home/jiazy/DVM_QA/resized_DVM"
OUTPUT_DIR = "/home/jiazy/DVM_QA/stage1"

# Options for the number of rows in the generated sub-tables
ROW_SIZE_OPTIONS = [10, 20, 50]
# Number of samples per decision mode and per row-size configuration
SAMPLES_PER_FILE = 150 
N_TABLE_COLS = 15
RANDOM_SEED = 2025

# Column Name Mapping
COLOR_COL = "Color"
BRAND_COL = "Maker"
MODEL_COL = "Genmodel"
YEAR_COL = "Reg_year"
ADV_ID_COL = "Adv_ID"

# Candidate columns for the 'attribute' tasks
ATTR_COLS = [
    "Runned_Miles",
    "Engin_size",
    "Price",
    "Wheelbase",
    "Height",
    "Width",
    "Average_mpg",
    "Top_speed",
]

# Blacklist for columns that should not appear as distractors or random content
# Includes identifiers, ground truth flags, and metadata
BASE_BLACKLIST = {
    ADV_ID_COL,
    "__image_abs_path__", "__image_rel_path__", "__is_answer__",
    "id", "ID", "adv_id", "image_url", "image_path", "Genmodel_ID",
    "Image_name", "uni_id", "v_id",
    "Predicted_viewpoint", "Image_ID",
    "Maker_from_image", "Color_from_image", "__color_norm__",
}

# Ambiguous or non-distinctive colors to be excluded in color-sensitive decision modes
NOISY_COLORS = {"silver", "grey", "gray", "unlisted"}


# =================================================================
# 2. Utility Functions
# =================================================================

def norm_piece(x: Any) -> str:
    """Standardizes string pieces by removing extra whitespace and replacing slashes."""
    s = str(x) if pd.notna(x) else ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", "_")
    return s.strip()


def build_image_abs_path(row: pd.Series) -> Optional[str]:
    """
    Parses the directory structure based on 'Image_name'.
    Expected hierarchy: resized_DVM/Maker/Genmodel/Reg_year/Color/Image_name
    Data is extracted from the '$$' delimited filename.
    """
    try:
        image_name = row.get("Image_name")
        if pd.isna(image_name) or not str(image_name).strip():
            return None
        image_name = str(image_name).strip()

        parts = image_name.split("$$")
        # Expected format: Maker$$Genmodel$$Reg_year$$Color$$Adv_ID$$image_*.ext
        if len(parts) < 6:
            return None

        maker = norm_piece(parts[0])
        genmodel = norm_piece(parts[1])
        year = norm_piece(parts[2])
        color_dir = norm_piece(parts[3])

        if not all([maker, genmodel, year, color_dir]):
            return None

        dir_path = Path(ROOT_IMG_DIR, maker, genmodel, year, color_dir)
        img_path = dir_path / image_name

        if img_path.exists():
            return str(img_path)
        return None
    except Exception:
        return None


def build_loc_prompt(md_table: str) -> str:
    """Constructs the prompt for Row Localization tasks."""
    prompt = (
        f"{md_table}\n\n"
        "You are given a photo of a car and the table above listing several candidate cars.\n"
        "Exactly one row in the table corresponds to the car in the image.\n"
        "Carefully compare the visual features of the car in the image with the attributes in each row to identify the correct match.\n"
        "Output the 'Row ID' of the matching row.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object:\n"
        "{\n"
        '  "answer": [ROW_ID],\n'
        '  "explanation": "Brief step-by-step reasoning..."\n'
        "}\n"
    )
    return prompt


def build_attr_prompt(md_table: str, attr_name: str) -> str:
    """Constructs the prompt for Attribute Value retrieval tasks."""
    prompt = (
        f"{md_table}\n\n"
        "You are given a photo of a car and the table above listing several candidate cars.\n"
        "Exactly one row in the table corresponds to the car in the image.\n"
        f"Based on the matching row, answer the following question about the car in the image:\n"
        f"Question: What is the value of '{attr_name}' for the car in the image?\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object:\n"
        "{\n"
        '  "answer": [VALUE],\n'
        '  "explanation": "Brief step-by-step reasoning..."\n'
        "}\n"
    )
    return prompt


def build_count_prompt(md_table: str, attr_name: str, group_type: str) -> str:
    """
    Constructs the prompt for Counting tasks with logical constraints.
    group_type: "maker" -> same brand; "color" -> same color.
    """
    if group_type == "maker":
        cond = "that have the same Maker as the car in the image"
    else:
        cond = "that have the same Color as the car in the image"

    prompt = (
        f"{md_table}\n\n"
        "You are given a photo of a car and the table above listing several candidate cars.\n"
        "Exactly one row in the table corresponds to the car in the image.\n"
        f"Among the cars in the table {cond}, how many have a larger '{attr_name}' value than the car in the image?\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object:\n"
        "{\n"
        '  "answer": [COUNT],\n'
        '  "explanation": "Brief step-by-step reasoning..."\n'
        "}\n"
    )
    return prompt


def build_mean_prompt(md_table: str, attr_name: str, group_type: str) -> str:
    """
    Constructs the prompt for Average (Mean) tasks with logical constraints.
    """
    if group_type == "maker":
        cond = "that have the same Maker as the car in the image"
    else:
        cond = "that have the same Color as the car in the image"

    prompt = (
        f"{md_table}\n\n"
        "You are given a photo of a car and the table above listing several candidate cars.\n"
        "Exactly one row in the table corresponds to the car in the image.\n"
        f"Consider the cars in the table {cond}.\n"
        f"Question: What is the average value of '{attr_name}' for these cars, based only on the values in the table?\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object:\n"
        "{\n"
        '  "answer": [VALUE],\n'
        '  "explanation": "Brief step-by-step reasoning..."\n'
        "}\n"
    )
    return prompt


def split_into_thirds(total: int):
    """Splits an integer into three parts as evenly as possible."""
    base = total // 3
    rem = total % 3
    parts = [base, base, base]
    for i in range(rem):
        parts[i] += 1
    return parts[0], parts[1], parts[2]


# =================================================================
# 3. Core Data Generation Logic
# =================================================================

def generate_all_modes():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Loading Ad CSV: {CSV_PATH}")
    df_ad = pd.read_csv(CSV_PATH, skipinitialspace=True)

    print(f"Loading Image CSV: {IMAGE_TABLE_CSV_PATH}")
    df_img = pd.read_csv(IMAGE_TABLE_CSV_PATH, skipinitialspace=True)

    # ---- Adv_ID Alignment ----
    df_ad[ADV_ID_COL] = df_ad[ADV_ID_COL].astype(str).str.strip()

    if ADV_ID_COL in df_img.columns:
        df_img[ADV_ID_COL] = df_img[ADV_ID_COL].astype(str).str.strip()
    else:
        if "Image_ID" not in df_img.columns:
            raise ValueError("Image_table missing required linkage columns: Image_ID or Adv_ID")
        # Extract Adv_ID from Image_ID if Adv_ID column is missing
        print("Extracting Adv_ID from Image_ID with rsplit('$$', 1)...")
        df_img[ADV_ID_COL] = (
            df_img["Image_ID"]
            .astype(str)
            .str.rsplit("$$", n=1)
            .str[0]
            .str.strip()
        )

    df_img[ADV_ID_COL] = df_img[ADV_ID_COL].astype(str).str.strip()

    # ---- Viewpoint Filtering: Retain only front-view images (viewpoint 0) ----
    if "Predicted_viewpoint" not in df_img.columns:
        raise ValueError("Image_table missing Predicted_viewpoint column")

    df_img["Predicted_viewpoint"] = pd.to_numeric(
        df_img["Predicted_viewpoint"], errors="coerce"
    )
    df_img_front = df_img[df_img["Predicted_viewpoint"] == 0].copy()
    print(f"Front-view images found: {len(df_img_front)}")

    keep_cols = [ADV_ID_COL, "Image_name", "Predicted_viewpoint"]
    df_img_front = df_img_front.dropna(subset=["Image_name", ADV_ID_COL])

    # ---- Merge Advertisement data with Front-view images ----
    df = pd.merge(df_ad, df_img_front[keep_cols], on=ADV_ID_COL, how="inner")
    print(f"Merged rows (Ad + Image): {len(df)}")

    # ---- Basic Data Cleaning ----
    required_cols = [COLOR_COL, BRAND_COL, MODEL_COL, ADV_ID_COL, "Image_name"]
    df = df.dropna(subset=required_cols)
    df[COLOR_COL] = df[COLOR_COL].astype(str).str.strip()
    df[BRAND_COL] = df[BRAND_COL].astype(str).str.strip()
    df = df[(df[COLOR_COL] != "") & (df[BRAND_COL] != "")]

    # ---- Cross-verify Maker/Color from Image_name to ensure data consistency ----
    def parse_from_image_name_color_maker(name: Any):
        parts = str(name).split("$$")
        if len(parts) < 4:
            return None, None
        maker_img = parts[0].strip()
        color_img = parts[3].strip()
        return maker_img, color_img

    maker_list = []
    color_list = []
    for v in df["Image_name"]:
        m, c = parse_from_image_name_color_maker(v)
        maker_list.append(m)
        color_list.append(c)
    df["Maker_from_image"] = maker_list
    df["Color_from_image"] = color_list

    def norm_for_compare(x):
        return norm_piece(x).lower()

    # Filter out rows where the tabular Maker/Color doesn't match the image metadata
    mask_consistent = (
        (df[COLOR_COL].apply(norm_for_compare) == df["Color_from_image"].apply(norm_for_compare)) &
        (df[BRAND_COL].apply(norm_for_compare) == df["Maker_from_image"].apply(norm_for_compare))
    )

    num_inconsistent = (~mask_consistent).sum()
    print(f"Rows with inconsistent Maker/Color (will be dropped): {num_inconsistent}")

    df = df[mask_consistent].copy()
    print(f"Rows kept after consistency filter: {len(df)}")

    # ---- Generate Absolute Image Paths ----
    print("Building paths...")
    tqdm.pandas(desc="Path checking")
    df["__image_abs_path__"] = df.progress_apply(build_image_abs_path, axis=1)

    df_valid = df[df["__image_abs_path__"].notna()].copy()

    # Ensure uniqueness of Adv_ID to prevent same car appearing across target/distractor sets
    df_valid = df_valid.drop_duplicates(subset=[ADV_ID_COL])
    print(f"Valid unique rows with existing images: {len(df_valid)}")
    if len(df_valid) == 0:
        raise ValueError("No valid images found after filtering. Check paths and viewpoint settings.")

    # Normalize color for noise filtering
    df_valid["__color_norm__"] = (
        df_valid[COLOR_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Relative path calculation (relative to DVM parent directory)
    root_path_obj = Path(ROOT_IMG_DIR)
    base_path_for_rel = root_path_obj.parent

    def get_rel_path(abs_p: str) -> str:
        try:
            return str(Path(abs_p).relative_to(base_path_for_rel))
        except ValueError:
            return abs_p

    df_valid["__image_rel_path__"] = df_valid["__image_abs_path__"].apply(get_rel_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Evaluation modes defined by the criteria for distinguishing the car
    MODES = ["color", "maker", "maker_color"]

    # ========== Main Loop: Generate nested sub-tables (10/20/50 rows) & 4 Task Categories per mode ==========
    for mode in MODES:
        print(f"\n### Generating mode: {mode}")

        # Filter out noisy colors for color-based decision modes
        if mode in ("color", "maker_color"):
            mode_df = df_valid[~df_valid["__color_norm__"].isin(NOISY_COLORS)].copy()
        else:
            mode_df = df_valid

        print(f"Rows available for mode '{mode}': {len(mode_df)}")

        # Adjust the random column pool based on the current mode to avoid leakage
        current_blacklist = BASE_BLACKLIST.copy()
        if mode == "color":
            current_blacklist.add(COLOR_COL)
            current_blacklist.add(BRAND_COL)
        elif mode == "maker":
            current_blacklist.add(BRAND_COL)
            current_blacklist.add(COLOR_COL)
        else:  # maker_color
            current_blacklist.add(BRAND_COL)
            current_blacklist.add(COLOR_COL)

        valid_cols_pool = [c for c in mode_df.columns if c not in current_blacklist]

        # Containers for the four QA task types across different row sizes
        loc_per_rowsize = {rs: [] for rs in ROW_SIZE_OPTIONS}
        attr_per_rowsize = {rs: [] for rs in ROW_SIZE_OPTIONS}
        count_per_rowsize = {rs: [] for rs in ROW_SIZE_OPTIONS}
        mean_per_rowsize = {rs: [] for rs in ROW_SIZE_OPTIONS}

        pbar = tqdm(total=SAMPLES_PER_FILE, desc=f"{mode}|nested_loc+attr+count+mean")
        attempts = 0
        max_attempts = SAMPLES_PER_FILE * 500

        while len(loc_per_rowsize[10]) < SAMPLES_PER_FILE and attempts < max_attempts:
            attempts += 1

            # ------- 1) Sample Target Car -------
            target_row = mode_df.sample(1).iloc[0]
            target_color = target_row[COLOR_COL]
            target_maker = target_row[BRAND_COL]

            # ------- 2) Sample Distractors (up to 49 for the 50-row table) -------
            n_distractors_full = 50 - 1 

            if mode == "color":
                distractor_pool = mode_df[mode_df[COLOR_COL] != target_color]
                if len(distractor_pool) < n_distractors_full:
                    continue
                distractors_full = distractor_pool.sample(
                    n_distractors_full, replace=False
                ).copy()

            elif mode == "maker":
                distractor_pool = mode_df[mode_df[BRAND_COL] != target_maker]
                if len(distractor_pool) < n_distractors_full:
                    continue
                distractors_full = distractor_pool.sample(
                    n_distractors_full, replace=False
                ).copy()

            else:  # maker_color: stratified sampling (1/3 Same Maker, 1/3 Same Color, 1/3 Both Diff)
                pool1 = mode_df[
                    (mode_df[BRAND_COL] == target_maker) &
                    (mode_df[COLOR_COL] != target_color)
                ]
                pool2 = mode_df[
                    (mode_df[COLOR_COL] == target_color) &
                    (mode_df[BRAND_COL] != target_maker)
                ]
                pool3 = mode_df[
                    (mode_df[BRAND_COL] != target_maker) &
                    (mode_df[COLOR_COL] != target_color)
                ]

                n1_target, n2_target, n3_target = split_into_thirds(n_distractors_full)
                k1 = min(len(pool1), n1_target)
                k2 = min(len(pool2), n2_target)

                part1 = pool1.sample(k1, replace=False) if k1 > 0 else pd.DataFrame()
                part2 = pool2.sample(k2, replace=False) if k2 > 0 else pd.DataFrame()

                current_count = len(part1) + len(part2)
                needed = n_distractors_full - current_count

                if len(pool3) < needed:
                    continue

                part3 = pool3.sample(needed, replace=False)
                distractors_full = pd.concat(
                    [part1, part2, part3], ignore_index=True
                ).copy()

                if len(distractors_full) != n_distractors_full:
                    continue

            # Shuffle once to ensure 10/20/50 nesting remains random yet consistent
            distractors_full = distractors_full.sample(frac=1).reset_index(drop=True)

            # ------- 3) Construct full DataFrames for each row-size -------
            tables_full = {}
            for rs in ROW_SIZE_OPTIONS:
                n_distractors = rs - 1
                chosen_distractors = distractors_full.iloc[:n_distractors].copy()

                target_df = target_row.to_frame().T.copy()
                target_df["__is_answer__"] = True
                chosen_distractors["__is_answer__"] = False

                combined_df = pd.concat(
                    [target_df, chosen_distractors],
                    ignore_index=True
                )
                combined_df = combined_df.sample(frac=1).reset_index(drop=True)
                combined_df["Row ID"] = range(1, len(combined_df) + 1)

                tables_full[rs] = combined_df

            # ------- 4) Select a suitable 'attribute' column for reasoning tasks -------
            # Must satisfy diversity constraints across all three table sizes
            cand_attrs = []
            for attr in ATTR_COLS:
                if attr in mode_df.columns and pd.notna(target_row.get(attr)):
                    cand_attrs.append(attr)

            if not cand_attrs:
                continue

            good_attrs = []
            for attr in cand_attrs:
                ok = True
                target_val = target_row[attr]
                for rs, df_rs in tables_full.items():
                    if attr not in df_rs.columns:
                        ok = False
                        break
                    col = df_rs[attr]
                    non_null = col.dropna()
                    # Ensure the attribute has enough unique values to be meaningful
                    if non_null.nunique() < 3:
                        ok = False
                        break
                    # Avoid columns where the target value is overly dominant
                    same_cnt = (col == target_val).sum()
                    if same_cnt > len(df_rs) // 2:
                        ok = False
                        break
                if ok:
                    good_attrs.append(attr)

            if not good_attrs:
                continue

            chosen_attr = random.choice(good_attrs)

            # Define decision-making columns based on mode
            if mode == "color":
                decision_cols = [COLOR_COL]
            elif mode == "maker":
                decision_cols = [BRAND_COL]
            else:
                decision_cols = [COLOR_COL, BRAND_COL]

            # Dimension for logical grouping in reasoning
            if mode == "color":
                group_type = "maker"
            elif mode == "maker":
                group_type = "color"
            else:
                group_type = random.choice(["maker", "color"])

            # Calculate number of random filler columns
            n_random_cols = N_TABLE_COLS - 1 - len(decision_cols) - 1
            if n_random_cols < 0:
                raise ValueError("N_TABLE_COLS too small for mandatory columns.")

            rand_pool = [
                c for c in valid_cols_pool
                if c not in decision_cols and c != chosen_attr
            ]

            base_idx = len(loc_per_rowsize[10])

            # Temporary buffers to ensure atomic commit of all task types for a sample
            tmp_loc_items = {}
            tmp_attr_items = {}
            tmp_count_items = {}
            tmp_mean_items = {}
            valid_for_all = True

            # ------- 5) Final Table & Prompt Construction -------
            for rs in ROW_SIZE_OPTIONS:
                combined_df = tables_full[rs].copy()

                if len(rand_pool) <= n_random_cols:
                    selected_random_cols = rand_pool
                else:
                    selected_random_cols = random.sample(rand_pool, k=n_random_cols)

                content_cols = decision_cols + [chosen_attr] + selected_random_cols
                random.shuffle(content_cols)
                final_cols = ["Row ID"] + content_cols

                table_view = combined_df[final_cols]
                md_table = table_view.to_markdown(index=False)

                answer_row_id = int(
                    combined_df[combined_df["__is_answer__"] == True]["Row ID"].iloc[0]
                )
                target_attr_val = target_row[chosen_attr]

                # Convert values to numeric for calculation tasks
                col_num = pd.to_numeric(combined_df[chosen_attr], errors="coerce")
                target_val_num = pd.to_numeric(target_attr_val, errors="coerce")

                if pd.isna(target_val_num):
                    valid_for_all = False
                    break

                # Filter for grouping logic (Same Maker or Same Color)
                if group_type == "maker":
                    group_mask = (combined_df[BRAND_COL] == target_maker)
                else:
                    group_mask = (combined_df[COLOR_COL] == target_color)

                # ---- Task: Counting (Count items in group with attr > target) ----
                count_mask = group_mask & col_num.notna() & (col_num > float(target_val_num))
                count_value = int(count_mask.sum())

                # ---- Task: Averaging (Calculate mean of group) ----
                group_vals = col_num[group_mask & col_num.notna()]
                if len(group_vals) == 0:
                    valid_for_all = False
                    break
                mean_value = float(group_vals.mean())
                mean_str = f"{mean_value:.4f}".rstrip("0").rstrip(".")

                # Generate task-specific explanations
                if mode == "color":
                    loc_expl = f"The car is {target_color}. Row {answer_row_id} is the only one with Color '{target_color}'."
                elif mode == "maker":
                    loc_expl = f"The car is a {target_maker}. Row {answer_row_id} is the only one with Maker '{target_maker}'."
                else:
                    loc_expl = f"The car is a {target_maker} in {target_color}. Row {answer_row_id} is the only match for both attributes."

                attr_expl = f"In the table, the row matching the car in the image has {chosen_attr} = {target_attr_val}."

                if group_type == "maker":
                    group_desc = f"the cars that share the same Maker '{target_maker}'"
                else:
                    group_desc = f"the cars that share the same Color '{target_color}'"

                count_expl = f"Among {group_desc}, there are {count_value} cars whose {chosen_attr} is greater than that of the target car."
                mean_expl = f"Among {group_desc}, the average {chosen_attr} is {mean_str} computed over {len(group_vals)} cars."

                common_meta = {
                    "decision_mode": mode,
                    "n_rows": int(rs),
                    "base_sample_idx": base_idx,
                    "target_color": target_color,
                    "target_maker": target_maker,
                    "attribute": chosen_attr,
                    "reason_group_type": group_type,
                }

                # Construct JSONL items
                tmp_loc_items[rs] = {
                    "id": f"dvm_{mode}_loc_{rs}rows_{base_idx}",
                    "prompt": build_loc_prompt(md_table),
                    "image": target_row["__image_rel_path__"],
                    "ground_truth": {"answer": [str(answer_row_id)], "explanation": loc_expl},
                    "meta": common_meta,
                }

                tmp_attr_items[rs] = {
                    "id": f"dvm_{mode}_attr_{rs}rows_{base_idx}",
                    "prompt": build_attr_prompt(md_table, chosen_attr),
                    "image": target_row["__image_rel_path__"],
                    "ground_truth": {"answer": [str(target_attr_val)], "explanation": attr_expl},
                    "meta": common_meta,
                }

                tmp_count_items[rs] = {
                    "id": f"dvm_{mode}_count_{rs}rows_{base_idx}",
                    "prompt": build_count_prompt(md_table, chosen_attr, group_type),
                    "image": target_row["__image_rel_path__"],
                    "ground_truth": {"answer": [str(count_value)], "explanation": count_expl},
                    "meta": common_meta,
                }

                tmp_mean_items[rs] = {
                    "id": f"dvm_{mode}_mean_{rs}rows_{base_idx}",
                    "prompt": build_mean_prompt(md_table, chosen_attr, group_type),
                    "image": target_row["__image_rel_path__"],
                    "ground_truth": {"answer": [mean_str], "explanation": mean_expl},
                    "meta": common_meta,
                }

            if not valid_for_all:
                continue

            # Commit valid items to global lists
            for rs in ROW_SIZE_OPTIONS:
                loc_per_rowsize[rs].append(tmp_loc_items[rs])
                attr_per_rowsize[rs].append(tmp_attr_items[rs])
                count_per_rowsize[rs].append(tmp_count_items[rs])
                mean_per_rowsize[rs].append(tmp_mean_items[rs])

            pbar.update(1)

        pbar.close()

        # ------- 6) Write output files in JSONL format -------
        for rs in ROW_SIZE_OPTIONS:
            task_data = [
                (loc_per_rowsize[rs], "loc"),
                (attr_per_rowsize[rs], "attr"),
                (count_per_rowsize[rs], "count"),
                (mean_per_rowsize[rs], "mean")
            ]
            for data, suffix in task_data:
                filename = f"dvm_{mode}_{suffix}_{rs}rows.jsonl"
                path = os.path.join(OUTPUT_DIR, filename)
                print(f"Saving {len(data)} items to {path}...")
                with open(path, "w", encoding="utf-8") as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nAll files generated successfully!")


if __name__ == "__main__":
    generate_all_modes()