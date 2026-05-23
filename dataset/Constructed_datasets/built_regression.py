import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ================= Configuration Area =================
CONFIG = {
    "MIMIC_IV_HOSP": "../physionet.org/files/mimiciv/2.2/hosp",
    "MIMIC_IV_ICU": "../physionet.org/files/mimiciv/2.2/icu",
    "MIMIC_CXR": "../physionet.org/files/mimic-cxr-jpg/2.0.0",
    "OUTPUT_DIR": "/mnt/hdd/jiazy/ehrxqa/regression",
    "TARGET_COUNT": 100000,
}

# Define clinical features to extract
FEATURES_CONFIG = {
    'chartevents': {
        'Temperature': [223761, 223762], 
        'HeartRate': [220045],
        'RespRate': [220210], # Note: This feature needs to be excluded in RR task
        'SpO2': [220277],
        'SysBP': [220179, 220050], 
        'DiaBP': [220180, 220051],
    },
    'labevents': {
        'WBC': [51301, 51300], 
        'Hemoglobin': [51222], 
        'Platelet': [51265],   
        'Glucose': [50931],    
        'Creatinine': [50912],
        'Sodium': [50983],
        'Potassium': [50971]
    }
}
# ===========================================

def get_file_path(base_dir, filename):
    path_gz = os.path.join(base_dir, filename + ".gz")
    path_csv = os.path.join(base_dir, filename)
    if os.path.exists(path_gz): return path_gz
    if os.path.exists(path_csv): return path_csv
    raise FileNotFoundError(f"Cannot find {filename} in {base_dir}")

def generate_image_path(row):
    subject_id = str(int(row['subject_id']))
    study_id = str(int(row['study_id']))
    image_id = str(row['image_id'])
    return f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{image_id}.jpg"

def split_data(df):
    print("    Splitting data into train (80%), valid (10%), test (10%)...")
    subjects = df['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n = len(subjects)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    
    train_subj = set(subjects[:n_train])
    valid_subj = set(subjects[n_train:n_train+n_valid])
    
    def get_split(sid):
        if sid in train_subj: return 'train'
        if sid in valid_subj: return 'valid'
        return 'test'
        
    df['split'] = df['subject_id'].apply(get_split)
    print("    Split stats:")
    print(df['split'].value_counts())
    return df

def extract_clinical_features(df_cohort, exclude_features=None, lookback_hours=24):
    """
    Generic feature extraction function
    exclude_features: list of str, feature names to exclude (e.g., exclude RespRate in RR prediction task)
    """
    print(f"    Extracting Clinical Features (Window: +/- {lookback_hours} hours)...")
    valid_hadms = set(df_cohort['hadm_id'].unique())
    
    # Prepare ItemID lists
    target_chart_ids = []
    target_lab_ids = []
    item_map = {}

    # Parse Chartevents configuration
    for name, ids in FEATURES_CONFIG['chartevents'].items():
        if exclude_features and name in exclude_features:
            continue
        target_chart_ids.extend(ids)
        for i in ids: item_map[i] = name
            
    # Parse Labevents configuration
    for name, ids in FEATURES_CONFIG['labevents'].items():
        if exclude_features and name in exclude_features:
            continue
        target_lab_ids.extend(ids)
        for i in ids: item_map[i] = name

    extracted_data = []

    # --- 1. Vitals ---
    print("    Scanning Vital Signs...")
    chart_path = get_file_path(CONFIG["MIMIC_IV_ICU"], "chartevents.csv")
    chunk_size = 5000000
    
    # No with syntax, compatible with older Pandas
    reader = pd.read_csv(chart_path, chunksize=chunk_size, usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'])
    
    for chunk in tqdm(reader, desc="    Reading Vitals"):
        chunk = chunk[chunk['hadm_id'].isin(valid_hadms)]
        chunk = chunk[chunk['itemid'].isin(target_chart_ids)]
        chunk = chunk.dropna(subset=['valuenum'])
        
        # Temperature conversion F -> C
        if 223761 in chunk['itemid'].values:
            f_mask = chunk['itemid'] == 223761
            chunk.loc[f_mask, 'valuenum'] = (chunk.loc[f_mask, 'valuenum'] - 32) * 5/9
            chunk.loc[f_mask, 'itemid'] = 223762 
            
        if not chunk.empty:
            extracted_data.append(chunk)

    # --- 2. Labs ---
    print("    Scanning Labs...")
    lab_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "labevents.csv")
    
    reader_lab = pd.read_csv(lab_path, chunksize=chunk_size, usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'])
    
    for chunk in tqdm(reader_lab, desc="    Reading Labs"):
        chunk = chunk[chunk['hadm_id'].isin(valid_hadms)]
        chunk = chunk[chunk['itemid'].isin(target_lab_ids)]
        chunk = chunk.dropna(subset=['valuenum'])
        
        if not chunk.empty:
            extracted_data.append(chunk)

    if not extracted_data:
        print("    Warning: No clinical features found.")
        return df_cohort

    # --- 3. Merge & Pivot ---
    print("    Aggregating features...")
    df_features = pd.concat(extracted_data)
    df_features['charttime'] = pd.to_datetime(df_features['charttime'])
    df_features['feature_name'] = df_features['itemid'].map(item_map)
    
    cohort_time = df_cohort[['hadm_id', 'study_id', 'studydatetime']].drop_duplicates()
    merged = pd.merge(df_features, cohort_time, on='hadm_id', how='inner')
    
    merged['hours_diff'] = (merged['charttime'] - merged['studydatetime']).dt.total_seconds() / 3600
    merged = merged[abs(merged['hours_diff']) <= lookback_hours]
    
    df_pivot = pd.pivot_table(merged, values='valuenum', index='study_id', columns='feature_name', aggfunc='mean')
    
    df_final = pd.merge(df_cohort, df_pivot, on='study_id', how='left')
    return df_final

def step_1_link_cxr_to_hadm():
    """Step 1: Full-scale alignment (no limit)"""
    print(">>> Step 1: Aligning CXR to Admissions (Full Scale)...")
    
    cxr_meta_path = get_file_path(CONFIG["MIMIC_CXR"], "mimic-cxr-2.0.0-metadata.csv")
    df_cxr = pd.read_csv(cxr_meta_path, usecols=['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime', 'ViewPosition'])
    df_cxr.rename(columns={'dicom_id': 'image_id'}, inplace=True)
    df_cxr = df_cxr[df_cxr['ViewPosition'].isin(['AP', 'PA'])]
    
    # Full-scale data processing
    print(f"    Loaded {len(df_cxr)} CXR studies candidates.")

    df_cxr['studydatetime'] = pd.to_datetime(
        df_cxr['StudyDate'].astype(str) + ' ' + 
        df_cxr['StudyTime'].astype(str).str.split('.').str[0].str.zfill(6),
        format='%Y%m%d %H%M%S', errors='coerce'
    )
    
    trans_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "transfers.csv")
    df_trans = pd.read_csv(trans_path, usecols=['subject_id', 'hadm_id', 'intime', 'outtime'])
    df_trans = df_trans.dropna(subset=['hadm_id', 'intime', 'outtime'])
    df_trans['intime'] = pd.to_datetime(df_trans['intime'])
    df_trans['outtime'] = pd.to_datetime(df_trans['outtime'])
    
    valid_subjects = set(df_cxr['subject_id'].unique())
    df_trans = df_trans[df_trans['subject_id'].isin(valid_subjects)]
    
    print("    Merging...")
    merged = pd.merge(df_cxr, df_trans, on='subject_id', how='left')
    mask = (merged['studydatetime'] >= merged['intime']) & (merged['studydatetime'] <= merged['outtime'])
    aligned = merged[mask].copy()
    aligned = aligned.drop_duplicates(subset=['study_id', 'image_id'])
    
    print(f"    Aligned {len(aligned)} images (Full Set).")
    return aligned[['subject_id', 'study_id', 'image_id', 'hadm_id', 'studydatetime', 'ViewPosition']]

def step_2_build_los_dataset(df_aligned):
    """Task 1: LOS prediction (with rich features)"""
    print("\n>>> Step 2: Building LOS Dataset (Enriched)...")
    
    adm_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "admissions.csv")
    df_adm = pd.read_csv(adm_path, usecols=['hadm_id', 'admittime', 'dischtime', 'admission_type', 'admission_location', 'insurance', 'marital_status'])
    df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])
    df_adm['dischtime'] = pd.to_datetime(df_adm['dischtime'])
    
    pat_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "patients.csv")
    df_pat = pd.read_csv(pat_path, usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    
    df_final = pd.merge(df_aligned, df_adm, on='hadm_id', how='inner')
    df_final = pd.merge(df_final, df_pat, on='subject_id', how='inner')
    
    df_final['target_los_days'] = (df_final['dischtime'] - df_final['admittime']).dt.total_seconds() / (3600 * 24)
    df_final['admit_year'] = df_final['admittime'].dt.year
    df_final['age'] = df_final['anchor_age'] + (df_final['admit_year'] - df_final['anchor_year'])
    
    df_final = df_final[df_final['target_los_days'] > 0]
    df_final = df_final[(df_final['studydatetime'] >= df_final['admittime']) & (df_final['studydatetime'] <= df_final['dischtime'])]

    # [New] Feature extraction (LOS task can include all features, including respiratory rate)
    df_final = extract_clinical_features(df_final, lookback_hours=24)
    
    # Clean missing values
    print("    Filtering missing values...")
    feature_cols = list(FEATURES_CONFIG['chartevents'].keys()) + list(FEATURES_CONFIG['labevents'].keys())
    # Ensure columns exist
    feature_cols = [c for c in feature_cols if c in df_final.columns]
    
    initial_len = len(df_final)
    df_final = df_final.dropna(subset=feature_cols, how='any')
    print(f"    Dropped {initial_len - len(df_final)} rows. Remaining: {len(df_final)}")
    
    # Truncate
    if len(df_final) > CONFIG["TARGET_COUNT"]:
        df_final = df_final.sample(n=CONFIG["TARGET_COUNT"], random_state=42).reset_index(drop=True)
        
    df_final['image_path'] = df_final.apply(generate_image_path, axis=1)
    df_final = split_data(df_final)
    
    cols = ['split', 'subject_id', 'study_id', 'image_path', 'gender', 'age', 'ViewPosition', 
            'admission_type', 'admission_location', 'target_los_days'] + feature_cols
            
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    save_path = os.path.join(CONFIG["OUTPUT_DIR"], "los_dataset_100k_enriched.csv")
    df_final[cols].to_csv(save_path, index=False)
    print(f"    -> Saved: {save_path}")

def step_3_build_rr_dataset(df_aligned):
    """Task 2: Respiratory rate prediction (with rich features)"""
    print("\n>>> Step 3: Building RR Dataset (Enriched)...")
    
    # 1. Build Target (respiratory rate)
    chart_path = get_file_path(CONFIG["MIMIC_IV_ICU"], "chartevents.csv")
    valid_hadms = set(df_aligned['hadm_id'].unique())
    
    print("    Scanning Target RR values (Full Scale)...")
    rr_records = []
    chunk_size = 5000000 
    
    reader = pd.read_csv(chart_path, chunksize=chunk_size, usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'])
    
    # To ensure quality, it's recommended to read as many Targets as possible, or read full-scale
    # Here we set to stop once enough raw records are scanned, or you can remove break to run full-scale
    total_found_raw = 0
    for chunk in tqdm(reader, desc="Scanning Targets"):
        chunk_filtered = chunk[
            (chunk['itemid'] == 220210) & 
            (chunk['hadm_id'].isin(valid_hadms)) &
            (chunk['valuenum'].notna())
        ]
        if not chunk_filtered.empty:
            rr_records.append(chunk_filtered)
            total_found_raw += len(chunk_filtered)
            
        # If raw records exceed 2 million, usually sufficient to generate >100k aggregated samples
        if total_found_raw > CONFIG["TARGET_COUNT"] * 20: 
            print("    Found sufficient raw targets, proceeding...")
            break
            
    if not rr_records:
        return

    df_rr = pd.concat(rr_records)
    df_rr['charttime'] = pd.to_datetime(df_rr['charttime'])
    
    merged = pd.merge(df_aligned, df_rr, on='hadm_id', how='inner')
    merged['hours_diff'] = (merged['charttime'] - merged['studydatetime']).dt.total_seconds() / 3600
    merged = merged[abs(merged['hours_diff']) <= 24]
    
    df_target = merged.groupby(['subject_id', 'study_id', 'image_id', 'ViewPosition', 'studydatetime', 'hadm_id']).agg(
        target_rr_value=('valuenum', 'mean')
    ).reset_index()
    
    print(f"    Target candidates found: {len(df_target)}")

    # 2. [New] Extract features
    # Key: When predicting respiratory rate, input features must not include respiratory rate!
    # exclude_features=['RespRate']
    df_target = extract_clinical_features(df_target, exclude_features=['RespRate'], lookback_hours=24)
    
    # 3. Clean
    print("    Filtering missing values...")
    # Get feature list (note RespRate is excluded)
    feature_cols = list(FEATURES_CONFIG['chartevents'].keys()) + list(FEATURES_CONFIG['labevents'].keys())
    feature_cols = [c for c in feature_cols if c != 'RespRate' and c in df_target.columns]
    
    initial_len = len(df_target)
    df_target = df_target.dropna(subset=feature_cols, how='any')
    print(f"    Dropped {initial_len - len(df_target)} rows. Remaining: {len(df_target)}")
    
    # Add basic information
    pat_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "patients.csv")
    df_pat = pd.read_csv(pat_path, usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    df_final = pd.merge(df_target, df_pat, on='subject_id', how='inner')
    df_final['admit_year'] = df_final['studydatetime'].dt.year
    df_final['age'] = df_final['anchor_age'] + (df_final['admit_year'] - df_final['anchor_year'])
    
    # Truncate
    if len(df_final) > CONFIG["TARGET_COUNT"]:
        df_final = df_final.sample(n=CONFIG["TARGET_COUNT"], random_state=42).reset_index(drop=True)
        
    df_final['image_path'] = df_final.apply(generate_image_path, axis=1)
    df_final = split_data(df_final)
    
    cols = ['split', 'subject_id', 'study_id', 'image_path', 'gender', 'age', 'ViewPosition', 'target_rr_value'] + feature_cols
    
    save_path = os.path.join(CONFIG["OUTPUT_DIR"], "rr_dataset_100k_enriched.csv")
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    df_final[cols].to_csv(save_path, index=False)
    print(f"    -> Saved: {save_path}")

if __name__ == "__main__":
    # 1. Must run this first to get full-scale aligned data
    df_aligned = step_1_link_cxr_to_hadm()
    
    # 2. Generate LOS dataset
    step_2_build_los_dataset(df_aligned)
    
    # 3. Generate RR dataset
    step_3_build_rr_dataset(df_aligned)
    
    print("\nRegression dataset construction complete!")