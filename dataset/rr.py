import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from PIL import Image
from tqdm import tqdm

# Ensure PIL version compatibility
try:
    LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_RESAMPLE = Image.LANCZOS

# --- 1. Define constants and paths ---
# [Configuration] Input paths (RR dataset)
CSV_FILE = "/mnt/hdd/jiazy/mimic/regression/rr/rr_dataset_100k.csv"
# Image root directory remains unchanged
IMAGE_ROOT = "/mnt/hdd/jiazy/mimic/image"

# [Configuration] Output directory (RR feature output)
OUTPUT_DIR = "/mnt/hdd/jiazy/mimic/regression/rr/features"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. Define feature columns ---
# Continuous features (13 features, RespRate removed)
CONTINUOUS_COLS = [
    'age', 'Temperature', 'HeartRate', 'SpO2', 'SysBP', 'DiaBP', 
    'WBC', 'Hemoglobin', 'Platelet', 'Glucose', 'Creatinine', 
    'Sodium', 'Potassium'
]

# Categorical features (2 features)
CATEGORICAL_COLS = [
    'gender', 'ViewPosition'
]

# Label column (respiratory rate value)
LABEL_COL = 'target_rr_value'

# Columns to use
USE_COLS = ['split', 'image_path', LABEL_COL] + CONTINUOUS_COLS + CATEGORICAL_COLS

# --- 3. Image processing helper function ---
def process_and_save_image(rel_path):
    """
    Load image, resize to 224x224 (without cropping), and save as .npy file.
    """
    if pd.isna(rel_path):
        return None

    # Construct full path
    full_img_path = os.path.join(IMAGE_ROOT, rel_path)
    
    # Construct .npy save path
    npy_path = full_img_path.replace(".jpg", ".npy").replace(".jpeg", ".npy").replace(".png", ".npy")

    # If .npy already exists, return path directly
    if os.path.exists(npy_path):
        return npy_path

    if not os.path.exists(full_img_path):
        return None 

    try:
        img = Image.open(full_img_path)
        img_resized = img.resize((224, 224), resample=LANCZOS_RESAMPLE)
        
        # Convert to RGB
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')
            
        np_img = np.array(img_resized)
        np.save(npy_path, np_img)
        
        return npy_path
    except Exception as e:
        print(f"Failed to process image {full_img_path}: {e}")
        return None

def preprocess_mimic_rr_flow():
    """
    Preprocess MIMIC-CXR (RR Regression) dataset
    """
    print("Starting MIMIC-CXR (RR Regression) preprocessing pipeline...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # --- 1. Load data ---
    print(f"Loading metadata {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, usecols=USE_COLS)
    print(f"Original data count: {len(df)}")

    # --- 2. Basic data cleaning ---
    df = df.dropna(subset=[LABEL_COL])
    print(f"Data count after filtering missing labels: {len(df)}")
    
    # --- 3. Categorical feature encoding ---
    print("Encoding categorical features...")
    cat_dims = []
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(-1) 
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
        num_cats = int(df[col].max() + 1)
        cat_dims.append(num_cats if num_cats > 0 else 1)
    
    print(f"Categorical feature dimensions: {dict(zip(CATEGORICAL_COLS, cat_dims))}")

    # --- 4. Dataset splitting ---
    print("Splitting dataset based on 'split' column...")
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'valid'].copy()
    test_df = df[df['split'] == 'test'].copy()

    print(f"Split results: Train {len(train_df)}, Validation {len(val_df)}, Test {len(test_df)}")

    # --- 5. Continuous feature standardization ---
    print("Standardizing continuous features...")
    if CONTINUOUS_COLS:
        scaler = StandardScaler()
        # Fill missing values
        train_df[CONTINUOUS_COLS] = train_df[CONTINUOUS_COLS].fillna(0)
        val_df[CONTINUOUS_COLS] = val_df[CONTINUOUS_COLS].fillna(0)
        test_df[CONTINUOUS_COLS] = test_df[CONTINUOUS_COLS].fillna(0)

        scaler.fit(train_df[CONTINUOUS_COLS])
        
        train_df[CONTINUOUS_COLS] = scaler.transform(train_df[CONTINUOUS_COLS])
        val_df[CONTINUOUS_COLS] = scaler.transform(val_df[CONTINUOUS_COLS])
        test_df[CONTINUOUS_COLS] = scaler.transform(test_df[CONTINUOUS_COLS])

    # --- 6. Image processing ---
    print("Processing images (Resize & Save .npy)...")
    tqdm.pandas(desc="Train Images")
    train_df['npy_path'] = train_df['image_path'].progress_apply(process_and_save_image)
    
    tqdm.pandas(desc="Val Images")
    val_df['npy_path'] = val_df['image_path'].progress_apply(process_and_save_image)
    
    tqdm.pandas(desc="Test Images")
    test_df['npy_path'] = test_df['image_path'].progress_apply(process_and_save_image)

    # Filter invalid images
    len_before = len(train_df) + len(val_df) + len(test_df)
    train_df = train_df.dropna(subset=['npy_path'])
    val_df = val_df.dropna(subset=['npy_path'])
    test_df = test_df.dropna(subset=['npy_path'])
    len_after = len(train_df) + len(val_df) + len(test_df)
    
    if len_before != len_after:
        print(f"⚠️ Warning: Filtered {len_before - len_after} rows with missing or corrupted images.")

    # --- 7. Save output ---
    print(f"[Final] Saving processed files to {OUTPUT_DIR} ...")

    # Save tabular_lengths
    tabular_lengths = cat_dims + [1] * len(CONTINUOUS_COLS)
    torch.save(tabular_lengths, os.path.join(OUTPUT_DIR, "tabular_lengths.pt"))
    print(f"Feature dimensions saved: categorical {len(cat_dims)} + continuous {len(CONTINUOUS_COLS)}")
    
    for split_name, df_split in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        # 1. Save features CSV
        features_path = os.path.join(OUTPUT_DIR, f"{split_name}_features.csv")
        cols_to_save = CATEGORICAL_COLS + CONTINUOUS_COLS
        df_split[cols_to_save].to_csv(features_path, index=False, header=False)
        
        # 2. Save labels Tensor
        labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.pt")
        labels_tensor = torch.tensor(df_split[LABEL_COL].values, dtype=torch.float32)
        torch.save(labels_tensor, labels_path)
        
        # 3. Save paths List
        paths_path = os.path.join(OUTPUT_DIR, f"{split_name}_paths.pt")
        npy_path_list = df_split['npy_path'].tolist()
        torch.save(npy_path_list, paths_path)

    print("✅ MIMIC (RR) preprocessing completed!")

if __name__ == "__main__":
    preprocess_mimic_rr_flow()