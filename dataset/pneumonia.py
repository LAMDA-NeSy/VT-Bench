import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# ================= Configuration Area =================

CSV_PATH = '/mnt/hdd/jiazy/mimic/classification/pneumonia_dataset_100k.csv'
IMAGE_ROOT = '/mnt/hdd/jiazy/mimic/image'
OUTPUT_DIR = '/mnt/hdd/jiazy/mimic/classification/features'

IMG_SIZE = (224, 224) 

TARGET_COL = 'target_pneumonia'
SPLIT_COL = 'split'
CAT_FEATURES = ['gender', 'ViewPosition']
CONT_FEATURES = [
    'age', 'Temperature', 'HeartRate', 'RespRate', 'SpO2', 
    'SysBP', 'WBC', 'Hemoglobin', 'Platelet', 'Glucose', 'Creatinine'
]

# ================= Main Processing Logic =================

def preprocess_final():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # --- Step 1: Basic cleaning ---
    print("Step 1: Cleaning data (NaN & path checking)...")
    check_cols = CAT_FEATURES + CONT_FEATURES + [TARGET_COL, 'image_path', SPLIT_COL]
    df = df.dropna(subset=check_cols)
    
    # Path checking
    def check_jpg_exists(path_suffix):
        return os.path.exists(os.path.join(IMAGE_ROOT, str(path_suffix)))

    tqdm.pandas(desc="Checking Paths")
    mask_exists = df['image_path'].progress_apply(check_jpg_exists)
    df = df[mask_exists].copy()
    print(f"   -> Valid samples remaining after cleaning: {len(df)}")

    # --- Step 2: Feature engineering (strict mode) ---
    print("Step 2: Processing tabular features (preventing data leakage)...")
    
    # 2.1 Categorical encoding (Fitting on all data is safe, as it only builds a mapping table)
    cat_counts = [] 
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_counts.append(len(le.classes_))
        
    # 2.2 Continuous feature standardization (!!! Key modification !!!)
    # Only use 'train' data to calculate mean and variance
    scaler = StandardScaler()
    
    train_mask = df[SPLIT_COL] == 'train'
    if train_mask.sum() == 0:
        raise ValueError("No samples with split='train' found, cannot perform standardization fitting!")
        
    # Fit only on TRAIN
    print("   -> Calculating mean and variance based on Train set...")
    scaler.fit(df.loc[train_mask, CONT_FEATURES])
    
    # Transform ALL (Train, Valid, Test)
    df[CONT_FEATURES] = scaler.transform(df[CONT_FEATURES])
    print("   -> Standardization applied to all data.")

    # 2.3 Generate tabular_lengths
    lengths_list = cat_counts + [1] * len(CONT_FEATURES)
    torch.save(lengths_list, os.path.join(OUTPUT_DIR, 'tabular_lengths.pt'))

    # --- Step 3: Convert images to .npy (with skip logic) ---
    print("Step 3: Converting images (Resize -> Save NPY)...")
    
    npy_paths = []
    process_count = 0
    skip_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting Images"):
        orig_path = os.path.join(IMAGE_ROOT, row['image_path'])
        save_path = os.path.splitext(orig_path)[0] + '.npy'
        
        # Record path, whether newly generated or not
        npy_paths.append(save_path)
        
        # If file already exists, skip processing (save time)
        if os.path.exists(save_path):
            skip_count += 1
            continue
            
        try:
            with Image.open(orig_path) as img:
                img = img.convert('RGB')
                img = img.resize(IMG_SIZE, Image.BILINEAR)
                # Save as uint8 (0-255) to save space
                img_array = np.array(img) 
                
            np.save(save_path, img_array)
            process_count += 1
            
        except Exception as e:
            print(f"Error processing {orig_path}: {e}")
            # If this image processing fails, change the path just added to the list to None for later filtering
            npy_paths[-1] = None
            
    df['npy_path'] = npy_paths
    
    # Clean again to remove failed conversions
    df = df.dropna(subset=['npy_path'])
    print(f"   -> Image processing completed: {process_count} newly generated, {skip_count} skipped (already exist).")

    # --- Step 4: Save final files ---
    print("Step 4: Saving Dataset files...")
    
    feature_cols = CAT_FEATURES + CONT_FEATURES
    splits = ['train', 'valid', 'test'] 
    
    for split_name in splits:
        # Handle possible naming differences
        if split_name == 'valid':
            subset = df[df[SPLIT_COL].isin(['valid', 'validate'])]
        else:
            subset = df[df[SPLIT_COL] == split_name]
            
        if len(subset) == 0:
            continue
            
        print(f"   Saving {split_name}: {len(subset)} samples")
        
        # A. Features CSV
        subset[feature_cols].to_csv(
            os.path.join(OUTPUT_DIR, f'{split_name}_features.csv'),
            header=False, index=False
        )
        
        # B. Labels PT (!!! Modified to Long for classification !!!)
        labels_tensor = torch.tensor(subset[TARGET_COL].values, dtype=torch.long)
        torch.save(labels_tensor, os.path.join(OUTPUT_DIR, f'{split_name}_labels.pt'))
        
        # C. Paths PT
        paths_list = subset['npy_path'].tolist()
        torch.save(paths_list, os.path.join(OUTPUT_DIR, f'{split_name}_paths.pt'))

    print("-" * 30)
    print("All preprocessing work completed.")

if __name__ == "__main__":
    preprocess_final()