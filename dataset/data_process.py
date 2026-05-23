import os
import shutil
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Tuple

# ==========================================
# 1. Global Configuration
# ==========================================

# --- Path Configuration ---
# Original CSV file path
RAW_CSV_PATH = "/home/admin_czj/jiazy/PetFinder_datasets/train_with_sentiment_clean.csv"
# Folder where all original images are located
RAW_SOURCE_IMAGE_DIR = "/home/admin_czj/jiazy/PetFinder_datasets/raw_dataset/petfinder_adoptionprediction/train_images"

# Total output root directory
OUTPUT_ROOT = "/data1/jiazy/tab_image_bench/PetFinder_datasets"
# Dataset name (will be created as a subdirectory)
DATASET_NAME = "petfinder_adoptionprediction"

# Construct final dataset directory
DATA_DIRECTORY = os.path.join(OUTPUT_ROOT, "dataset_test", DATASET_NAME)

# --- Parameter Configuration ---
LABEL_COLUMN_NAME = 'AdoptionSpeed'
ID_COLUMNS_TO_DROP = ['PetID', 'Description']
VALID_ADOPTION_SPEEDS = {0, 1, 2, 3, 4}
IMAGE_SIZE = 224  # Unified image size (224 recommended)

# ==========================================
# 2. Helper Function Definitions
# ==========================================

def text_stats(desc: str) -> Tuple[int, float, int]:
    """Generate simple text statistical features from the Description."""
    if not isinstance(desc, str) or not desc.strip():
        return 0, 0.0, 0
    desc = desc.strip()
    words = desc.split()
    desc_length = len(desc)
    desc_words = len(words)
    avg_len = (sum(len(w) for w in words) / desc_words) if desc_words > 0 else 0.0
    return desc_length, avg_len, desc_words

def copy_images_for_split(df: pd.DataFrame, source_img_dir: str, dest_img_dir: str, split_name: str):
    """Copy images to the corresponding split directory (train/valid/test)."""
    print(f"\n--- Copying images for the [{split_name}] set ---")
    os.makedirs(dest_img_dir, exist_ok=True)
    
    copied_count = 0
    # Assuming PetID corresponds to the filename PetID-1.jpg
    for pet_id in tqdm(df['PetID'], desc=f"Copying {split_name} images"):
        source_path = os.path.join(source_img_dir, f"{pet_id}-1.jpg")
        dest_path = os.path.join(dest_img_dir, f"{pet_id}-1.jpg")
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            
    print(f"✅ Successfully copied {copied_count}/{len(df)} images to: {dest_img_dir}")

def process_image_to_npy(source_dir, target_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Read images, resize, normalize, and save as .npy files."""
    os.makedirs(target_dir, exist_ok=True)
    
    image_filenames = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
    if not image_filenames:
        print(f"[Skip] No .jpg files found in directory {source_dir}.")
        return

    print(f"--- Converting images: {source_dir} -> {target_dir} (.npy) ---")

    for filename in tqdm(image_filenames, desc=f"Processing images"):
        jpg_path = os.path.join(source_dir, filename)
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        npy_path = os.path.join(target_dir, npy_filename)

        try:
            with Image.open(jpg_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Lanczos Resizing
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to float32 and normalize to 0-1
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Shape check (H, W, C)
            if img_array.shape != (target_size[1], target_size[0], 3):
                # Transpose may be needed in rare cases, but PIL resize usually returns (H, W, C)
                pass 
            
            np.save(npy_path, img_array)

        except Exception as e:
            print(f"Error processing file {jpg_path}: {e}")

# ==========================================
# 3. Core Processing Workflow
# ==========================================

def step_1_clean_and_split():
    """Step 1: Data cleaning, CSV splitting, and raw image copying."""
    print("\n===== [Step 1] Data Cleaning and Splitting =====")
    
    if not os.path.isfile(RAW_CSV_PATH):
        raise FileNotFoundError(f"Original CSV not found: {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)
    
    # Feature Engineering: Text statistics
    stats = df["Description"].apply(text_stats)
    df["desc_length"], df["average_word_length"], df["desc_words"] = zip(*stats)

    # Filter: Ensure image exists
    print("Checking image existence...")
    df['image_path'] = df['PetID'].apply(lambda pid: os.path.join(RAW_SOURCE_IMAGE_DIR, f"{pid}-1.jpg"))
    df = df[df['image_path'].apply(os.path.exists)].copy()
    df = df.drop(columns=['image_path'])
    
    # Filter: Drop invalid Names and filter Labels
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])
    df = df[df[LABEL_COLUMN_NAME].isin(VALID_ADOPTION_SPEEDS)]
    
    print(f"Cleaned dataset size: {len(df)}")

    # Split 8:1:1
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df[LABEL_COLUMN_NAME], random_state=2022
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[LABEL_COLUMN_NAME], random_state=2022
    )

    # Save CSV
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_DIRECTORY, "dataset_train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIRECTORY, "dataset_valid.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIRECTORY, "dataset_test.csv"), index=False)
    
    # Copy images (JPG) to split folders
    # Define image directories
    dirs = {
        "train": os.path.join(DATA_DIRECTORY, "train_image"),
        "valid": os.path.join(DATA_DIRECTORY, "valid_image"),
        "test":  os.path.join(DATA_DIRECTORY, "test_image")
    }
    
    copy_images_for_split(train_df, RAW_SOURCE_IMAGE_DIR, dirs["train"], "train")
    copy_images_for_split(val_df, RAW_SOURCE_IMAGE_DIR, dirs["valid"], "valid")
    copy_images_for_split(test_df, RAW_SOURCE_IMAGE_DIR, dirs["test"], "test")
    
    return dirs # Return image directory paths for subsequent steps

def step_2_tabular_processing():
    """Step 2: Process tabular features (Encoding, Scaling) -> generate .pt files."""
    print("\n===== [Step 2] Tabular Data Feature Processing =====")
    
    train_df = pd.read_csv(os.path.join(DATA_DIRECTORY, "dataset_train.csv"))
    valid_df = pd.read_csv(os.path.join(DATA_DIRECTORY, "dataset_valid.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIRECTORY, "dataset_test.csv"))

    feature_columns = [col for col in train_df.columns if col not in [LABEL_COLUMN_NAME] + ID_COLUMNS_TO_DROP]
    
    categorical_cols = train_df[feature_columns].select_dtypes(include=['object']).columns
    numerical_cols = train_df[feature_columns].select_dtypes(include=['number']).columns

    print(f"Numerical Features: {len(numerical_cols)}, Categorical Features: {len(categorical_cols)}")

    # 1. Create mapping (based on training set only)
    category_mappings = {}
    for col in categorical_cols:
        categories = train_df[col].astype('category').cat.categories
        category_mappings[col] = {category: i for i, category in enumerate(categories)}

    # 2. Create Scaler (based on training set only)
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        scaler.fit(train_df[numerical_cols])

    # 3. Apply transformations and save
    all_dfs = {'train': train_df, 'valid': valid_df, 'test': test_df}
    
    for split, df in all_dfs.items():
        proc_df = df.copy()
        
        # Apply Categorical Mapping
        for col, mapping in category_mappings.items():
            proc_df[col] = proc_df[col].map(mapping).fillna(-1)
        
        # Apply Scaler
        if len(numerical_cols) > 0:
            proc_df[numerical_cols] = scaler.transform(proc_df[numerical_cols])
            
        proc_df[feature_columns] = proc_df[feature_columns].fillna(0)

        # Convert to Tensor
        features = torch.tensor(proc_df[feature_columns].values, dtype=torch.float32)
        labels = torch.tensor(proc_df[LABEL_COLUMN_NAME].values, dtype=torch.long)

        torch.save(features, os.path.join(DATA_DIRECTORY, f"features_{split}.pt"))
        torch.save(labels, os.path.join(DATA_DIRECTORY, f"labels_{split}.pt"))
        print(f"  -> Saved {split}: Features {features.shape}, Labels {labels.shape}")

    # 4. Save feature length information (for Embedding layer use)
    field_lengths = [1] * len(numerical_cols) # Numerical feature length is 1
    for col in categorical_cols:
        field_lengths.append(train_df[col].nunique()) # Categorical feature length
    
    torch.save(field_lengths, os.path.join(DATA_DIRECTORY, "tabular_lengths.pt"))
    print("  -> Saved tabular_lengths.pt")

def step_3_image_preprocessing(img_dirs):
    """Step 3: Image preprocessing (.jpg -> .npy) with Resize and Normalize."""
    print("\n===== [Step 3] Image Preprocessing (.jpg -> .npy) =====")
    
    # Iterate through train, valid, and test directories
    for split_name, folder_path in img_dirs.items():
        # Here we save .npy in the same directory as the original jpg for easier management
        # Alternatively, target_dir could be set to a new folder like train_image_npy
        target_dir = folder_path 
        process_image_to_npy(folder_path, target_dir)

def step_4_generate_paths(img_dirs):
    """Step 4: Generate file path indices (.pt)."""
    print("\n===== [Step 4] Generating File Path Index =====")
    
    splits = ["train", "valid", "test"]
    
    for split in splits:
        csv_path = os.path.join(DATA_DIRECTORY, f"dataset_{split}.csv")
        df = pd.read_csv(csv_path)
        pet_ids = df['PetID'].tolist()
        
        # Note: We point to .npy files here as they are the product of Step 3
        # If the model needs to read original jpgs, change the extension to .jpg
        image_paths = []
        image_dir = img_dirs[split]
        
        missing = 0
        for pid in pet_ids:
            # Prioritize looking for .npy
            npy_path = os.path.join(image_dir, f"{pid}-1.npy")
            if os.path.exists(npy_path):
                image_paths.append(npy_path)
            else:
                # Fallback: if only jpg exists (though step 3 should have generated npy)
                jpg_path = os.path.join(image_dir, f"{pid}-1.jpg")
                if os.path.exists(jpg_path):
                    image_paths.append(jpg_path)
                else:
                    missing += 1
        
        if missing > 0:
            print(f"[Warning] {split} set has {missing} missing files!")

        save_path = os.path.join(DATA_DIRECTORY, f"paths_{split}.pt")
        torch.save(image_paths, save_path)
        print(f"  -> Saved paths_{split}.pt (Count: {len(image_paths)})")

def step_5_validation():
    """Step 5: Final Validation."""
    print("\n===== [Step 5] Final File Validation =====")
    required_files = [
        "dataset_train.csv", "features_train.pt", "labels_train.pt", "paths_train.pt",
        "dataset_valid.csv", "features_valid.pt", "labels_valid.pt", "paths_valid.pt",
        "dataset_test.csv",  "features_test.pt",  "labels_test.pt",  "paths_test.pt",
        "tabular_lengths.pt"
    ]
    
    all_exist = True
    for f in required_files:
        fpath = os.path.join(DATA_DIRECTORY, f)
        if not os.path.exists(fpath):
            print(f"❌ Missing file: {f}")
            all_exist = False
    
    if all_exist:
        print("🎉 All critical files validated! Data preprocessing complete.")
    else:
        print("⚠️ Some files are missing, please check the error logs above.")

# ==========================================
# Main Program Entry
# ==========================================

if __name__ == "__main__":
    # Ensure the output root directory exists
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    try:
        # 1. Cleaning and Splitting (returns dictionary of image folder paths)
        image_directories = step_1_clean_and_split()
        
        # 2. Tabular Processing
        step_2_tabular_processing()
        
        # 3. Image conversion to NPY
        step_3_image_preprocessing(image_directories)
        
        # 4. Generate path lists
        step_4_generate_paths(image_directories)
        
        # 5. Validation
        step_5_validation()
        
    except Exception as e:
        print(f"\n🚨 Critical error occurred, program terminated:\n{e}")
        import traceback
        traceback.print_exc()