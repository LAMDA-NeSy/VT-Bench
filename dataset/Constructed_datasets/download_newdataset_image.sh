#!/bin/bash

# ================= Configuration Area =================
# 1. Define CSV file paths (based on the paths generated in the previous step)
CSV_LOS="dataset_builder/regression_dataset_100k/los_dataset_100k.csv"
CSV_RR="dataset_builder/regression_dataset_100k/rr_dataset_100k.csv"
CSV_PNEUMONIA="dataset_builder/classification_dataset_100k/pneumonia_dataset_100k.csv"

# 2. Define the root directory to save images (absolute path)
TARGET_DIR="/mnt/hdd/jiazy/ehrxqa/physionet.org/files/mimic-cxr-jpg/2.0.0/files"

# 3. PhysioNet Base URL
MIMIC_CXR_JPG_DIR="https://physionet.org/files/mimic-cxr-jpg/2.0.0"
# ======================================================

# Check if CSV files exist
if [ ! -f "$CSV_LOS" ] && [ ! -f "$CSV_RR" ]; then
    echo "Error: CSV files not found in dataset_builder/regression_dataset_100k/"
    exit 1
fi

# Input credentials
echo "Enter your PhysioNet credentials"
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
echo

# Wget parameters
WGET_PARAMS="-c --user $USERNAME --password $PASSWORD"

# Download function
download() {
    local file_url=$1
    local local_path=$2
    
    # Create parent directory
    mkdir -p "$(dirname "$local_path")"

    # Use -O to force saving with the specified filename
    wget $WGET_PARAMS -O "$local_path" "$file_url"
    
    # Check if the download failed
    if [ $? -ne 0 ]; then
        echo "Error downloading $file_url"
        rm -f "$local_path"
    fi
}

# Function to extract 'image_path' column from regression CSVs
get_paths_from_csv() {
    local csv_file=$1
    if [ -f "$csv_file" ]; then
        # Use Python to read CSV and extract 'image_path' column, removing duplicates
        python -c "import pandas as pd; \
        df = pd.read_csv('$csv_file'); \
        print('\n'.join(df['image_path'].dropna().unique()))"
    else
        echo "Warning: $csv_file not found." >&2
    fi
}

echo "Generating file list from CSVs..."

# 1. Extract paths
paths_los=$(get_paths_from_csv "$CSV_LOS")
paths_rr=$(get_paths_from_csv "$CSV_RR")
paths_pneumonia=$(get_paths_from_csv "$CSV_PNEUMONIA")

# 2. Merge and deduplicate
all_paths=$(echo -e "$paths_los\n$paths_rr\n$paths_pneumonia")
readarray -t arr <<<"$(echo "$all_paths" | sort -u)"

echo "Total unique images required: ${#arr[@]}"
echo "Target directory: $TARGET_DIR"

# 3. Loop through, check, and download
count=0
total=${#arr[@]}

for image_path in "${arr[@]}"; do
    # Skip empty lines
    if [ -z "$image_path" ]; then continue; fi
    
    # Remove potential carriage return characters
    image_path=$(echo "$image_path" | tr -d '\r')
    
    # Construct local absolute path
    local_file="$TARGET_DIR/$image_path"
    
    # [Core Optimization] Check if the file exists locally; skip if it does
    if [ -f "$local_file" ]; then
        # Optional: Uncomment the line below to see which files are skipped
        # echo "Skipping (exists): $image_path"
        continue
    fi
    
    echo "Downloading [$((++count)) / $total]: $image_path"
    
    # Construct remote URL
    remote_url="$MIMIC_CXR_JPG_DIR/files/$image_path"
    
    # Execute download
    download "$remote_url" "$local_file"
done

echo "All missing images have been processed."