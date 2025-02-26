#!/bin/bash
#SBATCH --job-name=download_nasa_data
#SBATCH --output=download_nasa_data_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=4G

# Load necessary modules if needed on your cluster
# module load ...

# Set the output directory
OUTPUT_DIR="/storage/home/cta5244/work/pyWBM_yield_data/hydro_models/NOAH/daily_soil100cm"
mkdir -p $OUTPUT_DIR

# URL list file
URL_FILE="/storage/home/cta5244/work/pyWBM_yield_data/hydro_models/subset_NLDAS_NOAH0125_H_002_20250226_033900_.txt"

# Check if username and password are provided as environment variables
if [ -z "$earthnasa_user" ] || [ -z "$earthnasa_pass" ]; then
    echo "Please set earthnasa_user and earthnasa_pass environment variables"
    echo "Example: export earthnasa_user=your_username"
    echo "         export earthnasa_pass=your_password"
    exit 1
fi

# Create cookies file
COOKIE_FILE="$HOME/.urs_cookies"

# Initialize counter
count=0
total=$(wc -l < "$URL_FILE")

# Loop through each URL in the file
while IFS= read -r url || [[ -n "$url" ]]; do
    # Skip empty lines
    if [ -z "$url" ]; then
        continue
    fi
    
    # Extract filename from URL
    filename=$(basename "$url")
    output_path="$OUTPUT_DIR/$filename"
    
    # Increment counter and display progress
    count=$((count + 1))
    echo "Downloading file $count of $total: $filename"
    
    # Download the file using wget
    wget --load-cookies "$COOKIE_FILE" \
         --save-cookies "$COOKIE_FILE" \
         --keep-session-cookies \
         --user="$earthnasa_user" \
         --password="$earthnasa_pass" \
         --output-document="$output_path" \
         "$url"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $filename"
    else
        echo "Failed to download: $filename" >&2
    fi
    