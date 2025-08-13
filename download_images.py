import os
import requests
import pandas as pd
from tqdm import tqdm

# Configuration
CSV_PATH = "recipe_final_new4.csv"  # From Kaggle
OUTPUT_DIR = "raw-data-images/raw-data-images"
KAGGLE_BASE_URL = "https://www.kaggle.com/api/v1"  # For direct download (alternative)

def setup_directories():
    """Create local directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_from_kaggle_urls(df):
    """Download images using original URLs from CSV"""
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        local_path = os.path.join(OUTPUT_DIR, f"{row['recipe_id']}.jpg")
        if not os.path.exists(local_path):
            try:
                # Try original URL if exists
                if pd.notna(row.get('original_url')):
                    response = requests.get(row['original_url'], timeout=10)
                    response.raise_for_status()
                # Fallback to constructing Kaggle path
                else:
                    response = requests.get(
                        f"{KAGGLE_BASE_URL}/datasets/download/{row['recipe_id']}",
                        stream=True,
                        timeout=10
                    )
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            except Exception as e:
                print(f"Failed to download {row['recipe_id']}: {str(e)}")

def main():
    setup_directories()
    df = pd.read_csv(CSV_PATH)
    download_from_kaggle_urls(df)
    print(f"\nDownload complete. Images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()