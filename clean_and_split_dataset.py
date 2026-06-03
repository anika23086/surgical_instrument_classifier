import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

from utils import crop_multiscale_regions

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
METADATA_PATH = PROJECT_DIR / "dataset/metadata.csv"
PROCESSED_DIR = PROJECT_DIR / "dataset/processed"




def clean_and_split():
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file missing at {METADATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(METADATA_PATH)
    print(f"Processing database of {len(df)} entries...")
    
    start_time = time.time()
    
    # Add new columns if they do not exist
    new_cols = ['inset_path', 'jaw_path', 'original_image_path']
    for c in new_cols:
        if c not in df.columns:
            df[c] = ""
            
    insets_extracted = 0
    
    for idx, row in df.iterrows():
        id_str = row['id']
        raw_img_path = PROCESSED_DIR / f"{id_str}.png"
        
        if not raw_img_path.exists():
            # Fall back to checking row['image_path'] if it points to a raw file
            raw_img_path = PROJECT_DIR / row['image_path']
            if not raw_img_path.exists() or "_full" in raw_img_path.name:
                # Try finding the original raw name
                raw_img_path = PROCESSED_DIR / f"{id_str.split('_full')[0]}.png"
                
        if not raw_img_path.exists():
            print(f"Warning: Raw catalog image missing for {id_str} at {raw_img_path}. Skipping.")
            continue
            
        try:
            # Load raw catalog image
            img = Image.open(raw_img_path)
            
            # Extract crops with category-aware logic
            is_rubber = id_str.startswith("rub_") or id_str.startswith("furn_")
            full_body, jaw, inset = crop_multiscale_regions(img, is_rubber=is_rubber)
            
            # Paths to save
            full_body_name = f"{id_str}_full.png"
            jaw_name = f"{id_str}_jaw.png"
            inset_name = f"{id_str}_inset.png"
            
            full_body_path = PROCESSED_DIR / full_body_name
            jaw_path = PROCESSED_DIR / jaw_name
            inset_path = PROCESSED_DIR / inset_name
            
            # Save crops to disk
            full_body.save(full_body_path)
            jaw.save(jaw_path)
            
            inset_str = ""
            if inset is not None:
                inset.save(inset_path)
                inset_str = f"dataset/processed/{inset_name}"
                insets_extracted += 1
                
            # Update dataframe fields
            df.at[idx, 'image_path'] = f"dataset/processed/{full_body_name}"
            df.at[idx, 'inset_path'] = inset_str
            df.at[idx, 'jaw_path'] = f"dataset/processed/{jaw_name}"
            df.at[idx, 'original_image_path'] = f"dataset/processed/{raw_img_path.name}"
            
        except Exception as e:
            print(f"Error processing {id_str}: {e}")
            
    # Save back updated metadata
    df.to_csv(METADATA_PATH, index=False)
    
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("DATABASE SPLITTING COMPLETE")
    print("="*50)
    print(f"Successfully processed: {len(df)} instrument records.")
    print(f"Zoomed-in detail insets extracted: {insets_extracted} views.")
    print(f"Total time elapsed: {elapsed:.2f} seconds.")
    print(f"Updated metadata saved to: {METADATA_PATH}")
    print("="*50)

if __name__ == "__main__":
    clean_and_split()
