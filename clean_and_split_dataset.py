import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
METADATA_PATH = PROJECT_DIR / "dataset/metadata.csv"
PROCESSED_DIR = PROJECT_DIR / "dataset/processed"

def crop_multiscale_regions(img, padding=10, inset_threshold=400):
    """
    Programmatically splits a catalog image into three visual scales:
    1. Full-Body Crop: Isolated main instrument body (largest connected component, no text/inset).
    2. Jaw Crop: The top 45% region of the full-body crop.
    3. Inset Crop: Zoomed-in visual detail view (second-largest connected component, if >= threshold).
    """
    img = img.convert("RGB")
    gray = img.convert("L")
    img_np = np.array(gray)
    foreground_mask = img_np < 240

    if not np.any(foreground_mask):
        w, h = img.size
        max_dim = max(w, h)
        square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return square, square, None

    labeled_array, num_features = ndimage.label(foreground_mask)
    if num_features == 0:
        w, h = img.size
        max_dim = max(w, h)
        square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return square, square, None

    component_sizes = ndimage.sum(foreground_mask, labeled_array, range(1, num_features + 1))
    sorted_indices = np.argsort(component_sizes)[::-1]
    
    # 1. Main Instrument (largest component)
    largest_label = sorted_indices[0] + 1
    largest_mask = (labeled_array == largest_label)
    
    rows = np.any(largest_mask, axis=1)
    cols = np.any(largest_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    w, h = img.size
    ymin = max(0, ymin - padding)
    ymax = min(h - 1, ymax + padding)
    xmin = max(0, xmin - padding)
    xmax = min(w - 1, xmax + padding)
    
    cropped_main = img.crop((xmin, ymin, xmax + 1, ymax + 1))
    cw, ch = cropped_main.size
    max_dim = max(cw, ch)
    full_body = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    full_body.paste(cropped_main, ((max_dim - cw) // 2, (max_dim - ch) // 2))
    
    # 2. Jaw Region (top 45% of cropped main height)
    jaw_height = int(ch * 0.45)
    if jaw_height < 10:
        jaw_height = ch
    jaw_cropped = cropped_main.crop((0, 0, cw, jaw_height))
    jcw, jch = jaw_cropped.size
    jmax_dim = max(jcw, jch)
    jaw = Image.new("RGB", (jmax_dim, jmax_dim), (255, 255, 255))
    jaw.paste(jaw_cropped, ((jmax_dim - jcw) // 2, (jmax_dim - jch) // 2))
    
    # 3. Inset Crop (second-largest component if area >= threshold)
    inset = None
    if len(sorted_indices) > 1:
        second_label = sorted_indices[1] + 1
        second_size = component_sizes[sorted_indices[1]]
        
        if second_size >= inset_threshold:
            second_mask = (labeled_array == second_label)
            srows = np.any(second_mask, axis=1)
            scols = np.any(second_mask, axis=0)
            symin, symax = np.where(srows)[0][[0, -1]]
            sxmin, sxmax = np.where(scols)[0][[0, -1]]
            
            symin = max(0, symin - padding)
            symax = min(h - 1, symax + padding)
            sxmin = max(0, sxmin - padding)
            sxmax = min(w - 1, sxmax + padding)
            
            cropped_inset = img.crop((sxmin, symin, sxmax + 1, symax + 1))
            icw, ich = cropped_inset.size
            imax_dim = max(icw, ich)
            inset = Image.new("RGB", (imax_dim, imax_dim), (255, 255, 255))
            inset.paste(cropped_inset, ((imax_dim - icw) // 2, (imax_dim - ich) // 2))
            
    return full_body, jaw, inset

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
            
            # Extract crops
            full_body, jaw, inset = crop_multiscale_regions(img)
            
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
