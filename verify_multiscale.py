import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")

def crop_multiscale_regions(img, padding=10, inset_threshold=400):
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

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(PROJECT_DIR / "dataset/metadata.csv")
    targets = [
        "p02_r0_c0",  # Tissue Forceps (Straight) - has inset
        "p02_r0_c1",  # Artery Forceps (Curved) - has inset
        "p02_r0_c2",  # Artery Forceps (Straight) - has inset
        "p05_r0_c2"   # Tenaculum Forceps - no inset
    ]
    
    scratch_dir = PROJECT_DIR / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    
    print("Testing multiscale cropping:")
    for tid in targets:
        row = df[df['id'] == tid].iloc[0]
        img_path = PROJECT_DIR / row['image_path']
        if not img_path.exists():
            print(f"Skipping {tid}: missing image file.")
            continue
            
        img = Image.open(img_path)
        full, jaw, inset = crop_multiscale_regions(img)
        
        # Save crops
        full.save(scratch_dir / f"{tid}_full.png")
        jaw.save(scratch_dir / f"{tid}_jaw.png")
        
        inset_status = "No inset"
        if inset is not None:
            inset.save(scratch_dir / f"{tid}_inset.png")
            inset_status = f"Inset saved ({inset.size})"
            
        print(f"ID: {tid} | Full Size: {full.size} | Jaw Size: {jaw.size} | Inset: {inset_status}")
