import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage

sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import crop_main_instrument

downloads_dir = Path("/Users/anika/Downloads")
test_files = ["1.jpeg", "2.jpeg", "3.jpeg"]

for fname in test_files:
    fpath = downloads_dir / fname
    if not fpath.exists():
        continue
        
    print(f"\n==============================================")
    print(f"ANALYZING CROP FOR: {fname}")
    img = Image.open(fpath)
    print(f"Original size: {img.size}")
    
    # Let's inspect the grayscale histogram to understand the background
    gray = img.convert("L")
    img_np = np.array(gray)
    
    # Calculate mask
    foreground_mask = img_np < 240
    pct_foreground = (np.sum(foreground_mask) / img_np.size) * 100
    print(f"Percentage of image treated as foreground (<240): {pct_foreground:.2f}%")
    
    # Run the crop
    cropped = crop_main_instrument(img)
    print(f"Cropped image size: {cropped.size}")
    
    # Save cropped image to downloads to inspect
    out_path = downloads_dir / f"crop_{fname}"
    cropped.save(out_path)
    print(f"Saved cropped image to {out_path}")
