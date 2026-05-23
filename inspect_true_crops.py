import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import crop_main_instrument

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")

for i in range(1, 4):
    fpath = PROJECT_DIR / f"true_query_{i}.png"
    if not fpath.exists():
        continue
    img = Image.open(fpath)
    print(f"\n======================================")
    print(f"TRUE QUERY {i}: {fpath.name}")
    print(f"  Size: {img.size}")
    
    gray = img.convert("L")
    img_np = np.array(gray)
    foreground_mask = img_np < 240
    pct = (foreground_mask.sum() / img_np.size) * 100
    print(f"  Foreground (<240) percentage: {pct:.2f}%")
    
    # Check crop_main_instrument
    cropped = crop_main_instrument(img)
    print(f"  Cropped square canvas size: {cropped.size}")
    cropped.save(PROJECT_DIR / f"true_query_{i}_cropped.png")
