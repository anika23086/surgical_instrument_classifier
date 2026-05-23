from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import crop_main_instrument

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")

for i in range(1, 4):
    fpath = PROJECT_DIR / f"extracted_query_{i}.png"
    if not fpath.exists():
        continue
    img = Image.open(fpath)
    print(f"\n======================================")
    print(f"EXTRACTED QUERY {i}: {fpath.name}")
    print(f"  Size: {img.size}")
    
    # Check corner pixels
    img_np = np.array(img.convert("L"))
    h, w = img_np.shape
    corners = [img_np[0:5, 0:5], img_np[0:5, w-5:w], img_np[h-5:h, 0:5], img_np[h-5:h, w-5:w]]
    print(f"  Background check: corners max={max([c.max() for c in corners])}, min={min([c.min() for c in corners])}")
    
    # Run crop_main_instrument
    cropped = crop_main_instrument(img)
    print(f"  Cropped size: {cropped.size}")
    
    # Save cropped image to inspect
    cropped.save(PROJECT_DIR / f"extracted_query_{i}_cropped.png")
