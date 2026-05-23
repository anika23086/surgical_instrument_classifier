import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import sys

sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import crop_main_instrument

df = pd.read_csv("dataset/metadata.csv")
targets = ["p05_r0_c2", "p02_r0_c1", "p02_r0_c2"]

print("Catalog images inspect:")
for tid in targets:
    row = df[df['id'] == tid].iloc[0]
    img_path = Path(row['image_path'])
    if not img_path.exists():
        print(f"Missing {img_path}")
        continue
    img = Image.open(img_path)
    print(f"\nID: {row['id']} | Name: {row['name']}")
    print(f"  Original size: {img.size}")
    
    # Check if there are any disconnected components
    img_np = np.array(img.convert("L"))
    h, w = img_np.shape
    foreground_mask = img_np < 240
    print(f"  Foreground percentage in catalog: {(foreground_mask.sum() / img_np.size)*100:.2f}%")
    
    # Dynamic crop check
    cropped = crop_main_instrument(img)
    print(f"  Cropped square canvas size: {cropped.size}")
