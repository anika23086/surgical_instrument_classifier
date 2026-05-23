import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage
import pandas as pd

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
df = pd.read_csv(PROJECT_DIR / "dataset/metadata.csv")

row = df[df['id'] == "p02_r0_c2"].iloc[0]
img_path = PROJECT_DIR / row['image_path']
img = Image.open(img_path).convert("L")
img_np = np.array(img)
foreground_mask = img_np < 240

labeled_array, num_features = ndimage.label(foreground_mask)
print(f"Num features: {num_features}")

if num_features > 0:
    component_sizes = ndimage.sum(foreground_mask, labeled_array, range(1, num_features + 1))
    sorted_sizes = sorted(component_sizes, reverse=True)
    print("Sorted sizes:")
    for idx, s in enumerate(sorted_sizes):
        print(f"  Rank {idx+1}: size={s}")
