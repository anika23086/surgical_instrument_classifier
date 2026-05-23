from PIL import Image
import numpy as np
from pathlib import Path

downloads_dir = Path("/Users/anika/Downloads")
test_files = ["1.jpeg", "2.jpeg", "3.jpeg"]

for fname in test_files:
    fpath = downloads_dir / fname
    if not fpath.exists():
        continue
    img = Image.open(fpath).convert("L")
    img_np = np.array(img)
    h, w = img_np.shape
    
    # Let's inspect the corner pixels (background)
    corners = [
        img_np[0:10, 0:10],       # top-left
        img_np[0:10, w-10:w],     # top-right
        img_np[h-10:h, 0:10],     # bottom-left
        img_np[h-10:h, w-10:w]    # bottom-right
    ]
    
    print(f"\n======================================")
    print(f"IMAGE: {fname}")
    for idx, c in enumerate(["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]):
        corner_data = corners[idx]
        print(f"  {c} corner: min={corner_data.min()}, max={corner_data.max()}, mean={corner_data.mean():.1f}, median={np.median(corner_data)}")
