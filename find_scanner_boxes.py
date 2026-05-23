from PIL import Image
import numpy as np
from scipy import ndimage
from pathlib import Path

brain_dir = Path("/Users/anika/.gemini/antigravity/brain/b3c890b8-03c7-4ddd-a037-28c102bee4c2")

for idx, sname in enumerate(["media__1779383090446.png", "media__1779383090454.png", "media__1779383090461.png"]):
    spath = brain_dir / sname
    if not spath.exists():
        continue
    img = Image.open(spath).convert("RGB")
    img_np = np.array(img)
    print(f"\n======================================")
    print(f"SCREENSHOT {idx+1}: {sname}")
    
    white_mask = (img_np[:, :, 0] > 240) & (img_np[:, :, 1] > 240) & (img_np[:, :, 2] > 240)
    labeled_array, num_features = ndimage.label(white_mask)
    
    slices = ndimage.find_objects(labeled_array)
    for slice_idx, slc in enumerate(slices):
        if slc is None:
            continue
        ymin, ymax = slc[0].start, slc[0].stop
        xmin, xmax = slc[1].start, slc[1].stop
        w = xmax - xmin
        h = ymax - ymin
        
        # We want to find the query box in the scanner (x around 150-300, y around 100-350)
        if 50 < w < 200 and 100 < h < 300:
            print(f"  Scanner Box Candidate: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}], size={w}x{h}, aspect={w/h:.2f}")
