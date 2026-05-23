from PIL import Image
import numpy as np
from scipy import ndimage
from pathlib import Path

brain_dir = Path("/Users/anika/.gemini/antigravity/brain/b3c890b8-03c7-4ddd-a037-28c102bee4c2")
spath = brain_dir / "media__1779383090446.png"

img = Image.open(spath).convert("RGB")
img_np = np.array(img)
print(f"Screenshot size: {img.size}")

# Let's count white regions with different sizes
white_mask = (img_np[:, :, 0] > 240) & (img_np[:, :, 1] > 240) & (img_np[:, :, 2] > 240)
labeled_array, num_features = ndimage.label(white_mask)
print(f"Found {num_features} white regions total.")

slices = ndimage.find_objects(labeled_array)
for idx, slc in enumerate(slices):
    if slc is None:
        continue
    ymin, ymax = slc[0].start, slc[0].stop
    xmin, xmax = slc[1].start, slc[1].stop
    w = xmax - xmin
    h = ymax - ymin
    
    # Print all regions larger than 30x30
    if w > 30 and h > 30:
        # Check what the mean pixel inside is
        crop_np = img_np[ymin:ymax, xmin:xmax]
        print(f"Region {idx+1}: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}], size={w}x{h}, aspect={w/h:.2f}")
