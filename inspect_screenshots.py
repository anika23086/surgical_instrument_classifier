from pathlib import Path
from PIL import Image

desktop_dir = Path("/Users/anika/Desktop")

# Find screenshot files
screenshots = sorted(list(desktop_dir.glob("Screenshot 2026-05-21 at *.png")))
print(f"Found {len(screenshots)} screenshots:")
for s in screenshots:
    img = Image.open(s)
    print(f"  {s.name} | Size: {img.size}")
