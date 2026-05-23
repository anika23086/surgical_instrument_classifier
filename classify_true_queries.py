import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import SurgicalInstrumentSearchEngine

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
engine = SurgicalInstrumentSearchEngine(
    cache_path=str(PROJECT_DIR / "dataset/classifier_resnet18.pt"),
    metadata_path=str(PROJECT_DIR / "dataset/metadata.csv"),
    mapping_path=str(PROJECT_DIR / "dataset/class_mapping.json")
)

brain_dir = Path("/Users/anika/.gemini/antigravity/brain/b3c890b8-03c7-4ddd-a037-28c102bee4c2")

screenshots_data = [
    {
        "name": "media__1779383090446.png",
        "description": "Curved Artery Forceps",
        "box": (200, 135, 271, 299)
    },
    {
        "name": "media__1779383090454.png",
        "description": "Cheatle Forceps",
        "box": (184, 154, 277, 316)
    },
    {
        "name": "media__1779383090461.png",
        "description": "Straight Artery/Allis Forceps",
        "box": (195, 143, 277, 307)
    }
]

for idx, s in enumerate(screenshots_data):
    spath = brain_dir / s["name"]
    if not spath.exists():
        print(f"Screenshot {s['name']} not found.")
        continue
        
    print(f"\n==============================================")
    print(f"EVALUATING TRUE UPLOADED IMAGE FOR: {s['description']} ({s['name']})")
    
    # Extract the clean image uploaded in the scanner
    img = Image.open(spath).convert("RGB")
    xmin, ymin, xmax, ymax = s["box"]
    query_img = img.crop((xmin, ymin, xmax, ymax))
    
    # Save the true query image
    save_path = PROJECT_DIR / f"true_query_{idx+1}.png"
    query_img.save(save_path)
    print(f"  Cropped true query saved to: {save_path} (Size: {query_img.size})")
    
    # Multiscale Classifier Head Query
    print("\n  --- Multiscale Logits Classification ---")
    matches = engine.query_image(query_img, top_k=5)
    for rank, m in enumerate(matches):
        print(f"    Rank {rank+1}: {m['name']} ({m['id']}) | SKU: {m['sku']} | Prob: {m['similarity']*100:.2f}%")
