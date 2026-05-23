import sys
from pathlib import Path
from PIL import Image
import torch

sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import SurgicalInstrumentSearchEngine, crop_main_instrument

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
engine = SurgicalInstrumentSearchEngine(
    cache_path=str(PROJECT_DIR / "dataset/classifier_resnet18.pt"),
    metadata_path=str(PROJECT_DIR / "dataset/metadata.csv"),
    mapping_path=str(PROJECT_DIR / "dataset/class_mapping.json")
)

downloads_dir = Path("/Users/anika/Downloads")
test_files = ["1.jpeg", "2.jpeg", "3.jpeg"]

for fname in test_files:
    fpath = downloads_dir / fname
    if not fpath.exists():
        print(f"File {fname} not found in Downloads.")
        continue
    
    print(f"\n==============================================")
    print(f"CLASSIFYING: {fname}")
    try:
        img = Image.open(fpath)
        print(f"Original size: {img.size}")
        
        # Test connected-component crop size
        cropped = crop_main_instrument(img)
        print(f"Cropped size: {cropped.size}")
        
        # Let's see the classification results
        matches = engine.query_image(img, top_k=5)
        for idx, m in enumerate(matches):
            print(f"  Rank {idx+1}: {m['name']} ({m['id']}) | SKU: {m['sku']} | Similarity: {m['similarity']*100:.2f}%")
    except Exception as e:
        print(f"Error classifying {fname}: {e}")
