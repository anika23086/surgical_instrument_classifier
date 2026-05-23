import sys
from pathlib import Path
sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import SurgicalInstrumentSearchEngine
from PIL import Image

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
engine = SurgicalInstrumentSearchEngine(
    cache_path=str(PROJECT_DIR / "dataset/classifier_resnet18.pt"),
    metadata_path=str(PROJECT_DIR / "dataset/metadata.csv"),
    mapping_path=str(PROJECT_DIR / "dataset/class_mapping.json")
)

# Test a few query images
test_queries = [
    "dataset/processed/p02_r1_c5.png", # Crile Artery Forceps (Curved)
    "dataset/processed/p02_r0_c3.png", # Bozeman Needle Holder
    "dataset/processed/p20_r0_c3.png", # Adson-beckman Retractor Folding
]

for q_path in test_queries:
    img_path = PROJECT_DIR / q_path
    if img_path.exists():
        print(f"\n==================================================")
        print(f"QUERY: {q_path}")
        image = Image.open(img_path)
        matches = engine.query_image(image, top_k=5)
        for idx, m in enumerate(matches):
            print(f"  Rank {idx+1}: {m['name']} ({m['id']}) | SKU: {m['sku']} | Page: {m['page']} | Similarity: {m['similarity']*100:.2f}%")
