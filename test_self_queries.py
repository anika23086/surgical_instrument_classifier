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

# Iterate over all rows in metadata
mismatches = 0
for idx, row in engine.metadata_df.iterrows():
    img_path = PROJECT_DIR / row['image_path']
    if not img_path.exists():
        continue
    image = Image.open(img_path)
    matches = engine.query_image(image, top_k=1)
    top_match = matches[0]
    if top_match['id'] != row['id']:
        print(f"Mismatch: Query '{row['name']}' ({row['id']}) -> Match 1: '{top_match['name']}' ({top_match['id']}) | Similarity: {top_match['similarity']*100:.2f}%")
        mismatches += 1

print(f"\nTotal self-query mismatches: {mismatches} / {len(engine.metadata_df)}")
