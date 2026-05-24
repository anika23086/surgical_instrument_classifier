import sys
from pathlib import Path
from PIL import Image

sys.path.append("/Users/anika/Desktop/surgical_instrument_classifier")
from search_engine import SurgicalInstrumentSearchEngine

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")

def main():
    print("==================================================")
    print("INITIALIZING MULTISCALE CLASSIFICATION ENGINE")
    print("==================================================")
    
    engine = SurgicalInstrumentSearchEngine(
        cache_path=str(PROJECT_DIR / "dataset/classifier_resnet50.pt"),
        metadata_path=str(PROJECT_DIR / "dataset/metadata.csv"),
        mapping_path=str(PROJECT_DIR / "dataset/class_mapping.json")
    )
    
    queries = [
        {"file": "true_query_1.png", "description": "Curved Artery Forceps"},
        {"file": "true_query_2.png", "description": "Cheatle Forceps"},
        {"file": "true_query_3.png", "description": "Straight Artery/Allis Forceps"}
    ]
    
    print("\n==================================================")
    print("RUNNING MULTISCALE CLASSIFICATION ON USER UPLOADS")
    print("==================================================")
    
    for q in queries:
        img_path = PROJECT_DIR / q["file"]
        if not img_path.exists():
            print(f"Error: Query image file {q['file']} not found!")
            continue
            
        print(f"\nQUERY: {q['file']} ({q['description']})")
        image = Image.open(img_path)
        
        # Get matches using multiscale logits model
        matches = engine.query_image(image, top_k=5)
        
        for rank, m in enumerate(matches):
            star = "-->" if rank == 0 else "   "
            print(f"{star} Rank {rank+1}: {m['name']} ({m['id']})")
            print(f"    SKU: {m['sku']} | Page: {m['page']} | Confidence: {m['similarity']*100:.2f}%")

if __name__ == "__main__":
    main()
