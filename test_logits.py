import sys
from pathlib import Path
from PIL import Image
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

downloads_dir = Path("/Users/anika/Downloads")
test_files = ["1.jpeg", "2.jpeg", "3.jpeg"]

print("Classes mapped:")
print(f"Number of classes: {len(engine.class_mapping)}")

for fname in test_files:
    fpath = downloads_dir / fname
    if not fpath.exists():
        print(f"File {fname} not found in Downloads.")
        continue
    
    print(f"\n==============================================")
    print(f"CLASSIFYING: {fname}")
    try:
        img = Image.open(fpath).convert("RGB")
        print(f"Original size: {img.size}")
        
        # 1. Cosine similarity
        print("\n--- Cosine Similarity Method ---")
        matches = engine.query_image(img, top_k=5)
        for idx, m in enumerate(matches):
            print(f"  Rank {idx+1}: {m['name']} ({m['id']}) | SKU: {m['sku']} | Sim: {m['similarity']*100:.2f}%")
            
        # 2. Classifier Head Logits Method
        print("\n--- Classification Head (Logits) Method ---")
        # Let's generate TTA views and average their logits
        views = engine._generate_tta_views(img)
        batch = torch.stack(views).to(engine.device)
        
        with torch.no_grad():
            # In ResNetForImageClassification, forward pass returns ResNetClassifierOutput
            # containing logits of shape (N, num_labels)
            outputs = engine.model(batch)
            logits = outputs.logits # (N_tta, 210)
            avg_logits = logits.mean(dim=0) # (210,)
            probabilities = F.softmax(avg_logits, dim=-1)
            
        # Sort by probability
        top_probs, top_indices = torch.topk(probabilities, k=5)
        
        for idx, (prob, class_idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
            class_id = engine.class_mapping[class_idx]
            # Find in metadata
            record = engine.metadata_df[engine.metadata_df['id'] == class_id].iloc[0].to_dict()
            print(f"  Rank {idx+1}: {record['name']} ({record['id']}) | SKU: {record['sku']} | Prob: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"Error classifying {fname}: {e}")
        import traceback
        traceback.print_exc()
