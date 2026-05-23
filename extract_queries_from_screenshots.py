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
screenshot_files = [
    "media__1779383090446.png",  # screenshot 1: Artery Forceps (Curved)
    "media__1779383090454.png",  # screenshot 2: Cheatle Forceps
    "media__1779383090461.png"   # screenshot 3: Allis Tissue Forceps (Straight) / straight forcep
]

def extract_white_query_box(screenshot_path):
    img = Image.open(screenshot_path).convert("RGB")
    img_np = np.array(img)
    
    # We want to find the query image inside the "Instrument Scanner" box.
    # The query image is a white rectangle containing the metallic instrument.
    # Let's find pixels that are extremely white: R > 250, G > 250, B > 250
    white_mask = (img_np[:, :, 0] > 250) & (img_np[:, :, 1] > 250) & (img_np[:, :, 2] > 250)
    
    # We want to find the bounding box of the query box.
    # Note that there are two white boxes in the screenshot: the QUERY image on the left, and the MATCH image on the right.
    # The QUERY image on the left is the uploaded image inside the scanner.
    # Let's find the connected components of the white mask.
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(white_mask)
    if num_features == 0:
        print("No white regions found in screenshot.")
        return img
        
    # Find bounding boxes of all components
    slices = ndimage.find_objects(labeled_array)
    
    candidate_crops = []
    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        ymin, ymax = slc[0].start, slc[0].stop
        xmin, xmax = slc[1].start, slc[1].stop
        w = xmax - xmin
        h = ymax - ymin
        
        # We are looking for the rectangular query crop inside the dark container on the left.
        # It typically has size around 200-500 pixels in width/height.
        # Let's filter out components that are too small or too thin
        if w > 100 and h > 100 and 0.5 < w/h < 2.0:
            candidate_crops.append((xmin, ymin, xmax, ymax, w*h))
            
    if not candidate_crops:
        print("No candidate query boxes found.")
        return img
        
    # Sort candidate crops by xmin.
    # The leftmost large white box is the query image inside the "Instrument Scanner" box!
    # The second one is the "QUERY" thumbnail under "Matching Analysis".
    # The third one is the "MATCH" thumbnail under "Matching Analysis".
    candidate_crops.sort(key=lambda x: x[0])
    
    xmin, ymin, xmax, ymax, _ = candidate_crops[0]
    print(f"Detected leftmost query box: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}], size={xmax-xmin}x{ymax-ymin}")
    
    cropped = img.crop((xmin, ymin, xmax, ymax))
    return cropped

for idx, sname in enumerate(screenshot_files):
    spath = brain_dir / sname
    if not spath.exists():
        print(f"Screenshot {sname} not found.")
        continue
        
    print(f"\n==============================================")
    print(f"SCREENSHOT: {sname} (Index {idx+1})")
    
    try:
        # Extract the clean query image uploaded by the user
        query_img = extract_white_query_box(spath)
        query_img_save_path = PROJECT_DIR / f"extracted_query_{idx+1}.png"
        query_img.save(query_img_save_path)
        print(f"Extracted query image saved to {query_img_save_path}")
        
        # Let's see what the current classification engine (cosine similarity) predicts
        print("\n--- Cosine Similarity Method ---")
        matches = engine.query_image(query_img, top_k=3)
        for rank, m in enumerate(matches):
            print(f"  Rank {rank+1}: {m['name']} ({m['id']}) | SKU: {m['sku']} | Sim: {m['similarity']*100:.2f}%")
            
        # Let's see what the classification head (logits) predicts
        print("\n--- Classification Head (Logits) Method ---")
        views = engine._generate_tta_views(query_img)
        batch = torch.stack(views).to(engine.device)
        with torch.no_grad():
            outputs = engine.model(batch)
            logits = outputs.logits
            avg_logits = logits.mean(dim=0)
            probs = F.softmax(avg_logits, dim=-1)
            
        top_probs, top_indices = torch.topk(probs, k=3)
        for rank, (prob, class_idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
            class_id = engine.class_mapping[class_idx]
            record = engine.metadata_df[engine.metadata_df['id'] == class_id].iloc[0].to_dict()
            print(f"  Rank {rank+1}: {record['name']} ({record['id']}) | SKU: {record['sku']} | Prob: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"Error processing screenshot {sname}: {e}")
        import traceback
        traceback.print_exc()
