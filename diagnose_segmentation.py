import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage
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

def crop_with_dynamic_threshold(img, padding=10):
    img = img.convert("RGB")
    gray = img.convert("L")
    img_np = np.array(gray)
    
    # Let's find a threshold dynamically
    # Since the instrument is metallic and darker than the background,
    # let's look at the distribution.
    # If the background is bright and the instrument is dark:
    # A standard threshold is the Otsu threshold, or a percentile.
    # Let's try Otsu's thresholding.
    # Otsu thresholding finds the threshold that minimizes the intra-class variance.
    pixel_counts, bin_edges = np.histogram(img_np, bins=256, range=(0, 256))
    
    # Calculate cumulative sums and means
    weight1 = np.cumsum(pixel_counts)
    weight2 = np.cumsum(pixel_counts[::-1])[::-1]
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean1 = np.cumsum(pixel_counts * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((pixel_counts * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    
    variance_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    otsu_thresh = bin_centers[np.argmax(variance_between)]
    
    print(f"  Otsu threshold calculated: {otsu_thresh}")
    
    # Foreground mask: pixels darker than the threshold
    # But wait, in a dark background image, the instrument might be lighter than the background!
    # Let's assume standard white/light background.
    # If the background is light, foreground is darker.
    foreground_mask = img_np < otsu_thresh
    
    # If foreground_mask is empty or covers the entire image, fall back to simple threshold
    if not np.any(foreground_mask) or np.sum(foreground_mask) > 0.9 * img_np.size:
        foreground_mask = img_np < 240
        
    labeled_array, num_features = ndimage.label(foreground_mask)
    if num_features == 0:
        return img
        
    component_sizes = ndimage.sum(foreground_mask, labeled_array, range(1, num_features + 1))
    largest_component_label = np.argmax(component_sizes) + 1
    main_mask = (labeled_array == largest_component_label)
    
    rows = np.any(main_mask, axis=1)
    cols = np.any(main_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    w, h = img.size
    ymin = max(0, ymin - padding)
    ymax = min(h - 1, ymax + padding)
    xmin = max(0, xmin - padding)
    xmax = min(w - 1, xmax + padding)
    
    cropped = img.crop((xmin, ymin, xmax + 1, ymax + 1))
    
    # Pad to square
    cw, ch = cropped.size
    max_dim = max(cw, ch)
    square_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    offset_x = (max_dim - cw) // 2
    offset_y = (max_dim - ch) // 2
    square_img.paste(cropped, (offset_x, offset_y))
    return square_img

print("Testing dynamic threshold cropping:")
for fname in test_files:
    fpath = downloads_dir / fname
    if not fpath.exists():
        continue
    print(f"\n======================================")
    print(f"IMAGE: {fname}")
    img = Image.open(fpath)
    
    # Let's crop using dynamic threshold
    cropped = crop_with_dynamic_threshold(img)
    print(f"  Cropped size: {cropped.size}")
    
    # Let's predict with this crop!
    # Preprocess
    tensor = engine.preprocess_to_tensor(cropped).unsqueeze(0).to(engine.device)
    
    # Get logits prediction
    with torch.no_grad():
        outputs = engine.model(tensor)
        probabilities = F.softmax(outputs.logits.squeeze(0), dim=-1)
        
    top_probs, top_indices = torch.topk(probabilities, k=3)
    print("  Top predicted classes (Logits on dynamic crop):")
    for idx, (prob, class_idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
        class_id = engine.class_mapping[class_idx]
        record = engine.metadata_df[engine.metadata_df['id'] == class_id].iloc[0].to_dict()
        print(f"    Rank {idx+1}: {record['name']} ({record['id']}) | Prob: {prob*100:.2f}%")
