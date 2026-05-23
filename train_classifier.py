import sys
import os
import random
import time
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import ResNetForImageClassification
from scipy import ndimage

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
METADATA_PATH = PROJECT_DIR / "dataset/metadata.csv"
MODEL_SAVE_PATH = PROJECT_DIR / "dataset/classifier_resnet18.pt"

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def crop_multiscale_regions(img, padding=10, inset_threshold=400):
    """
    Programmatically splits a catalog image into three visual scales:
    1. Full-Body Crop: Isolated main instrument body (largest connected component, no text/inset).
    2. Jaw Crop: The top 45% region of the full-body crop.
    3. Inset Crop: Zoomed-in visual detail view (second-largest connected component, if >= threshold).
    """
    img = img.convert("RGB")
    gray = img.convert("L")
    img_np = np.array(gray)
    foreground_mask = img_np < 240

    if not np.any(foreground_mask):
        w, h = img.size
        max_dim = max(w, h)
        square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return square, square, None

    labeled_array, num_features = ndimage.label(foreground_mask)
    if num_features == 0:
        w, h = img.size
        max_dim = max(w, h)
        square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return square, square, None

    component_sizes = ndimage.sum(foreground_mask, labeled_array, range(1, num_features + 1))
    sorted_indices = np.argsort(component_sizes)[::-1]
    
    # 1. Main Instrument (largest component)
    largest_label = sorted_indices[0] + 1
    largest_mask = (labeled_array == largest_label)
    
    rows = np.any(largest_mask, axis=1)
    cols = np.any(largest_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    w, h = img.size
    ymin = max(0, ymin - padding)
    ymax = min(h - 1, ymax + padding)
    xmin = max(0, xmin - padding)
    xmax = min(w - 1, xmax + padding)
    
    cropped_main = img.crop((xmin, ymin, xmax + 1, ymax + 1))
    cw, ch = cropped_main.size
    max_dim = max(cw, ch)
    full_body = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    full_body.paste(cropped_main, ((max_dim - cw) // 2, (max_dim - ch) // 2))
    
    # 2. Jaw Region (top 45% of cropped main height)
    jaw_height = int(ch * 0.45)
    if jaw_height < 10:
        jaw_height = ch
    jaw_cropped = cropped_main.crop((0, 0, cw, jaw_height))
    jcw, jch = jaw_cropped.size
    jmax_dim = max(jcw, jch)
    jaw = Image.new("RGB", (jmax_dim, jmax_dim), (255, 255, 255))
    jaw.paste(jaw_cropped, ((jmax_dim - jcw) // 2, (jmax_dim - jch) // 2))
    
    # 3. Inset Crop (second-largest component if area >= threshold)
    inset = None
    if len(sorted_indices) > 1:
        second_label = sorted_indices[1] + 1
        second_size = component_sizes[sorted_indices[1]]
        
        if second_size >= inset_threshold:
            second_mask = (labeled_array == second_label)
            srows = np.any(second_mask, axis=1)
            scols = np.any(second_mask, axis=0)
            symin, symax = np.where(srows)[0][[0, -1]]
            sxmin, sxmax = np.where(scols)[0][[0, -1]]
            
            symin = max(0, symin - padding)
            symax = min(h - 1, symax + padding)
            sxmin = max(0, sxmin - padding)
            sxmax = min(w - 1, sxmax + padding)
            
            cropped_inset = img.crop((sxmin, symin, sxmax + 1, symax + 1))
            icw, ich = cropped_inset.size
            imax_dim = max(icw, ich)
            inset = Image.new("RGB", (imax_dim, imax_dim), (255, 255, 255))
            inset.paste(cropped_inset, ((imax_dim - icw) // 2, (imax_dim - ich) // 2))
            
    return full_body, jaw, inset





def augment_image(img):
    """
    Applies aggressive augmentation on 224x224 canvas covering:
    - Horizontal/vertical flips
    - 360° rotation
    - Scale translation (zoom 0.6x-1.0x)
    - Brightness, contrast, sharpness variations
    - Gaussian blur
    """
    canvas = img.copy()

    # 1. Random horizontal flip (50% chance)
    if random.random() > 0.5:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Random vertical flip (30% chance)
    if random.random() > 0.7:
        canvas = canvas.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Random 360-degree rotation
    if random.random() > 0.05:
        angle = random.uniform(0, 360)
        canvas = canvas.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    # 4. Random scale/translation (more aggressive range: 0.6x - 1.0x)
    if random.random() > 0.05:
        scale_factor = random.uniform(0.6, 1.0)
        new_dim = int(224 * scale_factor)
        resized = canvas.resize((new_dim, new_dim), Image.Resampling.LANCZOS)

        padded_canvas = Image.new("RGB", (224, 224), (255, 255, 255))
        offset_x = random.randint(0, 224 - new_dim)
        offset_y = random.randint(0, 224 - new_dim)
        padded_canvas.paste(resized, (offset_x, offset_y))
        canvas = padded_canvas

    # 5. Brightness variation (wider range)
    if random.random() > 0.1:
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(random.uniform(0.7, 1.3))

    # 6. Contrast variation (wider range)
    if random.random() > 0.1:
        enhancer = ImageEnhance.Contrast(canvas)
        canvas = enhancer.enhance(random.uniform(0.7, 1.3))

    # 7. Sharpness variation
    if random.random() > 0.5:
        enhancer = ImageEnhance.Sharpness(canvas)
        canvas = enhancer.enhance(random.uniform(0.5, 2.0))

    # 8. Gaussian blur
    if random.random() > 0.8:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return canvas


def preprocess_tensor(img):
    """Direct fast numpy normalization (ImageNet standards) returning a FloatTensor."""
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img_np.transpose(2, 0, 1)).float()


def train_model():
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file not found at {METADATA_PATH}.", flush=True)
        sys.exit(1)

    metadata_df = pd.read_csv(METADATA_PATH)
    num_classes = len(metadata_df)
    print(f"Target catalog size: {num_classes} instruments.", flush=True)

    # Consistent class index mapping
    classes = sorted(metadata_df['id'].tolist())
    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(classes)}

    # 1. Instantiate Pretrained ResNet-18
    print("Loading pretrained Microsoft ResNet-18 backbone...", flush=True)
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-18",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    num_augmentations = 30  # 30 synthetic variations per base image = 6,300 total samples
    print(f"\n==================================================", flush=True)
    print(f"STAGE 1: Pre-extracting ResNet-18 Stage-2 embeddings for {num_classes * num_augmentations} images...", flush=True)
    print(f"  Using connected-component cropping (instrument-only, no text/SKU)", flush=True)
    print(f"  Augmentation: flips + 360° rotation + 0.6x-1.0x scale + color jitter", flush=True)
    print(f"==================================================", flush=True)

    all_features = []
    all_labels = []

    t_start = time.time()

    with torch.no_grad():
        for idx, row in metadata_df.iterrows():
            cls_id = row['id']
            full_path = PROJECT_DIR / row['image_path']
            jaw_path = PROJECT_DIR / row['jaw_path']
            inset_path_str = row['inset_path'] if pd.notna(row['inset_path']) and str(row['inset_path']).strip() != "" else None
            
            label_idx = class_to_idx[cls_id]

            if not full_path.exists():
                print(f"Warning: Clean full image missing at {full_path}. Skipping.", flush=True)
                continue

            try:
                # Load pre-saved clean full-body and jaw crops directly
                full_body = Image.open(full_path).convert("RGB")
                jaw = Image.open(PROJECT_DIR / row['jaw_path']).convert("RGB")
                
                inset = None
                if inset_path_str is not None:
                    inset_path = PROJECT_DIR / inset_path_str
                    if inset_path.exists():
                        inset = Image.open(inset_path).convert("RGB")

                # Resize to target 224x224
                full_body = full_body.resize((224, 224), Image.Resampling.LANCZOS)
                jaw = jaw.resize((224, 224), Image.Resampling.LANCZOS)
                if inset is not None:
                    inset = inset.resize((224, 224), Image.Resampling.LANCZOS)

                batch_tensors = []

                # Helper to generate N augmented variations for a given crop
                def generate_crops(base_crop, count):
                    crops = []
                    # 1. Clean crop
                    crops.append(preprocess_tensor(base_crop))
                    # 2. Horizontal flip of clean crop
                    if count > 1:
                        flipped = base_crop.transpose(Image.FLIP_LEFT_RIGHT)
                        crops.append(preprocess_tensor(flipped))
                    # 3. Rest are random augmentations
                    for _ in range(count - 2):
                        aug = augment_image(base_crop)
                        crops.append(preprocess_tensor(aug))
                    return crops

                if inset is not None:
                    # 10 variations per scale (Full-Body, Jaw, Inset)
                    batch_tensors.extend(generate_crops(full_body, 10))
                    batch_tensors.extend(generate_crops(jaw, 10))
                    batch_tensors.extend(generate_crops(inset, 10))
                else:
                    # 15 variations per scale (Full-Body, Jaw)
                    batch_tensors.extend(generate_crops(full_body, 15))
                    batch_tensors.extend(generate_crops(jaw, 15))

                # Forward only through embedder -> stage 0 -> stage 1 -> stage 2
                batch_x = torch.stack(batch_tensors).to(device)
                h = model.resnet.embedder(batch_x)
                h = model.resnet.encoder.stages[0](h)
                h = model.resnet.encoder.stages[1](h)
                h = model.resnet.encoder.stages[2](h)  # shape (30, 256, 14, 14)

                all_features.append(h.cpu())
                all_labels.extend([label_idx] * num_augmentations)

                if (idx + 1) % 30 == 0 or (idx + 1) == num_classes:
                    print(f"Processed features for {idx + 1}/{num_classes} instruments...", flush=True)

            except Exception as e:
                print(f"Error extracting features for {cls_id}: {e}", flush=True)

    # Concatenate
    X_train = torch.cat(all_features, dim=0)
    y_train = torch.tensor(all_labels, dtype=torch.long)

    extraction_time = time.time() - t_start
    print(f"\nSuccessfully pre-extracted features in {extraction_time:.1f}s!", flush=True)
    print(f"Feature tensor shape: {X_train.shape} | Labels shape: {y_train.shape}", flush=True)

    # 3. Fine-tune Stage 3 & Classifier Head
    print(f"\n==================================================", flush=True)
    print(f"STAGE 2: Fine-tuning ResNet-18 Stage 3 + Classifier Head...", flush=True)
    print(f"==================================================", flush=True)

    # Freeze embedder and stages 0-2
    for param in model.resnet.embedder.parameters():
        param.requires_grad = False
    for i in range(3):
        for param in model.resnet.encoder.stages[i].parameters():
            param.requires_grad = False

    # Unfreeze stage 3 and classifier
    for param in model.resnet.encoder.stages[3].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.resnet.encoder.stages[3].to(device)
    model.classifier.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.resnet.encoder.stages[3].parameters(), "lr": 3e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3}
    ], weight_decay=1e-3)

    epochs = 25
    t_train_start = time.time()

    for epoch in range(epochs):
        model.resnet.encoder.stages[3].train()
        model.classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            out_stage3 = model.resnet.encoder.stages[3](images)
            pooler_output = model.resnet.pooler(out_stage3)
            outputs = model.classifier(pooler_output)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100

        if (epoch + 1) % 2 == 0 or (epoch + 1) == 1 or epoch_acc > 99.9:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%", flush=True)
            if epoch_acc > 99.9:
                break

    total_train_time = time.time() - t_train_start
    print(f"Fine-tuning completed in {total_train_time:.1f}s!", flush=True)

    # 4. Save model
    print(f"\nSaving fine-tuned model to {MODEL_SAVE_PATH.name}...", flush=True)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    import json
    class_mapping_path = PROJECT_DIR / "dataset/class_mapping.json"
    with open(class_mapping_path, "w") as f:
        json.dump(classes, f, indent=4)

    print(f"Class mapping saved to: {class_mapping_path.name}", flush=True)
    print("All tasks completed successfully!", flush=True)


if __name__ == "__main__":
    train_model()
