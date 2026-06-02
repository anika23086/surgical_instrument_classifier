import sys
import os
import math
import random
import time
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import ResNetForImageClassification

from utils import crop_multiscale_regions, preprocess_crop_to_tensor, augment_image

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
METADATA_PATH = PROJECT_DIR / "dataset/metadata.csv"
MODEL_SAVE_PATH = PROJECT_DIR / "dataset/classifier_resnet50.pt"

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin) loss head.

    Adds an angular margin penalty to the target class cosine similarity
    during training, forcing the model to learn more discriminative embeddings
    where similar-looking instruments are pushed further apart in feature space.

    This directly addresses the "confusion between similar instruments" problem
    by enforcing a minimum angular distance between class clusters.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep
    Face Recognition" (CVPR 2019)
    """

    def __init__(self, embedding_dim, num_classes, scale=30.0, margin=0.50):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # L2 normalize embeddings and class-center weights
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity between embeddings and class centers
        cosine = F.linear(emb_norm, w_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical safety: when cos(theta) < threshold, use linear fallback
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # Apply angular margin only to the target (correct) class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        return output * self.scale


def train_model():
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file not found at {METADATA_PATH}.", flush=True)
        sys.exit(1)

    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Consistent unique class index mapping
    classes = sorted(list(set(metadata_df['id'].tolist())))
    num_classes = len(classes)
    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(classes)}
    
    print(f"Target catalog size: {num_classes} unique instruments (total database rows: {len(metadata_df)}).", flush=True)

    # === Instantiate Pretrained ResNet-50 ===
    print("Loading pretrained Microsoft ResNet-50 backbone...", flush=True)
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    num_augmentations = 20  # 20 synthetic variations per base image
    total_database_rows = len(metadata_df)
    total_samples = total_database_rows * num_augmentations
    
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 1: Pre-extracting ResNet-50 Pooler embeddings", flush=True)
    print(f"  {total_database_rows} images × {num_augmentations} augmentations = {total_samples} samples", flush=True)
    print(f"  Using connected-component cropping + enhanced augmentation", flush=True)
    print(f"  (grayscale, color temperature, noise for domain gap bridging)", flush=True)
    print(f"{'='*60}", flush=True)

    # Pre-allocate feature tensor for memory efficiency
    # ResNet-50 pooler output: (2048, 1, 1)
    X_train = torch.empty(total_samples, 2048, 1, 1)
    y_train = torch.empty(total_samples, dtype=torch.long)
    sample_offset = 0

    t_start = time.time()

    with torch.no_grad():
        for idx, row in metadata_df.iterrows():
            cls_id = row['id']
            full_path = PROJECT_DIR / row['image_path']
            jaw_path_str = row['jaw_path']
            inset_path_str = row['inset_path'] if pd.notna(row['inset_path']) and str(row['inset_path']).strip() != "" else None

            label_idx = class_to_idx[cls_id]

            if not full_path.exists():
                print(f"Warning: Clean full image missing at {full_path}. Skipping.", flush=True)
                # Fill with zeros to maintain alignment
                X_train[sample_offset:sample_offset + num_augmentations] = 0
                y_train[sample_offset:sample_offset + num_augmentations] = label_idx
                sample_offset += num_augmentations
                continue

            try:
                # Load pre-saved clean full-body and jaw crops directly
                full_body = Image.open(full_path).convert("RGB")
                jaw = Image.open(PROJECT_DIR / jaw_path_str).convert("RGB")

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
                    crops.append(preprocess_crop_to_tensor(base_crop))
                    # 2. Horizontal flip of clean crop
                    if count > 1:
                        flipped = base_crop.transpose(Image.FLIP_LEFT_RIGHT)
                        crops.append(preprocess_crop_to_tensor(flipped))
                    # 3. Rest are random augmentations (with enhanced pipeline)
                    for _ in range(count - 2):
                        aug = augment_image(base_crop)
                        crops.append(preprocess_crop_to_tensor(aug))
                    return crops

                if inset is not None:
                    # 7 variations for full-body, 7 for jaw, 6 for inset (total 20)
                    batch_tensors.extend(generate_crops(full_body, 7))
                    batch_tensors.extend(generate_crops(jaw, 7))
                    batch_tensors.extend(generate_crops(inset, 6))
                else:
                    # 10 variations per scale (Full-Body, Jaw) (total 20)
                    batch_tensors.extend(generate_crops(full_body, 10))
                    batch_tensors.extend(generate_crops(jaw, 10))

                # Forward through the entire backbone to pre-extract pooler outputs
                batch_x = torch.stack(batch_tensors).to(device)
                h = model.resnet.embedder(batch_x)
                h = model.resnet.encoder.stages[0](h)
                h = model.resnet.encoder.stages[1](h)
                h = model.resnet.encoder.stages[2](h)
                h = model.resnet.encoder.stages[3](h)
                pooler_output = model.resnet.pooler(h)  # shape (20, 2048, 1, 1)

                X_train[sample_offset:sample_offset + num_augmentations] = pooler_output.cpu()
                y_train[sample_offset:sample_offset + num_augmentations] = label_idx
                sample_offset += num_augmentations

                if (idx + 1) % 30 == 0 or (idx + 1) == num_classes:
                    print(f"Extracted features for {idx + 1}/{num_classes} instruments...", flush=True)

            except Exception as e:
                print(f"Error extracting features for {cls_id}: {e}", flush=True)
                X_train[sample_offset:sample_offset + num_augmentations] = 0
                y_train[sample_offset:sample_offset + num_augmentations] = label_idx
                sample_offset += num_augmentations

    # Trim to actual samples
    X_train = X_train[:sample_offset]
    y_train = y_train[:sample_offset]

    extraction_time = time.time() - t_start
    print(f"\nFeature extraction completed in {extraction_time:.1f}s!", flush=True)
    print(f"Feature tensor shape: {X_train.shape} | Labels shape: {y_train.shape}", flush=True)

    # ===================================================================
    # STAGE 2: Fine-tune Classifier Head + ArcFace Head (Backbone Frozen)
    # ===================================================================
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 2: Fine-tuning Classifier + ArcFace (Backbone Frozen)", flush=True)
    print(f"  Loss: 0.5 × CrossEntropy(label_smoothing=0.1) + 0.5 × ArcFace(margin=0.5)", flush=True)
    print(f"  Scheduler: Cosine Annealing (eta_min=1e-6)", flush=True)
    print(f"{'='*60}", flush=True)

    # Freeze the entire backbone to prevent overfitting and speed up training
    for param in model.resnet.parameters():
        param.requires_grad = False

    # Unfreeze only the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.classifier.to(device)

    # Initialize ArcFace head (embedding_dim=2048 for ResNet-50 pooler output)
    arcface = ArcFaceHead(embedding_dim=2048, num_classes=num_classes, scale=30.0, margin=0.5)
    arcface.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # CrossEntropy with label smoothing prevents overconfident predictions
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": 1e-3},
        {"params": arcface.parameters(), "lr": 1e-3},
    ], weight_decay=1e-3)

    epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    t_train_start = time.time()

    for epoch in range(epochs):
        model.classifier.train()
        arcface.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            # === Branch 1: Standard classifier (CE loss) ===
            ce_logits = model.classifier(features)

            # === Branch 2: ArcFace head (angular margin loss) ===
            embeddings = features.flatten(1)  # (batch, 2048)
            arcface_logits = arcface(embeddings, labels)

            # Combined loss: CE for classification + ArcFace for embedding discrimination
            ce_loss = ce_criterion(ce_logits, labels)
            arc_loss = ce_criterion(arcface_logits, labels)
            loss = 0.5 * ce_loss + 0.5 * arc_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(ce_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 2 == 0 or (epoch + 1) == 1 or epoch_acc > 99.9:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {current_lr:.2e}", flush=True)
            if epoch_acc > 99.9:
                break

    total_train_time = time.time() - t_train_start
    print(f"Fine-tuning completed in {total_train_time:.1f}s!", flush=True)

    # Save model (ArcFace head is NOT saved — it's only needed during training
    # to shape the features; at inference the standard classifier head is used)
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
