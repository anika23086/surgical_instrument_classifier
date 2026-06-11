import numpy as np
import pandas as pd
from pathlib import Path
import json
from PIL import Image
import torch
from transformers import ResNetForImageClassification

from utils import crop_multiscale_regions, preprocess_crop_to_tensor

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")


class SurgicalInstrumentSearchEngine:
    def __init__(self, cache_path="dataset/classifier_resnet50.pt",
                 metadata_path="dataset/metadata.csv",
                 mapping_path="dataset/class_mapping.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata_path = Path(metadata_path)
        self.cache_path = Path(cache_path)
        self.mapping_path = Path(mapping_path)

        self.metadata_df = None
        self.metadata_lookup = {}
        self.class_mapping = []
        self.model = None

        # Load metadata and mapping
        self.load_metadata()
        self.load_class_mapping()

        # Initialize ResNet-50 backbone
        self.initialize_model()

    def load_metadata(self):
        if self.metadata_path.exists():
            self.metadata_df = pd.read_csv(self.metadata_path).fillna("")
            # Build high-performance lookup dictionary deduplicated by unique ID (keep first occurrence)
            df_unique = self.metadata_df.drop_duplicates(subset=['id'], keep='first')
            for _, row in df_unique.iterrows():
                self.metadata_lookup[row['id']] = row.to_dict()
            print(f"Loaded metadata database containing {len(self.metadata_df)} items (lookup index: {len(self.metadata_lookup)} unique classes).")
        else:
            print("Warning: metadata.csv not found.")

    def load_class_mapping(self):
        if self.mapping_path.exists():
            with open(self.mapping_path, "r") as f:
                self.class_mapping = json.load(f)
            print(f"Loaded class mapping for {len(self.class_mapping)} instruments.")
        else:
            print("Warning: class_mapping.json not found.")

    def initialize_model(self):
        """Loads the pretrained ResNet-50 model and applies fine-tuned weights."""
        num_classes = len(self.metadata_df) if self.metadata_df is not None else 210
        if len(self.class_mapping) > 0:
            num_classes = len(self.class_mapping)

        print(f"Initializing ResNet-50 backbone on {self.device}...")
        self.model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        if self.cache_path.exists():
            print(f"Loading fine-tuned weights from {self.cache_path.name}...")
            state_dict = torch.load(self.cache_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Successfully loaded fine-tuned model state dictionary!")
        else:
            print("Warning: No fine-tuned weights found. Run train_classifier.py first.")

        self.model.to(self.device)
        self.model.eval()

    def reload(self):
        """
        Hot-reload all data and model weights after retraining.
        Called automatically by the pipeline after training completes.
        Zero downtime — the old model continues serving until reload is done.
        """
        print("Hot-reloading model and metadata...")
        self.load_metadata()
        self.load_class_mapping()
        self.initialize_model()
        print("Hot-reload complete! New model is now serving.")

    @staticmethod
    def preprocess_to_tensor(image):
        """
        Backward compatible preprocessing function. Crops to full-body, pads to square,
        resizes to 224x224, and normalizes using ImageNet stats.
        """
        img = image.convert("RGB")
        full_body, _, _ = crop_multiscale_regions(img)
        img_resized = full_body.resize((224, 224), Image.Resampling.LANCZOS)
        return preprocess_crop_to_tensor(img_resized)

    def query_image(self, query_img_path_or_pil, top_k=5):
        """
        Classify a query image using weighted multiscale TTA.

        Uses separate full-body and jaw crops with weighted logit fusion:
        - Full-body views contribute 60% (overall shape/silhouette)
        - Jaw views contribute 40% (fine-grained tip/jaw details)

        Returns top-K matches with softmax probability scores and confidence flags.
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        # 1. Load image
        if isinstance(query_img_path_or_pil, (str, Path)):
            img_path = Path(query_img_path_or_pil)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at: {query_img_path_or_pil}")
            image = Image.open(img_path).convert("RGB")
        else:
            image = query_img_path_or_pil.convert("RGB")

        # 2. Extract Full-Body and Jaw-Focused Crops
        full_body, jaw, _ = crop_multiscale_regions(image)

        full_body = full_body.resize((224, 224), Image.Resampling.LANCZOS)
        jaw = jaw.resize((224, 224), Image.Resampling.LANCZOS)

        # 3. Test-Time Augmentation: generate multiple views for both crops
        full_body_views = self._generate_tta_views_for_crop(full_body)
        jaw_views = self._generate_tta_views_for_crop(jaw)

        batch_views = full_body_views + jaw_views  # 10 views total
        batch_tensor = torch.stack(batch_views).to(self.device)

        # 4. Predict logits with weighted fusion
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            logits = outputs.logits  # shape (10, num_classes)

            # Weighted averaging: full-body (60%) captures overall shape,
            # jaw (40%) captures fine-grained distinguishing details
            full_body_logits = logits[:5].mean(dim=0)   # average 5 full-body views
            jaw_logits = logits[5:10].mean(dim=0)        # average 5 jaw views
            avg_logits = 0.6 * full_body_logits + 0.4 * jaw_logits

            probabilities = torch.softmax(avg_logits, dim=0)  # shape (num_classes,)

        # 5. Get top-K matches sorted by probability
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))

        top_matches = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            class_id = self.class_mapping[idx]
            record = self.metadata_lookup.get(class_id)
            if record:
                # Copy record to avoid mutating the cached lookup
                match_record = record.copy()
                match_record['similarity'] = round(prob, 4)
                # Flag low-confidence results so the UI can warn the user
                match_record['low_confidence'] = prob < 0.15
                top_matches.append(match_record)

        return top_matches

    def _generate_tta_views_for_crop(self, crop):
        """
        Generate Test-Time Augmentation views for a given crop PIL Image (224x224):
        1. Original
        2. Horizontal flip
        3. Three slight rotations (-15°, +15°, 180°)
        """
        views = []

        # 1. Original
        views.append(preprocess_crop_to_tensor(crop))

        # 2. Horizontal flip
        flipped = crop.transpose(Image.FLIP_LEFT_RIGHT)
        views.append(preprocess_crop_to_tensor(flipped))

        # 3. Rotation -15°
        rot_neg = crop.rotate(-15, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        views.append(preprocess_crop_to_tensor(rot_neg))

        # 4. Rotation +15°
        rot_pos = crop.rotate(15, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        views.append(preprocess_crop_to_tensor(rot_pos))

        # 5. 180° rotation (instrument upside down)
        rot_180 = crop.rotate(180, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        views.append(preprocess_crop_to_tensor(rot_180))

        return views


# Legacy compatibility aliases
preprocess_image_resnet = SurgicalInstrumentSearchEngine.preprocess_to_tensor
crop_main_instrument = lambda img, padding=10: crop_multiscale_regions(img, padding)[0]


if __name__ == "__main__":
    print("Initializing test engine...")
    engine = SurgicalInstrumentSearchEngine(
        cache_path=str(PROJECT_DIR / "dataset/classifier_resnet50.pt"),
        metadata_path=str(PROJECT_DIR / "dataset/metadata.csv"),
        mapping_path=str(PROJECT_DIR / "dataset/class_mapping.json")
    )

    if engine.metadata_df is not None and not engine.metadata_df.empty:
        test_row = engine.metadata_df.iloc[5]
        test_path = PROJECT_DIR / test_row['image_path']
        test_name = test_row['name']

        print(f"\nTesting similarity query for: '{test_name}'")
        if test_path.exists():
            matches = engine.query_image(test_path, top_k=3)
            for idx, match in enumerate(matches):
                confidence = "⚠️ LOW" if match.get('low_confidence') else "✓ OK"
                print(f"Rank {idx+1}: {match['name']} | Probability: {match['similarity']*100:.2f}% | Confidence: {confidence}")
        else:
            print(f"Skipping: Test image missing at {test_path}")
