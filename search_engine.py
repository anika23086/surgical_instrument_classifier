import numpy as np
import pandas as pd
from pathlib import Path
import json
from PIL import Image
import torch
from transformers import ResNetForImageClassification
from scipy import ndimage

PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")


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


def preprocess_crop_to_tensor(img_resized):
    """
    Direct fast numpy normalization (ImageNet standards) returning a FloatTensor.
    """
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img_np.transpose(2, 0, 1)).float()


class SurgicalInstrumentSearchEngine:
    def __init__(self, cache_path="dataset/classifier_resnet18.pt",
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

        # Initialize ResNet-18 backbone
        self.initialize_model()

    def load_metadata(self):
        if self.metadata_path.exists():
            self.metadata_df = pd.read_csv(self.metadata_path).fillna("")
            # Build high-performance lookup dictionary
            for _, row in self.metadata_df.iterrows():
                self.metadata_lookup[row['id']] = row.to_dict()
            print(f"Loaded metadata database containing {len(self.metadata_df)} items.")
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
        """Loads the pretrained ResNet-18 model and applies fine-tuned weights."""
        num_classes = len(self.metadata_df) if self.metadata_df is not None else 210
        if len(self.class_mapping) > 0:
            num_classes = len(self.class_mapping)

        print(f"Initializing ResNet-18 backbone on {self.device}...")
        self.model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-18",
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
        Classify a query image using averaged logits across multiscale crops and test-time augmentation.
        Returns top-K matches with softmax probability scores.
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

        # 2. Extract Full-Body Crop and Jaw-Focused Crop
        full_body, jaw, _ = crop_multiscale_regions(image)
        
        full_body = full_body.resize((224, 224), Image.Resampling.LANCZOS)
        jaw = jaw.resize((224, 224), Image.Resampling.LANCZOS)

        # 3. Test-Time Augmentation: generate multiple views for both crops
        full_body_views = self._generate_tta_views_for_crop(full_body)
        jaw_views = self._generate_tta_views_for_crop(jaw)
        
        batch_views = full_body_views + jaw_views  # 10 views total
        batch_tensor = torch.stack(batch_views).to(self.device)

        # 4. Predict logits and average
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            logits = outputs.logits  # shape (10, num_classes)
            avg_logits = logits.mean(dim=0)  # shape (num_classes,)
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
        cache_path=str(PROJECT_DIR / "dataset/classifier_resnet18.pt"),
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
                print(f"Rank {idx+1}: {match['name']} | Probability: {match['similarity']*100:.2f}% | SKU: {match['sku']}")
        else:
            print(f"Skipping: Test image missing at {test_path}")
