import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

try:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"Successfully loaded CLIP model on {device}!")
except Exception as e:
    print(f"Error loading CLIP: {e}")
