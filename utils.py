"""
Shared utility functions for surgical instrument image processing.

Contains image cropping, preprocessing, and augmentation functions used
across training, inference, and dataset building pipelines.
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from scipy import ndimage


def crop_multiscale_regions(img, padding=10, inset_threshold=400, is_rubber=False):
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

    if is_rubber:
        # Category-aware crop fallback: medical rubber products bypass jaw/inset extraction
        return full_body, full_body, None

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
    Expects a 224x224 RGB PIL Image.
    """
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img_np.transpose(2, 0, 1)).float()


def augment_image(img):
    """
    Applies aggressive augmentation on a 224x224 canvas covering:
    - Geometric: horizontal/vertical flips, 360° rotation, scale/translation
    - Photometric: brightness, contrast, sharpness, blur, grayscale, color temperature
    - Noise: Gaussian noise injection for domain gap bridging
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

    # 4. Random scale/translation (more aggressive: 0.55x - 1.0x)
    if random.random() > 0.05:
        scale_factor = random.uniform(0.55, 1.0)
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
        canvas = enhancer.enhance(random.uniform(0.65, 1.35))

    # 6. Contrast variation (wider range)
    if random.random() > 0.1:
        enhancer = ImageEnhance.Contrast(canvas)
        canvas = enhancer.enhance(random.uniform(0.65, 1.35))

    # 7. Sharpness variation
    if random.random() > 0.5:
        enhancer = ImageEnhance.Sharpness(canvas)
        canvas = enhancer.enhance(random.uniform(0.5, 2.0))

    # 8. Gaussian blur
    if random.random() > 0.8:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # 9. Random grayscale conversion (10%) — bridges domain gap with grayscale photos
    if random.random() > 0.9:
        canvas = canvas.convert("L").convert("RGB")

    # 10. Color temperature shift (15%) — simulates different lighting conditions
    if random.random() > 0.85:
        arr = np.array(canvas, dtype=np.float32)
        r_scale = random.uniform(0.9, 1.1)
        g_scale = random.uniform(0.95, 1.05)
        b_scale = random.uniform(0.9, 1.1)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * r_scale, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * g_scale, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * b_scale, 0, 255)
        canvas = Image.fromarray(arr.astype(np.uint8))

    # 11. Random noise injection (10%) — simulates camera noise
    if random.random() > 0.9:
        arr = np.array(canvas, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        canvas = Image.fromarray(arr.astype(np.uint8))

    return canvas
