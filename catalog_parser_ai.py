"""
AI-Powered Catalog Parser — Two-Stage Extraction

Stage 1: PyMuPDF extracts all embedded images pixel-perfectly with coordinates.
Stage 2: Vision LLM (Groq) labels each image with product name, SKU, category, etc.

This approach is more accurate than pure LLM bounding boxes because PyMuPDF
handles images losslessly, while the LLM only needs to read text and match it
to positions — a task it does well.
"""

import io
import os
import json
import time
import base64
import hashlib
import traceback
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, MAX_LOGO_DUPLICATES,
    RAW_DIR, MAX_PDF_PAGES
)


class CatalogParserAI:
    """
    Parses any medical equipment catalog PDF using a two-stage approach:
    1. PyMuPDF for pixel-perfect image extraction
    2. Vision LLM (Groq) for text labeling and image-text matching
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model_name = GROQ_MODEL
        self._client = None

    @property
    def client(self):
        """Lazy-initialize the Groq LLM client."""
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def parse_catalog(self, pdf_path: str, progress_callback=None) -> list[dict]:
        """
        Main entry point: parse a catalog PDF and return a list of extracted items.

        Each item has:
            - image_data: PIL Image object (the extracted product image)
            - raw_image_path: str (path where image was saved in dataset/raw/)
            - name: str (product name from LLM)
            - sku: str (SKU/product code)
            - category: str (product category)
            - description: str (brief description)
            - size: str (available sizes)
            - page: int (source page number)
            - confidence: str ('high' or 'low')
            - position: dict (x, y coordinates on page for matching)

        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callable(stage, message, progress_pct)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        total_pages = min(len(doc), MAX_PDF_PAGES)

        if progress_callback:
            progress_callback("extracting", f"Opened PDF with {total_pages} pages", 5)

        # ---------------------------------------------------------------
        # Stage 1: Extract all images from all pages with PyMuPDF
        # ---------------------------------------------------------------
        all_page_images = []  # list of (page_num, image_list)
        image_hash_counts = {}  # MD5 → count (for logo detection)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_images = self._extract_page_images(doc, page, page_idx)

            # Count image hashes for logo detection
            for img_info in page_images:
                h = img_info["hash"]
                image_hash_counts[h] = image_hash_counts.get(h, 0) + 1

            all_page_images.append((page_idx, page_images))

            if progress_callback:
                pct = 5 + int(20 * (page_idx + 1) / total_pages)
                progress_callback("extracting", f"Extracting images from page {page_idx + 1}/{total_pages}...", pct)

        # Count coordinate occurrences across pages
        all_flat_images = []
        for p_idx, page_images in all_page_images:
            for img_info in page_images:
                img_info["page_idx"] = p_idx
                all_flat_images.append(img_info)

        for img in all_flat_images:
            matching_pages = {img["page_idx"]}
            box_a = img["rect"]
            for other in all_flat_images:
                if other["page_idx"] in matching_pages:
                    continue
                box_b = other["rect"]
                # Check if box_a and box_b are nearly identical (within 4.0 points tolerance)
                if (abs(box_a[0] - box_b[0]) <= 4.0 and
                    abs(box_a[1] - box_b[1]) <= 4.0 and
                    abs(box_a[2] - box_b[2]) <= 4.0 and
                    abs(box_a[3] - box_b[3]) <= 4.0):
                    matching_pages.add(other["page_idx"])
            img["coordinate_page_count"] = len(matching_pages)

        # ---------------------------------------------------------------
        # Filter out logos (images that appear >N times across pages, by hash or position)
        # ---------------------------------------------------------------
        logo_hashes = {h for h, count in image_hash_counts.items() if count > MAX_LOGO_DUPLICATES}

        for p_idx, page_images in all_page_images:
            for img_info in page_images:
                is_logo_hash = img_info["hash"] in logo_hashes
                is_logo_coord = img_info.get("coordinate_page_count", 0) > MAX_LOGO_DUPLICATES
                img_info["is_logo"] = is_logo_hash or is_logo_coord

        # ---------------------------------------------------------------
        # Stage 2: Send each crop to the LLM for text labeling
        # ---------------------------------------------------------------
        all_items = []
        pages_with_images = [(p_idx, imgs) for p_idx, imgs in all_page_images
                             if any(not img["is_logo"] for img in imgs)]

        provider_label = "Groq"

        for i, (page_idx, page_images) in enumerate(pages_with_images):
            # Filter out logos
            valid_images = [img for img in page_images if not img["is_logo"]]
            if not valid_images:
                continue

            page = doc[page_idx]
            page_w = page.rect.width
            page_h = page.rect.height
            
            # Extract header context
            header_context = self._extract_page_header_text(page)
            
            # Render page at 200 DPI for cropping
            dpi = 200
            zoom = dpi / 72
            page_png_bytes = self._render_page_as_png(page, dpi=200)
            page_img = Image.open(io.BytesIO(page_png_bytes)).convert("RGB")

            for img_idx_in_page, img_info in enumerate(valid_images):
                if progress_callback:
                    # Calculate sub-page progress pct
                    base_pct = 25 + int(65 * i / max(len(pages_with_images), 1))
                    step_pct = int(65 * (img_idx_in_page / len(valid_images)) / max(len(pages_with_images), 1))
                    progress_callback("labeling", f"{provider_label} AI analyzing instrument {img_idx_in_page + 1}/{len(valid_images)} on page {page_idx + 1}...", min(90, base_pct + step_pct))

                # Crop contextual region (expanded around bbox to capture adjacent text)
                x0, y0, x1, y1 = img_info["rect"]
                cx0 = max(0, x0 - 60)
                cy0 = max(0, y0 - 30)
                cx1 = min(page_w, x1 + 180)
                cy1 = min(page_h, y1 + 120)
                
                left = int(cx0 * zoom)
                top = int(cy0 * zoom)
                right = int(cx1 * zoom)
                bottom = int(cy1 * zoom)
                
                w, h = page_img.size
                left = max(0, min(left, w - 1))
                top = max(0, min(top, h - 1))
                right = max(left + 1, min(right, w))
                bottom = max(top + 1, min(bottom, h))
                
                context_img = page_img.crop((left, top, right, bottom))
                buf = io.BytesIO()
                context_img.save(buf, format="PNG")
                context_png = buf.getvalue()

                # Label this crop
                label = self._groq_label_single_crop(context_png, page_idx + 1, header_context)

                # Save raw product crop
                raw_filename = self._save_raw_image(img_info["pil_image"], page_idx, img_info["img_index"])
                confidence = "high" if label.get("name") and label.get("sku") else "low"

                all_items.append({
                    "image_data": img_info["pil_image"],
                    "raw_image_path": str(raw_filename),
                    "name": label.get("name", ""),
                    "sku": label.get("sku", ""),
                    "category": label.get("category", ""),
                    "description": label.get("description", ""),
                    "size": label.get("size", ""),
                    "page": page_idx + 1,
                    "confidence": confidence,
                    "position": {
                        "x": img_info["rect"][0],
                        "y": img_info["rect"][1]
                    }
                })

                # Avoid hitting rate limits (300ms delay between crops)
                time.sleep(0.3)

        doc.close()

        if progress_callback:
            progress_callback("labeling", f"AI analysis complete — {len(all_items)} items extracted", 90)

        return all_items

    # ------------------------------------------------------------------
    # Stage 1 helpers: PyMuPDF image extraction
    # ------------------------------------------------------------------

    def _merge_bboxes(self, bboxes: list[tuple], threshold: float = 3.0) -> list[tuple]:
        """
        Merge bounding boxes that are close or overlapping.
        """
        merged = True
        boxes = list(bboxes)
        while merged:
            merged = False
            new_boxes = []
            used = set()
            
            for i in range(len(boxes)):
                if i in used:
                    continue
                
                box_a = list(boxes[i])
                ax0, ay0, ax1, ay1 = box_a
                ax0_e, ay0_e, ax1_e, ay1_e = ax0 - threshold, ay0 - threshold, ax1 + threshold, ay1 + threshold
                
                for j in range(i + 1, len(boxes)):
                    if j in used:
                        continue
                    bx0, by0, bx1, by1 = boxes[j]
                    
                    # Check overlap between expanded box_a and box_b
                    overlap_x = max(ax0_e, bx0) < min(ax1_e, bx1)
                    overlap_y = max(ay0_e, by0) < min(ay1_e, by1)
                    
                    if overlap_x and overlap_y:
                        box_a[0] = min(box_a[0], bx0)
                        box_a[1] = min(box_a[1], by0)
                        box_a[2] = max(box_a[2], bx1)
                        box_a[3] = max(box_a[3], by1)
                        used.add(j)
                        merged = True
                        ax0, ay0, ax1, ay1 = box_a
                        ax0_e, ay0_e, ax1_e, ay1_e = ax0 - threshold, ay0 - threshold, ax1 + threshold, ay1 + threshold
                
                new_boxes.append(tuple(box_a))
                used.add(i)
                
            boxes = new_boxes
        return boxes

    def _sort_bboxes_visually(self, bboxes: list[tuple]) -> list[tuple]:
        """
        Sort bounding boxes visually in reading order (top-to-bottom, left-to-right).
        Groups boxes into rows with a vertical tolerance, then sorts each row by X.
        """
        if not bboxes:
            return []
            
        # Sort by y0 first as a baseline
        sorted_by_y = sorted(bboxes, key=lambda b: b[1])
        
        rows = []
        for box in sorted_by_y:
            x0, y0, x1, y1 = box
            placed = False
            for row in rows:
                # Compare with the average y0/y1 of the row
                row_y0 = sum(r[1] for r in row) / len(row)
                row_y1 = sum(r[3] for r in row) / len(row)
                row_h = row_y1 - row_y0
                box_h = y1 - y0
                
                # Check vertical overlap
                overlap_y = max(y0, row_y0) < min(y1, row_y1)
                overlap_h = min(y1, row_y1) - max(y0, row_y0) if overlap_y else 0
                
                # If they overlap by more than 40% of either height, or if y0 is within 15 points
                if (overlap_y and (overlap_h > 0.4 * box_h or overlap_h > 0.4 * row_h)) or abs(y0 - row_y0) < 15:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])
                
        sorted_bboxes = []
        # Sort the rows by their average y0
        rows.sort(key=lambda r: sum(b[1] for b in r) / len(r))
        for row in rows:
            # Sort boxes in the row by x0
            row.sort(key=lambda b: b[0])
            sorted_bboxes.extend(row)
            
        return sorted_bboxes

    def _extract_page_images(self, doc, page, page_idx: int) -> list[dict]:
        """
        Extract visual images from a single page by rendering the page at 200 DPI
        and cropping the merged image bounding boxes.
        """
        # Render the page at 200 DPI
        dpi = 200
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        try:
            pix = page.get_pixmap(matrix=mat)
            page_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        except Exception as e:
            print(f"  Warning: Could not render page {page_idx + 1}: {e}")
            return []
            
        page_w = page.rect.width
        page_h = page.rect.height
        
        # Get all image bboxes
        bboxes = []
        try:
            for img_info in page.get_image_info(xrefs=True):
                bbox = img_info.get("bbox")
                if not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                width = x1 - x0
                height = y1 - y0
                
                # Filter out background/template images that cover a large portion of the page
                if width > 0.8 * page_w or height > 0.8 * page_h:
                    continue
                    
                # Filter tiny images (icons, decorations)
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue
                    
                bboxes.append(bbox)
        except Exception as e:
            print(f"  Warning: Could not read image info on page {page_idx + 1}: {e}")
            
        # Merge bboxes that are close/overlapping (threshold = 3.0 points)
        merged_bboxes = self._merge_bboxes(bboxes, threshold=3.0)
        
        # Define header/footer margins (8% of page height)
        margin_t = 0.08 * page_h
        margin_b = 0.92 * page_h
        
        # Filter out bboxes entirely within header or footer zones
        filtered_bboxes = []
        for bbox in merged_bboxes:
            x0, y0, x1, y1 = bbox
            if y1 < margin_t:
                continue  # entirely in header
            if y0 > margin_b:
                continue  # entirely in footer
            filtered_bboxes.append(bbox)
            
        # Sort bboxes visually in reading order (top-to-bottom, left-to-right)
        sorted_bboxes = self._sort_bboxes_visually(filtered_bboxes)
        
        images = []
        for img_index, bbox in enumerate(sorted_bboxes):
            try:
                x0, y0, x1, y1 = bbox
                
                # Clamp coordinates to page boundaries
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(page_w, x1)
                y1 = min(page_h, y1)
                
                if x1 <= x0 or y1 <= y0:
                    continue
                    
                # Scale coordinates to pixel values
                left = int(x0 * zoom)
                top = int(y0 * zoom)
                right = int(x1 * zoom)
                bottom = int(y1 * zoom)
                
                # Ensure crop box is within rendered page image boundaries
                w, h = page_img.size
                left = max(0, min(left, w - 1))
                top = max(0, min(top, h - 1))
                right = max(left + 1, min(right, w))
                bottom = max(top + 1, min(bottom, h))
                
                # Crop image
                pil_img = page_img.crop((left, top, right, bottom))
                
                # Compute hash for logo detection by saving to PNG bytes
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                img_hash = hashlib.md5(img_bytes).hexdigest()
                
                images.append({
                    "pil_image": pil_img,
                    "width": pil_img.width,
                    "height": pil_img.height,
                    "rect": bbox,
                    "hash": img_hash,
                    "img_index": img_index,
                    "is_logo": False
                })
            except Exception as e:
                print(f"  Warning: Could not crop merged box {bbox} on page {page_idx + 1}: {e}")
                continue
                
        return images

    def _get_image_rect(self, page, xref: int) -> tuple:
        """Get the bounding rectangle of an image on the page."""
        try:
            for img_info in page.get_image_info(xrefs=True):
                if img_info.get("xref") == xref:
                    bbox = img_info.get("bbox", (0, 0, 0, 0))
                    return tuple(bbox)
        except Exception:
            pass
        return (0, 0, 0, 0)

    def _render_page_as_png(self, page, dpi: int = 200) -> bytes:
        """Render a PDF page as a PNG image for the LLM to analyze."""
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")

    def _save_raw_image(self, pil_img: Image.Image, page_idx: int, img_index: int) -> Path:
        """Save an extracted image to dataset/raw/ with a unique filename."""
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename using timestamp to prevent collisions
        timestamp = int(time.time() * 1000)
        filename = f"ai_p{page_idx:02d}_i{img_index:02d}_{timestamp}.png"
        save_path = RAW_DIR / filename
        pil_img.save(save_path, "PNG")
        return Path(f"dataset/raw/{filename}")

    # ------------------------------------------------------------------
    # Stage 2 helpers: Vision LLM single crop labeling
    # ------------------------------------------------------------------

    def _extract_page_header_text(self, page) -> str:
        """Extract text from the top 15% of the page to serve as category/brand context."""
        try:
            page_h = page.rect.height
            header_rect = fitz.Rect(0, 0, page.rect.width, 0.15 * page_h)
            text = page.get_text("text", clip=header_rect)
            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return " | ".join(lines[:5])  # Limit to top 5 lines
        except Exception:
            return ""

    def _build_single_crop_prompt(self, page_num: int, header_context: str) -> str:
        """Build prompt for a single product contextual crop."""
        context_str = f"Page Header / Section Context: {header_context}\n" if header_context else ""
        
        return f"""You are analyzing a cropped region from page {page_num} of a medical/surgical equipment catalog.
{context_str}
This cropped image shows a single product and its associated description text.

Please extract the following information for this product by reading the text visible within this crop:

- product_name: The full product name (e.g., "Tissue Forceps (Straight)")
- sku: The SKU or product/catalog code number (e.g., "93131")
- category: The product category (e.g., "Surgical Forceps", "Surgical Scissors", "Hospital Furniture"). Use the Page Header / Section Context to guide you.
- description: A brief description if visible (e.g., material, features)
- size: Available sizes if listed (e.g., "6\" & 8\"")

IMPORTANT RULES:
1. If you cannot determine a field, use an empty string "".
2. Do NOT invent or hallucinate information — only extract what is actually visible in this crop.
3. SKU codes are usually numeric codes near the product image.
4. If there is no visible product label at all, return empty strings.

Return ONLY a valid JSON object with keys: product_name, sku, category, description, size
Do NOT include any markdown formatting, code fences, or explanatory text — just the raw JSON object."""

    def _parse_single_label(self, result_text: str) -> dict:
        """Parse a single JSON object response."""
        text = result_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            item = json.loads(text)
        except Exception:
            item = {}
            
        return {
            "name": str(item.get("product_name", "") or "").strip(),
            "sku": str(item.get("sku", "") or "").strip(),
            "category": str(item.get("category", "") or "").strip(),
            "description": str(item.get("description", "") or "").strip(),
            "size": str(item.get("size", "") or "").strip(),
        }

    def _groq_label_single_crop(self, context_png: bytes, page_num: int, header_context: str) -> dict:
        """
        Send a contextual product crop to Groq Vision and get its structured labels.
        Retries up to 3 times with exponential backoff on API errors.
        """
        prompt = self._build_single_crop_prompt(page_num, header_context)
        page_b64 = base64.b64encode(context_png).decode("utf-8")

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{page_b64}",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    temperature=0.1,
                    max_completion_tokens=1024,
                )

                result_text = response.choices[0].message.content.strip()
                return self._parse_single_label(result_text)

            except Exception as e:
                wait_time = 2 ** (attempt + 1)
                print(f"  Groq API crop labeling error on page {page_num} (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(wait_time)

        return {"name": "", "sku": "", "category": "", "description": "", "size": ""}
