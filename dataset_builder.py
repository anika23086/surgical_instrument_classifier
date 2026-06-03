import fitz
import hashlib
import pandas as pd
from pathlib import Path
import re
import json

def clean_reconstructed_text(text):
    """
    Cleans up spacing and removes duplicate labels in cell text.
    """
    text = text.strip()
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def parse_surgical_catalog(pdf_path, out_images_dir, logo_hashes):
    """
    Parses the original surgical instruments catalog.
    Grid coordinates: 2 rows x 6 columns standard.
    Handles custom 8-column coordinates on page 17 & 18.
    """
    print("\n--- Parsing Surgical Instruments Catalog ---")
    doc = fitz.open(pdf_path)
    
    # Grid Coordinates Setup
    X_COLS = [0, 150, 280, 420, 570, 700, 850]
    Y_ROWS = [50, 300, 550]
    
    # Custom 8-Column Coordinates for Ophthalmic page 17 & 18
    X_COLS_P17_18 = [10, 105, 190, 280, 410, 520, 630, 710, 830]
    Y_ROWS_P17_18 = [50, 300, 560]
    
    metadata = []
    total_parsed = 0
    
    # Loop from page 2 to 21 (grid pages)
    for page_num in range(2, len(doc) - 1):
        page = doc[page_num]
        print(f"Processing Page {page_num}...")
        
        if page_num in [17, 18]:
            # Custom 8-Column Layout Parsing
            words = page.get_text("words")
            
            # Group words by (block_no, line_no)
            lines = {}
            for w in words:
                x0, y0, x1, y1, text, block_no, line_no, word_no = w
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                
                if cy < 50 or cy > 560 or cx < 10 or cx > 830:
                    continue
                if "Copyright" in text or "indosurgicals.com" in text or "supplier" in text or "medical" in text or text == "®":
                    continue
                    
                key = (block_no, line_no)
                if key not in lines:
                    lines[key] = []
                lines[key].append(w)
                
            grid_words = { (r, c): [] for r in range(2) for c in range(8) }
            for key, line_words in lines.items():
                line_words.sort(key=lambda w: w[7])
                x0 = min(w[0] for w in line_words)
                y0 = min(w[1] for w in line_words)
                x1 = max(w[2] for w in line_words)
                y1 = max(w[3] for w in line_words)
                
                lcx = (x0 + x1) / 2
                lcy = (y0 + y1) / 2
                
                cell_r, cell_c = -1, -1
                for r in range(2):
                    if Y_ROWS_P17_18[r] <= lcy < Y_ROWS_P17_18[r+1]:
                        cell_r = r
                        break
                for c in range(8):
                    if X_COLS_P17_18[c] <= lcx < X_COLS_P17_18[c+1]:
                        cell_c = c
                        break
                if cell_r != -1 and cell_c != -1:
                    grid_words[(cell_r, cell_c)].extend(line_words)
                    
            grid_images = { (r, c): [] for r in range(2) for c in range(8) }
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                rect = rects[0]
                
                base = doc.extract_image(xref)
                width, height = base["width"], base["height"]
                if width < 15 or height < 15:
                    continue
                    
                h = hashlib.md5(base["image"]).hexdigest()
                if h in logo_hashes:
                    continue
                    
                cx = (rect.x0 + rect.x1) / 2
                cy = (rect.y0 + rect.y1) / 2
                
                cell_r, cell_c = -1, -1
                for r in range(2):
                    if Y_ROWS_P17_18[r] <= cy < Y_ROWS_P17_18[r+1]:
                        cell_r = r
                        break
                for c in range(8):
                    if X_COLS_P17_18[c] <= cx < X_COLS_P17_18[c+1]:
                        cell_c = c
                        break
                if cell_r != -1 and cell_c != -1:
                    grid_images[(cell_r, cell_c)].append((rect, width, height))
                    
            left_category = "ENT & Ophthalmic Instruments" if page_num == 17 else "Ophthalmic Instruments"
            right_category = "Ophthalmic Instruments"
            
            for r in range(2):
                for c in range(8):
                    txt_words = grid_words[(r, c)]
                    imgs = grid_images[(r, c)]
                    
                    if txt_words and imgs:
                        txt_words.sort(key=lambda w: (w[5], w[6], w[7]))
                        raw_text = " ".join([w[4] for w in txt_words])
                        clean_text = clean_reconstructed_text(raw_text)
                        
                        sku_match = re.search(r'SKU:\s*([A-Za-z0-9\-/]+(?:\s+SKU:\s*[A-Za-z0-9\-/]+)*)', clean_text, re.IGNORECASE)
                        size_match = re.search(r'Size:\s*([^S]+)', clean_text, re.IGNORECASE)
                        
                        sku = sku_match.group(1).strip() if sku_match else "N/A"
                        size = size_match.group(1).strip() if size_match else "N/A"
                        
                        name = clean_text
                        name = re.sub(r'SKU:\s*[A-Za-z0-9\-/]+', '', name, flags=re.IGNORECASE)
                        name = re.sub(r'Size:\s*[^S]+', '', name, flags=re.IGNORECASE)
                        name = re.sub(r'\s+', ' ', name).strip()
                        
                        if not name or len(name) < 3:
                            name = clean_text
                            
                        category = left_category if c < 4 else right_category
                        
                        imgs.sort(key=lambda x: x[1] * x[2], reverse=True)
                        best_rect = imgs[0][0]
                        
                        padded_rect = fitz.Rect(
                            max(0, best_rect.x0 - 3),
                            max(0, best_rect.y0 - 3),
                            min(page.rect.x1, best_rect.x1 + 3),
                            min(page.rect.y1, best_rect.y1 + 3)
                        )
                        
                        img_id = f"p{page_num:02d}_r{r}_c{c}"
                        img_filename = f"{img_id}.png"
                        img_path = out_images_dir / img_filename
                        
                        pix = page.get_pixmap(clip=padded_rect, dpi=200)
                        pix.save(str(img_path))
                        
                        catalogs_arr = [{
                            "catalog": "Surgical Instruments Catalog",
                            "page": int(page_num),
                            "image_path": f"dataset/processed/{img_id}_full.png"
                        }]
                        
                        metadata.append({
                            "id": img_id,
                            "name": name,
                            "sku": sku,
                            "size": size,
                            "category": category,
                            "page": page_num,
                            "image_path": f"dataset/processed/{img_filename}",
                            "catalogs": json.dumps(catalogs_arr),
                            "description": "This surgical instrument is precision-crafted from high-grade medical steel, designed to meet rigorous clinical standards for hospital surgical workflows."
                        })
                        total_parsed += 1
                        
        else:
            # Standard 6-Column Layout Parsing
            left_header_words = []
            right_header_words = []
            
            words = page.get_text("words")
            for w in words:
                x0, y0, x1, y1, text, block_no, line_no, word_no = w
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                
                if cy < 50:
                    if "Copyright" in text or "indosurgicals" in text or text.isdigit() or "supplier" in text or "medical" in text:
                        continue
                    if cx < 420:
                        left_header_words.append(w)
                    else:
                        right_header_words.append(w)
                        
            left_header_words.sort(key=lambda w: (w[5], w[6], w[7]))
            right_header_words.sort(key=lambda w: (w[5], w[6], w[7]))
            left_category = " ".join([w[4] for w in left_header_words]).strip()
            right_category = " ".join([w[4] for w in right_header_words]).strip()
            
            if not left_category: left_category = "General Instruments"
            if not right_category: right_category = left_category
            
            grid_words = { (r, c): [] for r in range(2) for c in range(6) }
            for w in words:
                x0, y0, x1, y1, text, block_no, line_no, word_no = w
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                
                if cy < 50 or cy > 550 or cx < 10 or cx > 830:
                    continue
                if "Copyright" in text or "indosurgicals.com" in text or "supplier" in text or "medical" in text or text == "®":
                    continue
                    
                cell_r, cell_c = -1, -1
                for r in range(2):
                    if Y_ROWS[r] <= cy < Y_ROWS[r+1]:
                        cell_r = r
                        break
                for c in range(6):
                    if X_COLS[c] <= cx < X_COLS[c+1]:
                        cell_c = c
                        break
                if cell_r != -1 and cell_c != -1:
                    grid_words[(cell_r, cell_c)].append(w)
                    
            grid_images = { (r, c): [] for r in range(2) for c in range(6) }
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                rect = rects[0]
                
                base = doc.extract_image(xref)
                width, height = base["width"], base["height"]
                if width < 30 or height < 30:
                    continue
                    
                h = hashlib.md5(base["image"]).hexdigest()
                if h in logo_hashes:
                    continue
                    
                cx = (rect.x0 + rect.x1) / 2
                cy = (rect.y0 + rect.y1) / 2
                
                cell_r, cell_c = -1, -1
                for r in range(2):
                    if Y_ROWS[r] <= cy < Y_ROWS[r+1]:
                        cell_r = r
                        break
                for c in range(6):
                    if X_COLS[c] <= cx < X_COLS[c+1]:
                        cell_c = c
                        break
                if cell_r != -1 and cell_c != -1:
                    grid_images[(cell_r, cell_c)].append((rect, width, height))
                    
            for r in range(2):
                for c in range(6):
                    txt_words = grid_words[(r, c)]
                    imgs = grid_images[(r, c)]
                    
                    if txt_words and imgs:
                        txt_words.sort(key=lambda w: (w[5], w[6], w[7]))
                        raw_text = " ".join([w[4] for w in txt_words])
                        clean_text = clean_reconstructed_text(raw_text)
                        
                        sku_match = re.search(r'SKU:\s*([A-Za-z0-9\-/]+(?:\s+SKU:\s*[A-Za-z0-9\-/]+)*)', clean_text, re.IGNORECASE)
                        size_match = re.search(r'Size:\s*([^S]+)', clean_text, re.IGNORECASE)
                        
                        sku = sku_match.group(1).strip() if sku_match else "N/A"
                        size = size_match.group(1).strip() if size_match else "N/A"
                        
                        name = clean_text
                        name = re.sub(r'SKU:\s*[A-Za-z0-9\-/]+', '', name, flags=re.IGNORECASE)
                        name = re.sub(r'Size:\s*[^S]+', '', name, flags=re.IGNORECASE)
                        name = re.sub(r'\s+', ' ', name).strip()
                        
                        if not name or len(name) < 3:
                            name = clean_text
                            
                        category = left_category if c < 3 else right_category
                        
                        imgs.sort(key=lambda x: x[1] * x[2], reverse=True)
                        best_rect = imgs[0][0]
                        
                        padded_rect = fitz.Rect(
                            max(0, best_rect.x0 - 3),
                            max(0, best_rect.y0 - 3),
                            min(page.rect.x1, best_rect.x1 + 3),
                            min(page.rect.y1, best_rect.y1 + 3)
                        )
                        
                        img_id = f"p{page_num:02d}_r{r}_c{c}"
                        img_filename = f"{img_id}.png"
                        img_path = out_images_dir / img_filename
                        
                        pix = page.get_pixmap(clip=padded_rect, dpi=200)
                        pix.save(str(img_path))
                        
                        catalogs_arr = [{
                            "catalog": "Surgical Instruments Catalog",
                            "page": int(page_num),
                            "image_path": f"dataset/processed/{img_id}_full.png"
                        }]
                        
                        metadata.append({
                            "id": img_id,
                            "name": name,
                            "sku": sku,
                            "size": size,
                            "category": category,
                            "page": page_num,
                            "image_path": f"dataset/processed/{img_filename}",
                            "catalogs": json.dumps(catalogs_arr),
                            "description": "This surgical instrument is precision-crafted from high-grade medical steel, designed to meet rigorous clinical standards for hospital surgical workflows."
                        })
                        total_parsed += 1
                        
    doc.close()
    print(f"Surgical catalog extraction completed. Mapped: {total_parsed} instruments.")
    return metadata

def parse_ophthalmic_catalog(pdf_path, out_images_dir, logo_hashes, surgical_metadata):
    """
    Parses the ophthalmic instruments catalog.
    Grid coordinates: 3 rows x 4 columns on double landscape page spreads.
    Unifies overlapping items by matching SKUs.
    """
    print("\n--- Parsing Ophthalmic Instruments Catalog ---")
    doc = fitz.open(pdf_path)
    
    # 3 Rows x 4 Columns landscape coordinates
    X_COLS = [0, 310, 595, 890, 1190]
    Y_ROWS = [50, 330, 580, 800]
    
    # Build a lookup table from surgical catalog to find overlapping SKUs
    # Match standard alphanumeric SKUs (e.g. "98100")
    sku_to_surgical_item = {}
    for item in surgical_metadata:
        s_clean = re.sub(r'[^a-zA-Z0-9]', '', str(item["sku"])).lower().strip()
        if s_clean and s_clean != "na":
            # Map clean SKU to the primary ID and list of catalogs
            sku_to_surgical_item[s_clean] = item
            
    metadata = []
    total_parsed = 0
    total_unified = 0
    
    # Catalog pages are 1 to 4
    for page_num in range(1, len(doc) - 1):
        page = doc[page_num]
        print(f"Processing Page {page_num}...")
        
        words = page.get_text("words")
        
        # Grid Words
        grid_words = { (r, c): [] for r in range(3) for c in range(4) }
        for w in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = w
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            
            if cy < 50 or cy > 800 or cx < 10 or cx > 1180:
                continue
            if "Copyright" in text or "indosurgicals" in text or "WELCOME" in text or "CERTIFICATE" in text:
                continue
                
            cell_r, cell_c = -1, -1
            for r in range(3):
                if Y_ROWS[r] <= cy < Y_ROWS[r+1]:
                    cell_r = r
                    break
            for c in range(4):
                if X_COLS[c] <= cx < X_COLS[c+1]:
                    cell_c = c
                    break
            if cell_r != -1 and cell_c != -1:
                grid_words[(cell_r, cell_c)].append(w)
                
        # Grid Images
        grid_images = { (r, c): [] for r in range(3) for c in range(4) }
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            
            base = doc.extract_image(xref)
            width, height = base["width"], base["height"]
            if width < 15 or height < 15:
                continue
                
            h = hashlib.md5(base["image"]).hexdigest()
            if h in logo_hashes:
                continue
                
            cx = (rect.x0 + rect.x1) / 2
            cy = (rect.y0 + rect.y1) / 2
            
            # Skip left half images on page 1 (non-product welcome/certificate content)
            if page_num == 1 and cx < 595:
                continue
                
            cell_r, cell_c = -1, -1
            for r in range(3):
                if Y_ROWS[r] <= cy < Y_ROWS[r+1]:
                    cell_r = r
                    break
            for c in range(4):
                if X_COLS[c] <= cx < X_COLS[c+1]:
                    cell_c = c
                    break
            if cell_r != -1 and cell_c != -1:
                grid_images[(cell_r, cell_c)].append((rect, width, height))
                
        # Pair grid content
        for r in range(3):
            for c in range(4):
                if page_num == 1 and c < 2:
                    continue # Skip welcome and certificates
                    
                txt_words = grid_words[(r, c)]
                imgs = grid_images[(r, c)]
                
                if txt_words and imgs:
                    txt_words.sort(key=lambda w: (w[5], w[6], w[7]))
                    raw_text = " ".join([w[4] for w in txt_words])
                    clean_text = clean_reconstructed_text(raw_text)
                    
                    sku_match = re.search(r'SKU\s*:\s*([A-Za-z0-9\-/]+)', clean_text, re.IGNORECASE)
                    sku = sku_match.group(1).strip() if sku_match else "N/A"
                    
                    name = clean_text
                    name = re.sub(r'SKU\s*:\s*[A-Za-z0-9\-/]+', '', name, flags=re.IGNORECASE)
                    name = re.sub(r'\s+', ' ', name).strip()
                    
                    size = "N/A"
                    size_match = re.search(r'\(([^)]+)\)', name)
                    if size_match:
                        size = size_match.group(1).strip()
                        
                    category = "Ophthalmic Instruments"
                    
                    imgs.sort(key=lambda x: x[1] * x[2], reverse=True)
                    best_rect = imgs[0][0]
                    
                    padded_rect = fitz.Rect(
                        max(0, best_rect.x0 - 3),
                        max(0, best_rect.y0 - 3),
                        min(page.rect.x1, best_rect.x1 + 3),
                        min(page.rect.y1, best_rect.y1 + 3)
                    )
                    
                    # Unification logic
                    clean_sku = re.sub(r'[^a-zA-Z0-9]', '', str(sku)).lower().strip()
                    is_duplicate = clean_sku in sku_to_surgical_item
                    
                    if is_duplicate:
                        # Shared Class ID matching original
                        primary_item = sku_to_surgical_item[clean_sku]
                        class_id = primary_item["id"]
                        
                        # Generate unique file suffix to avoid naming conflicts on disk
                        img_filename = f"op_{class_id}.png"
                        
                        # Parse original catalogs array and append current Ophthalmic occurrence
                        try:
                            orig_cats = json.loads(primary_item["catalogs"])
                        except Exception:
                            orig_cats = [{
                                "catalog": "Surgical Instruments Catalog",
                                "page": int(primary_item["page"]),
                                "image_path": f"dataset/processed/{class_id}_full.png"
                            }]
                            
                        # Add new occurrence
                        new_occurrence = {
                            "catalog": "Ophthalmic Instruments Catalog",
                            "page": int(page_num),
                            "image_path": f"dataset/processed/op_{class_id}_full.png"
                        }
                        
                        # Merge if not already added
                        if not any(cat["catalog"] == "Ophthalmic Instruments Catalog" for cat in orig_cats):
                            orig_cats.append(new_occurrence)
                            
                        updated_catalogs_str = json.dumps(orig_cats)
                        
                        # Update the existing items in both lists so they all share the complete aggregated list!
                        primary_item["catalogs"] = updated_catalogs_str
                        
                        for item in surgical_metadata:
                            if item["id"] == class_id:
                                item["catalogs"] = updated_catalogs_str
                        for item in metadata:
                            if item["id"] == class_id:
                                item["catalogs"] = updated_catalogs_str
                                
                        total_unified += 1
                        print(f"  Unified duplicated SKU {sku}: {name} -> Class ID: {class_id}")
                    else:
                        # Brand new item
                        class_id = f"op_p{page_num:02d}_r{r}_c{c}"
                        img_filename = f"{class_id}.png"
                        
                        updated_catalogs_str = json.dumps([{
                            "catalog": "Ophthalmic Instruments Catalog",
                            "page": int(page_num),
                            "image_path": f"dataset/processed/{class_id}_full.png"
                        }])
                    
                    img_path = out_images_dir / img_filename
                    pix = page.get_pixmap(clip=padded_rect, dpi=200)
                    pix.save(str(img_path))
                    
                    metadata.append({
                        "id": class_id,
                        "name": name,
                        "sku": sku,
                        "size": size,
                        "category": category,
                        "page": page_num,
                        "image_path": f"dataset/processed/{img_filename}",
                        "catalogs": updated_catalogs_str,
                        "description": "This surgical instrument is precision-crafted from high-grade medical steel, designed to meet rigorous clinical standards for hospital surgical workflows."
                    })
                    total_parsed += 1
                    
    doc.close()
    print(f"Ophthalmic catalog extraction completed. Mapped: {total_parsed} items ({total_unified} unified).")
    return metadata

def parse_medical_rubber_catalog(pdf_path, out_images_dir, logo_hashes):
    """
    Parses the single page medical rubber products catalog.
    Uses precise 2D Euclidean spatial closest-bold-title pairing and hand-crafted mappers
    for 100% correct, bolded names, clean SKUs, and premium descriptions.
    """
    print("\n--- Parsing Medical Rubber Products Catalog ---")
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    # Hand-crafted high-fidelity metadata mapper for the 19 rubber products
    RUBBER_METADATA_MAP = {
        "Air Cushion (Invalid Air Rings)": {
            "name": "Air Cushion (Invalid Air Rings)",
            "sku": "35000 to 35003 & 35043 (General), 35004 to 35006 (Deluxe)",
            "size": "General: 30cm, 35cm, 40cm, 42.5cm & 45cm; Deluxe: 35cm, 40cm & 45cm",
            "description": "IndoSurgicals Air Cushions (Invalid Air Rings) are designed for patients who require relief from pressure when sitting for extended periods. Made from high-quality, durable soft rubber."
        },
        "Breast Pump": {
            "name": "Breast Pump",
            "sku": "35007 (Complete), 35008 (55ml Bulb), 35009 (25ml Bulb)",
            "size": "Standard Size",
            "description": "Designed for comfortable and efficient breast milk expression. Complete with glass barrel and premium rubber suction bulb."
        },
        "Enema Syringe": {
            "name": "Enema Syringe",
            "sku": "35010",
            "size": "Standard Size",
            "description": "Enema Syringe Higginson, in Red Colour with Rectal Nozzle & Valve. High-quality rubber bulb and tubing for gentle and effective deep cleansing."
        },
        "Pipette Bulb": {
            "name": "Pipette Bulb",
            "sku": "35031 to 35033",
            "size": "Small (10ml), Medium (15ml), Large (20ml)",
            "description": "Premium rubber pipette bulbs designed for laboratory and clinical applications. Highly responsive suction and durable construction."
        },
        "Drainage Sheet Corrugated": {
            "name": "Drainage Sheet Corrugated",
            "sku": "35011",
            "size": "150mm x 300mm",
            "description": "Corrugated latex drainage sheet used to facilitate surgical drainage. Made from soft, flexible, medical-grade rubber."
        },
        "Chloroform Bellow": {
            "name": "Chloroform Bellow",
            "sku": "35012 & 35013",
            "size": "500ml & 1000ml",
            "description": "Double-ended chloroform bellow bulb designed for clinical anesthesia workflows. Durable, natural red rubber construction."
        },
        "Tourniquet (Latex)": {
            "name": "Tourniquet (Latex)",
            "sku": "35034",
            "size": "Length: 75-100cm approx.",
            "description": "Tourniquet Latex Elastic Strip, tubular reusable construction. Natural latex sheeting with high elasticity, non-sterile."
        },
        "Tourniquet": {
            "name": "Tourniquet",
            "sku": "35026 & 35027",
            "size": "Standard Size",
            "description": "Reusable elastic band tourniquet with secure plastic buckle closure for quick and easy blood draw occlusion. Available in Blue & Red."
        },
        "Mackintosh Sheet": {
            "name": "Mackintosh Sheet",
            "sku": "35028 (Deluxe), 35029 (Regular)",
            "size": "Width: 90cm, Roll Length: 10 meter",
            "description": "IndoSurgicals Mackintosh Sheeting (Hospital Sheeting) is made from the best quality Indian soft rubber. The sheeting has high breaking and tearing strength. Washable with cold water and mild soap. Colors: Red/Green, Red/Blue."
        },
        "Douche Bag": {
            "name": "Douche Bag",
            "sku": "35030, 35040 & 35041",
            "size": "1 Litre, 2 Litre & 4 Litre",
            "description": "High Quality - Superior Rubber Latex Bag with Superior Tubing and Smooth Nozzles. Fountain top for easy cleaning & hygiene - Non-leeching & no disturbing odors. Suited for High (Deep) Enemas; Supplied with complete accessories (hose, nozzles, clamp and a flexible colon tip) and instruction manual in English."
        },
        "Stomach Pump Tube": {
            "name": "Stomach Pump Tube",
            "sku": "35024",
            "size": "150 cm",
            "description": "Premium medical-grade stomach pump tube, 150 cm length with suction bulb, funnel and plastic mouth gag."
        },
        "Eye/Ear & Ulcer Syringe": {
            "name": "Eye/Ear & Ulcer Syringe",
            "sku": "35014 to 35017",
            "size": "25ml, 50ml, 75ml & 100ml",
            "description": "Premium soft red rubber suction bulb designed for gentle fluid aspiration and irrigation of the eye, ear, or surgical ulcer sites."
        },
        "Infant / Rectal Syringe": {
            "name": "Infant / Rectal Syringe",
            "sku": "35018 to 35020",
            "size": "25ml, 64ml & 114ml",
            "description": "Designed for gentle infant nasal aspiration or rectal irrigation. Features a flexible, soft rubber suction bulb."
        },
        "Vaginal Douche Spray": {
            "name": "Vaginal Douche Spray",
            "sku": "35021",
            "size": "Standard Size",
            "description": "Vaginal Douche Spray, complete with premium plastic nozzle and flexible red rubber bulb reservoir."
        },
        "Hot Water Bottles": {
            "name": "Hot Water Bottles",
            "sku": "35036 to 35038",
            "size": "Standard Size",
            "description": "Made from natural rubber, the flexible bottle conforms to your body. Two-side ribbed design. There's nothing like an old fashioned hot water bottle to soothe aches and pains or keep you warm at night."
        },
        "Kelly's Pad with Pump": {
            "name": "Kelly's Pad with Pump",
            "sku": "35100",
            "size": "Standard Size",
            "description": "Kelly's Pad (Kellys Pad) is made up of good quality rubber, supplied with a pump. Widely used for delivery and obstetrics workflows."
        },
        "Silicone Nasal Sucker Suction Unit": {
            "name": "Silicone Nasal Sucker Suction Unit",
            "sku": "35060",
            "size": "Capacity: Approx. 80 ML",
            "description": "Premium food-grade silicone nasal sucker suction unit, easy to sterilize and designed for infant nasal congestion relief."
        },
        "Pessary Rubber Ring": {
            "name": "Pessary Rubber Ring",
            "sku": "35023, 35044 & 35045",
            "size": "50 mm, 75 mm & 87 mm",
            "description": "Pessary rubber ring gynecological support device. Medical-grade soft rubber ring construction."
        },
        "Nasal Aspirator": {
            "name": "Nasal Aspirator",
            "sku": "35022",
            "size": "Standard Size",
            "description": "Nasal Aspirator with premium rubber bulb fitted with a smooth plastic nozzle for safe baby congestion relief."
        }
    }
    
    # 1. Extract Images (Lowered threshold to 15px to be consistent and safe)
    raw_images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        
        base = doc.extract_image(xref)
        width, height = base["width"], base["height"]
        if width < 15 or height < 15:
            continue
            
        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2
        
        # Filter vertical center line template/logo graphics
        if 500 < cx < 600 and (cy < 80 or cy > 760):
            continue
            
        raw_images.append({
            "xref": xref,
            "rect": rect,
            "cx": cx,
            "cy": cy,
            "w": width,
            "h": height
        })
        
    # Sort images vertically, then horizontally
    raw_images.sort(key=lambda img: (img["cy"], img["cx"]))
    
    # 2. Extract Bold Titles
    bold_spans = []
    blocks_dict = page.get_text("dict")["blocks"]
    for b in blocks_dict:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                txt = span["text"].strip()
                if len(txt) < 2 or "MEDICAL RUBBER" in txt or "IndoSurgicals" in txt:
                    continue
                is_bold = "Bold" in span["font"] or span["size"] == 8.0
                if is_bold:
                    x0, y0, x1, y1 = span["bbox"]
                    bold_spans.append({
                        "text": txt,
                        "cx": (x0 + x1) / 2,
                        "cy": (y0 + y1) / 2,
                        "bbox": span["bbox"]
                    })
                    
    # Deduplicate bold spans
    unique_bold_spans = []
    seen = set()
    for s in bold_spans:
        key = (s["text"], round(s["cx"], 1), round(s["cy"], 1))
        if key not in seen:
            seen.add(key)
            unique_bold_spans.append(s)
            
    print(f"Found {len(raw_images)} product images and {len(unique_bold_spans)} bold titles.")
    
    metadata = []
    total_parsed = 0
    
    # 3. Spatial pairing: match each image to the closest bold title
    for idx, img in enumerate(raw_images):
        best_title = None
        best_dist = float('inf')
        
        for span in unique_bold_spans:
            dx = span["cx"] - img["cx"]
            dy = span["cy"] - img["cy"]
            
            if dy < -5:
                # penalize titles located above the image
                dist = dx**2 + 5 * (dy**2)
            else:
                dist = dx**2 + dy**2
                
            if dist < best_dist:
                best_dist = dist
                best_title = span
                
        if best_title:
            title_text = best_title["text"]
            
            # Map key transformations
            clean_title_key = title_text.strip()
            if "Kelly's Pad" in clean_title_key or "Kellys Pad" in clean_title_key:
                clean_title_key = "Kelly's Pad with Pump"
                
            mapped = RUBBER_METADATA_MAP.get(clean_title_key, {
                "name": clean_title_key,
                "sku": "N/A",
                "size": "Standard Size",
                "description": "Medical-grade rubber product catalog item."
            })
            
            category = "Medical Rubber Products"
            
            # Crop image
            best_rect = img["rect"]
            padded_rect = fitz.Rect(
                max(0, best_rect.x0 - 3),
                max(0, best_rect.y0 - 3),
                min(page.rect.x1, best_rect.x1 + 3),
                min(page.rect.y1, best_rect.y1 + 3)
            )
            
            class_id = f"rub_p00_r{idx:02d}"
            img_filename = f"{class_id}.png"
            img_path = out_images_dir / img_filename
            
            pix = page.get_pixmap(clip=padded_rect, dpi=200)
            pix.save(str(img_path))
            
            catalogs_arr = [{
                "catalog": "Medical Rubber Products Catalog",
                "page": 0,
                "image_path": f"dataset/processed/{class_id}_full.png"
            }]
            
            metadata.append({
                "id": class_id,
                "name": mapped["name"],
                "sku": mapped["sku"],
                "size": mapped["size"],
                "category": category,
                "page": 0,
                "image_path": f"dataset/processed/{img_filename}",
                "catalogs": json.dumps(catalogs_arr),
                "description": mapped["description"]
            })
            total_parsed += 1
            print(f"  Paired image {idx} -> '{mapped['name']}' (SKU: {mapped['sku']})")
            
    doc.close()
    print(f"Medical rubber products catalog parsing completed. Mapped: {total_parsed} items.")
    return metadata

def parse_hospital_furniture_catalog(pdf_path, out_images_dir, logo_hashes):
    """
    Parses the 10-page hospital furniture and holloware catalog.
    Uses closest vertical distance spatial 2D pairing to pair product images to titles,
    Poppins-Medium title extraction, and Poppins-Light description gathering.
    """
    print("\n--- Parsing Hospital Furniture & Holloware Catalog ---")
    doc = fitz.open(pdf_path)
    metadata = []
    total_parsed = 0

    HOLLOWARE_MAP = {
        "Kidney Tray (Stainless Steel)": {
            "sku": "70000 to 70007",
            "size": "150mm (6\") to 300mm (12\")",
            "description": "Made from high quality stainless steel which is long lasting and rust free. Without cover. Available in General & Deluxe quality.",
            "category": "Hospital Holloware"
        },
        "Kidney Tray (Polypropylene)": {
            "sku": "70077 to 70080",
            "size": "6\" to 12\"",
            "description": "These kidney trays are autoclavable and light in weight. Material: Polypropylene.",
            "category": "Hospital Holloware"
        },
        "Instrument Tray (Stainless Steel)": {
            "sku": "70008 to 70027",
            "size": "200x79x40mm to 450x300x50mm",
            "description": "These instrument trays are made from high quality stainless steel which is long lasting and rust free. Instrument tray with cover/lid. Available in General & Deluxe quality.",
            "category": "Hospital Holloware"
        }
    }

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        width = page.rect.width
        mid_x = width / 2.0
        
        text_info = page.get_text("dict")
        
        # 1. Group text spans by Left/Right side
        left_spans = []
        right_spans = []
        for block in text_info["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    txt = span["text"].strip()
                    if not txt:
                        continue
                    if span["bbox"][0] < mid_x:
                        left_spans.append(span)
                    else:
                        right_spans.append(span)

        # 2. Extract Left/Right titles (Poppins-Medium, size > 9.5)
        left_titles = []
        right_titles = []

        for span in left_spans:
            txt = span["text"].strip()
            if "Poppins-Medium" in span["font"] and span["size"] > 9.5:
                # Include standard 10xxx furniture items and holloware items
                is_furniture = False
                words = txt.split()
                if words and words[0].isdigit() and len(words[0]) == 5:
                    is_furniture = True
                is_holloware = txt in HOLLOWARE_MAP
                if is_furniture or is_holloware:
                    left_titles.append({"text": txt, "y": span["bbox"][1], "span": span, "is_holloware": is_holloware})

        for span in right_spans:
            txt = span["text"].strip()
            if "Poppins-Medium" in span["font"] and span["size"] > 9.5:
                is_furniture = False
                words = txt.split()
                if words and words[0].isdigit() and len(words[0]) == 5:
                    is_furniture = True
                is_holloware = txt in HOLLOWARE_MAP
                if is_furniture or is_holloware:
                    right_titles.append({"text": txt, "y": span["bbox"][1], "span": span, "is_holloware": is_holloware})

        # Sort titles vertically
        left_titles.sort(key=lambda x: x["y"])
        right_titles.sort(key=lambda x: x["y"])

        # 3. Extract Left/Right product images (width > 15, height > 15, excluding logos)
        left_images = []
        right_images = []

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            for r in rects:
                if r.width < 15 or r.height < 15:
                    continue
                # Skip middle logo/header templates
                if 500 < r.x0 < 520:
                    continue
                if r.y0 < 50 or r.y0 > 780:
                    continue
                
                img_item = {"xref": xref, "rect": r, "y": r.y0}
                if r.x0 < mid_x:
                    left_images.append(img_item)
                else:
                    right_images.append(img_item)

        # 4. Process Left Side Items
        left_page_num = 50 + (page_idx * 2)
        for i, title in enumerate(left_titles):
            title_y = title["y"]
            next_y = left_titles[i+1]["y"] if i + 1 < len(left_titles) else 780.0
            
            # Gather description bullet points under this title
            desc_lines = []
            for span in left_spans:
                x0, y0, x1, y1 = span["bbox"]
                if title_y + 5 < y0 < next_y - 2:
                    if x0 > 350: # exclude image area
                        continue
                    if "Poppins-Light" in span["font"] or span["size"] < 9.0:
                        txt = span["text"].strip()
                        if txt and txt != "»":
                            desc_lines.append(txt)

            # Parse metadata
            if title["is_holloware"]:
                mapped = HOLLOWARE_MAP[title["text"]]
                sku = mapped["sku"]
                name = title["text"]
                size = mapped["size"]
                description = mapped["description"]
                category = mapped["category"]
            else:
                match = re.match(r'^(\d{5})\s*-\s*(.*)$', title["text"])
                sku = match.group(1).strip() if match else "N/A"
                name = match.group(2).strip() if match else title["text"]
                category = "Hospital Furniture"
                # Size extraction from description
                size = "Standard Size"
                for line in desc_lines:
                    if "size" in line.lower() or "approx" in line.lower() or "dimension" in line.lower():
                        size = line
                        break
                description = " ".join(desc_lines) if desc_lines else "Hospital furniture catalog item."

            # Find closest image vertically
            best_img = None
            min_dist = float('inf')
            for img in left_images:
                dist = abs(img["y"] - title_y)
                if dist < min_dist:
                    min_dist = dist
                    best_img = img

            if best_img:
                class_id = f"furn_p{left_page_num:02d}_l{i:02d}"
                img_filename = f"{class_id}.png"
                img_path = out_images_dir / img_filename
                
                # Save cropped image
                best_rect = best_img["rect"]
                padded_rect = fitz.Rect(
                    max(0, best_rect.x0 - 3),
                    max(0, best_rect.y0 - 3),
                    min(page.rect.x1, best_rect.x1 + 3),
                    min(page.rect.y1, best_rect.y1 + 3)
                )
                pix = page.get_pixmap(clip=padded_rect, dpi=200)
                pix.save(str(img_path))
                
                catalogs_arr = [{
                    "catalog": "Hospital Furniture Catalog",
                    "page": left_page_num,
                    "image_path": f"dataset/processed/{class_id}_full.png"
                }]
                
                metadata.append({
                    "id": class_id,
                    "name": name,
                    "sku": sku,
                    "size": size,
                    "category": category,
                    "page": left_page_num,
                    "image_path": f"dataset/processed/{img_filename}",
                    "catalogs": json.dumps(catalogs_arr),
                    "description": description
                })
                total_parsed += 1

        # 5. Process Right Side Items
        right_page_num = 50 + (page_idx * 2) + 1
        for i, title in enumerate(right_titles):
            title_y = title["y"]
            next_y = right_titles[i+1]["y"] if i + 1 < len(right_titles) else 780.0
            
            desc_lines = []
            for span in right_spans:
                x0, y0, x1, y1 = span["bbox"]
                if title_y + 5 < y0 < next_y - 2:
                    if x0 > 950: # exclude image area
                        continue
                    if "Poppins-Light" in span["font"] or span["size"] < 9.0:
                        txt = span["text"].strip()
                        if txt and txt != "»":
                            desc_lines.append(txt)

            if title["is_holloware"]:
                mapped = HOLLOWARE_MAP[title["text"]]
                sku = mapped["sku"]
                name = title["text"]
                size = mapped["size"]
                description = mapped["description"]
                category = mapped["category"]
            else:
                match = re.match(r'^(\d{5})\s*-\s*(.*)$', title["text"])
                sku = match.group(1).strip() if match else "N/A"
                name = match.group(2).strip() if match else title["text"]
                category = "Hospital Furniture"
                size = "Standard Size"
                for line in desc_lines:
                    if "size" in line.lower() or "approx" in line.lower() or "dimension" in line.lower():
                        size = line
                        break
                description = " ".join(desc_lines) if desc_lines else "Hospital furniture catalog item."

            best_img = None
            min_dist = float('inf')
            for img in right_images:
                dist = abs(img["y"] - title_y)
                if dist < min_dist:
                    min_dist = dist
                    best_img = img

            if best_img:
                class_id = f"furn_p{right_page_num:02d}_r{i:02d}"
                img_filename = f"{class_id}.png"
                img_path = out_images_dir / img_filename
                
                best_rect = best_img["rect"]
                padded_rect = fitz.Rect(
                    max(0, best_rect.x0 - 3),
                    max(0, best_rect.y0 - 3),
                    min(page.rect.x1, best_rect.x1 + 3),
                    min(page.rect.y1, best_rect.y1 + 3)
                )
                pix = page.get_pixmap(clip=padded_rect, dpi=200)
                pix.save(str(img_path))
                
                catalogs_arr = [{
                    "catalog": "Hospital Furniture Catalog",
                    "page": right_page_num,
                    "image_path": f"dataset/processed/{class_id}_full.png"
                }]
                
                metadata.append({
                    "id": class_id,
                    "name": name,
                    "sku": sku,
                    "size": size,
                    "category": category,
                    "page": right_page_num,
                    "image_path": f"dataset/processed/{img_filename}",
                    "catalogs": json.dumps(catalogs_arr),
                    "description": description
                })
                total_parsed += 1

    doc.close()
    print(f"Hospital Furniture catalog parsing completed. Mapped: {total_parsed} items.")
    return metadata

def build_unified_dataset():
    """
    Main orchestrator that parses all four catalogs and creates a unified metadata.csv database.
    """
    project_dir = Path("/Users/anika/Desktop/surgical_instrument_classifier")
    out_dir = project_dir / "dataset"
    out_images_dir = out_dir / "processed"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Identify logo hashes from the original catalog to ignore recurring background templates
    print("Identifying template/logo hashes...")
    doc = fitz.open(project_dir / "surgical-instrument-catalog.pdf")
    hash_counter = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                h = hashlib.md5(base["image"]).hexdigest()
                hash_counter[h] = hash_counter.get(h, 0) + 1
            except Exception:
                continue
    doc.close()
    
    # Identify logos that appear more than 5 times
    logo_hashes = {h for h, count in hash_counter.items() if count > 5}
    print(f"Identified {len(logo_hashes)} template hashes to ignore.")
    
    # 2. Parse Catalogs
    surgical_metadata = parse_surgical_catalog(
        project_dir / "surgical-instrument-catalog.pdf", 
        out_images_dir, 
        logo_hashes
    )
    
    ophthalmic_metadata = parse_ophthalmic_catalog(
        project_dir / "ophthalmic-instruments-catalog.pdf", 
        out_images_dir, 
        logo_hashes,
        surgical_metadata
    )
    
    medical_rubber_metadata = parse_medical_rubber_catalog(
        project_dir / "medical-rubber-products.pdf", 
        out_images_dir, 
        logo_hashes
    )

    hospital_furniture_metadata = parse_hospital_furniture_catalog(
        project_dir / "hospital-furniture.pdf",
        out_images_dir,
        logo_hashes
    )
    
    # Combine metadata
    unified_metadata = surgical_metadata + ophthalmic_metadata + medical_rubber_metadata + hospital_furniture_metadata
    
    # Convert to DataFrame and export
    metadata_df = pd.DataFrame(unified_metadata)
    metadata_csv_path = out_dir / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    unique_classes = len(metadata_df["id"].unique())
    print("\n" + "="*50)
    print("UNIFIED CATALOG BUILDING COMPLETE")
    print("="*50)
    print(f"Total entries in database: {len(metadata_df)}")
    print(f"Total unique classes in model: {unique_classes}")
    print(f"Spreadsheet saved to: {metadata_csv_path}")
    print(f"Processed cropped images saved in: {out_images_dir}")
    print("="*50)

if __name__ == "__main__":
    build_unified_dataset()
