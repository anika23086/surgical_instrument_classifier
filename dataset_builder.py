import fitz
import hashlib
import pandas as pd
from pathlib import Path
import re

def clean_reconstructed_text(text):
    """
    Cleans up spacing and removes duplicate labels in cell text.
    """
    text = text.strip()
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove redundant repeating headers if PyMuPDF extracted them twice
    return text

def parse_catalog(pdf_path, output_dir):
    print("Initializing PDF catalog parser...")
    doc = fitz.open(pdf_path)
    
    out_images_dir = Path(output_dir) / "processed"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Identify logo hashes to ignore background template graphics
    print("Scanning document for recurring template/logo graphics...")
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
                
    logo_hashes = {h for h, count in hash_counter.items() if count > 5}
    print(f"Identified {len(logo_hashes)} logo/template hashes to filter out.")

    # 2. Grid Coordinates Setup (2 Rows x 6 Columns)
    # Left page has columns 0, 1, 2. Right page has columns 3, 4, 5.
    X_COLS = [0, 150, 280, 420, 570, 700, 850]
    Y_ROWS = [50, 300, 550]
    
    metadata = []
    total_parsed = 0
    
    print("\nStarting page-by-page grid analysis and extraction...")
    # Loop from page 2 to 21 (grid pages)
    for page_num in range(2, len(doc) - 1):
        page = doc[page_num]
        print(f"Processing Page {page_num}...")
        
        # 2a. Dynamic Category Header Extraction
        # Left header is y < 50, x < 420. Right header is y < 50, x >= 420.
        left_header_words = []
        right_header_words = []
        
        words = page.get_text("words")
        for w in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = w
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            
            # Header filter
            if cy < 50:
                # Exclude website, copyright, or page numbers
                if "Copyright" in text or "indosurgicals" in text or text.isdigit() or "supplier" in text or "medical" in text:
                    continue
                if cx < 420:
                    left_header_words.append(w)
                else:
                    right_header_words.append(w)
                    
        # Sort and join headers
        left_header_words.sort(key=lambda w: (w[5], w[6], w[7]))
        right_header_words.sort(key=lambda w: (w[5], w[6], w[7]))
        left_category = " ".join([w[4] for w in left_header_words]).strip()
        right_category = " ".join([w[4] for w in right_header_words]).strip()
        
        # Fallbacks for empty categories
        if not left_category: left_category = "General Instruments"
        if not right_category: right_category = left_category
        
        # 2b. Map Words to Grid Cells
        grid_words = { (r, c): [] for r in range(2) for c in range(6) }
        for w in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = w
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            
            # Skip header area, page margins, website links, or logo symbols
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
                
        # 2c. Map Images to Grid Cells
        grid_images = { (r, c): [] for r in range(2) for c in range(6) }
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            
            # Skip tiny images
            base = doc.extract_image(xref)
            width, height = base["width"], base["height"]
            if width < 30 or height < 30:
                continue
                
            # Filter logos
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

        # 2d. Pair Images and Labels in each Cell
        for r in range(2):
            for c in range(6):
                txt_words = grid_words[(r, c)]
                imgs = grid_images[(r, c)]
                
                if txt_words and imgs:
                    # Reconstruct text sorted logically by line number and word number
                    txt_words.sort(key=lambda w: (w[5], w[6], w[7]))
                    raw_text = " ".join([w[4] for w in txt_words])
                    clean_text = clean_reconstructed_text(raw_text)
                    
                    # Extract fields using regex
                    # Name is usually at the start or end of the text. 
                    # Let's clean up SKU and Size patterns to separate the instrument name.
                    sku_match = re.search(r'SKU:\s*([A-Za-z0-9\-/]+(?:\s+SKU:\s*[A-Za-z0-9\-/]+)*)', clean_text, re.IGNORECASE)
                    size_match = re.search(r'Size:\s*([^S]+)', clean_text, re.IGNORECASE)
                    
                    sku = sku_match.group(1).strip() if sku_match else "N/A"
                    size = size_match.group(1).strip() if size_match else "N/A"
                    
                    # Remove SKU: ... and Size: ... to get the clean instrument name
                    name = clean_text
                    name = re.sub(r'SKU:\s*[A-Za-z0-9\-/]+', '', name, flags=re.IGNORECASE)
                    name = re.sub(r'Size:\s*[^S]+', '', name, flags=re.IGNORECASE)
                    name = re.sub(r'\s+', ' ', name).strip()
                    
                    # If name is empty, fall back to clean text
                    if not name or len(name) < 3:
                        name = clean_text
                        
                    category = left_category if c < 3 else right_category
                    
                    # Get the largest image in the cell (or first if equal) to represent the tool
                    imgs.sort(key=lambda x: x[1] * x[2], reverse=True)
                    best_rect = imgs[0][0]
                    
                    # Dynamic Cropping from PDF Page: ensures beautiful high-res render and crisp white background
                    # Expand the rect slightly to ensure no parts of the instrument are clipped
                    padded_rect = fitz.Rect(
                        max(0, best_rect.x0 - 3),
                        max(0, best_rect.y0 - 3),
                        min(page.rect.x1, best_rect.x1 + 3),
                        min(page.rect.y1, best_rect.y1 + 3)
                    )
                    
                    img_filename = f"p{page_num:02d}_r{r}_c{c}.png"
                    img_path = out_images_dir / img_filename
                    
                    # Render cropped area at high resolution (DPI=200)
                    pix = page.get_pixmap(clip=padded_rect, dpi=200)
                    pix.save(str(img_path))
                    
                    metadata.append({
                        "id": f"p{page_num:02d}_r{r}_c{c}",
                        "name": name,
                        "sku": sku,
                        "size": size,
                        "category": category,
                        "page": page_num,
                        "image_path": f"dataset/processed/{img_filename}"
                    })
                    total_parsed += 1

    doc.close()
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = Path(output_dir) / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"\nCompleted catalog extraction successfully!")
    print(f"Total instrument records mapped: {total_parsed}")
    print(f"Spreadsheet saved to: {metadata_csv_path}")
    print(f"High-resolution cropped images saved in: {out_images_dir}")

if __name__ == "__main__":
    parse_catalog("surgical-instrument-catalog.pdf", ".")
