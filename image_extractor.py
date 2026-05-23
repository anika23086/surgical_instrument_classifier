import fitz  # pymupdf
from pathlib import Path

def extract_all_images(pdf_path: str, out_dir: str):
    doc = fitz.open(pdf_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []  # will hold (filename, page_num) for your reference

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            base = doc.extract_image(xref)

            ext      = base["ext"]           # usually "png" or "jpeg"
            img_data = base["image"]
            width    = base["width"]
            height   = base["height"]

            # Skip tiny images (logos, icons, decorative elements)
            if width < 100 or height < 100:
                continue

            filename = f"p{page_num:02d}_i{img_idx:02d}.{ext}"
            filepath = out / filename

            with open(filepath, "wb") as f:
                f.write(img_data)

            records.append({
                "filename": filename,
                "page":     page_num,
                "width":    width,
                "height":   height,
                "label":    ""          # to be filled in next step
            })
            print(f"Saved {filename}  ({width}x{height})")

    doc.close()
    return records

records = extract_all_images("surgical-instrument-catalog.pdf", "dataset/raw")
print(f"\nExtracted {len(records)} images")