"""
Background Pipeline Orchestrator for Automated Catalog Ingestion.

Chains every step together as a background thread so the Flask server
stays responsive. Each job gets a UUID and exposes a status dict that
the frontend polls via /api/pipeline-status.

Stages:
  1. PDF Conversion
  2. Image Extraction
  3. AI Labeling
  4. Awaiting Review (paused until user approves)
  5. Deduplication
  6. Image Processing (crop multiscale)
  7. Database Update (metadata.csv)
  8. Model Training
  9. Model Reload
  10. Complete
"""

import os
import sys
import uuid
import time
import json
import threading
import traceback
from pathlib import Path
from io import StringIO

import pandas as pd
from PIL import Image

from config import (
    BASE_DIR, DATASET_DIR, RAW_DIR, PROCESSED_DIR, UPLOADS_DIR,
    METADATA_CSV, CLASS_MAPPING_JSON, MODEL_WEIGHTS, TRAINING_EPOCHS
)
from utils import crop_multiscale_regions


# ---------------------------------------------------------------------------
# Job status tracking
# ---------------------------------------------------------------------------
active_jobs: dict[str, dict] = {}


def get_job(job_id: str) -> dict | None:
    return active_jobs.get(job_id)


def create_job(pdf_filename: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    active_jobs[job_id] = {
        "id": job_id,
        "pdf_filename": pdf_filename,
        "stage": "queued",
        "message": "Waiting to start...",
        "progress_pct": 0,
        "is_complete": False,
        "error": None,
        "items": [],             # extracted items for review
        "approved_items": None,  # user-edited items (set on approve)
        "review_event": threading.Event(),  # signals when user approves
        "stats": {},             # final statistics
    }
    return job_id


# ---------------------------------------------------------------------------
# Main pipeline — runs in a background thread
# ---------------------------------------------------------------------------
def run_pipeline(job_id: str, pdf_path: str, engine):
    """
    Execute the full ingestion pipeline in a background thread.

    Args:
        job_id: The job UUID
        pdf_path: Absolute path to the uploaded PDF
        engine: The SurgicalInstrumentSearchEngine instance (for hot-reload)
    """
    job = active_jobs[job_id]

    try:
        # ==============================================================
        # STAGES 1-3: AI Extraction (parse_catalog handles all three)
        # ==============================================================
        _update(job, "extracting", "Starting catalog extraction...", 2)

        from catalog_parser_ai import CatalogParserAI
        from config import GROQ_API_KEY

        if not GROQ_API_KEY:
            raise ValueError("Groq API key not configured. Please add your GROQ_API_KEY in Settings.")
        parser = CatalogParserAI(api_key=GROQ_API_KEY)

        def progress_cb(stage, message, pct):
            _update(job, stage, message, pct)

        extracted_items = parser.parse_catalog(pdf_path, progress_callback=progress_cb)

        if not extracted_items:
            _update(job, "complete", "No product images found in this PDF.", 100)
            job["is_complete"] = True
            return

        # Encode images as base64 for frontend preview
        import base64
        from io import BytesIO

        for item in extracted_items:
            if item.get("image_data"):
                buf = BytesIO()
                item["image_data"].save(buf, format="PNG")
                item["image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
                del item["image_data"]  # Don't keep PIL objects in the dict

        job["items"] = extracted_items

        # ==============================================================
        # STAGE 4: Await user review
        # ==============================================================
        _update(job, "review", "Ready for review", 92)

        # Block this thread until the user clicks "Approve & Train"
        job["review_event"].wait()

        # User has approved — get the edited items
        approved_items = job["approved_items"]
        if not approved_items:
            _update(job, "complete", "No items approved. Pipeline cancelled.", 100)
            job["is_complete"] = True
            return

        # ==============================================================
        # STAGE 5: Deduplication against existing database
        # ==============================================================
        _update(job, "deduplicating", "Checking for duplicates...", 93)

        existing_skus = set()
        existing_names = set()
        if METADATA_CSV.exists():
            existing_df = pd.read_csv(METADATA_CSV).fillna("")
            existing_skus = set(str(s).strip() for s in existing_df["sku"].tolist() if str(s).strip())
            existing_names = set(str(n).strip().lower() for n in existing_df["name"].tolist() if str(n).strip())

        new_items = []
        duplicates = 0
        for item in approved_items:
            sku = str(item.get("sku", "")).strip()
            name = str(item.get("name", "")).strip()

            if sku and sku in existing_skus:
                duplicates += 1
                continue
            if name and name.lower() in existing_names:
                duplicates += 1
                continue
            if not name:
                continue  # Skip items with no name at all
            new_items.append(item)

        if not new_items:
            _update(job, "complete", f"All items are duplicates of existing records. Nothing to add.", 100)
            job["is_complete"] = True
            job["stats"] = {"new_items": 0, "duplicates": duplicates}
            return

        _update(job, "deduplicating",
                f"Found {len(new_items)} new items ({duplicates} duplicates skipped)", 94)

        # ==============================================================
        # STAGE 6: Image Processing — crop multiscale regions
        # ==============================================================
        _update(job, "processing", "Processing images...", 95)

        metadata_rows = []
        for i, item in enumerate(new_items):
            raw_path_str = item.get("raw_image_path", "")
            raw_path = BASE_DIR / raw_path_str

            if not raw_path.exists():
                print(f"  Warning: Raw image missing for {item.get('name')}: {raw_path}")
                continue

            # Generate a unique ID for this item
            category_prefix = _category_prefix(item.get("category", ""))
            page_num = item.get("page", 0)
            item_id = f"{category_prefix}_p{page_num:02d}_ai{i:02d}"

            try:
                img = Image.open(raw_path).convert("RGB")

                # Determine if this category uses rubber-style cropping
                is_rubber = category_prefix in ("rub", "furn", "holl", "scal", "auto", "misc")
                full_body, jaw, inset = crop_multiscale_regions(img, is_rubber=is_rubber)

                # Save crops
                full_name = f"{item_id}_full.png"
                jaw_name = f"{item_id}_jaw.png"
                full_body.save(PROCESSED_DIR / full_name)
                jaw.save(PROCESSED_DIR / jaw_name)

                inset_path_str = ""
                if inset is not None:
                    inset_name = f"{item_id}_inset.png"
                    inset.save(PROCESSED_DIR / inset_name)
                    inset_path_str = f"dataset/processed/{inset_name}"

                # Also save the raw image into processed for display consistency
                raw_display_name = f"{item_id}.png"
                img.save(PROCESSED_DIR / raw_display_name)

                catalog_name = item.get("source_catalog", job["pdf_filename"])
                catalogs_json = json.dumps([{
                    "catalog": catalog_name,
                    "page": item.get("page", 0)
                }])

                metadata_rows.append({
                    "id": item_id,
                    "name": item.get("name", ""),
                    "sku": item.get("sku", ""),
                    "size": item.get("size", ""),
                    "category": item.get("category", "General Medical"),
                    "page": item.get("page", 0),
                    "image_path": f"dataset/processed/{full_name}",
                    "jaw_path": f"dataset/processed/{jaw_name}",
                    "inset_path": inset_path_str,
                    "original_image_path": f"dataset/processed/{raw_display_name}",
                    "catalogs": catalogs_json,
                    "description": item.get("description", ""),
                })
            except Exception as e:
                print(f"  Error processing item {item_id}: {e}")
                continue

            pct = 95 + int(2 * (i + 1) / len(new_items))
            _update(job, "processing", f"Processing image {i + 1}/{len(new_items)}...", pct)

        if not metadata_rows:
            _update(job, "complete", "No valid images could be processed.", 100)
            job["is_complete"] = True
            return

        # ==============================================================
        # STAGE 7: Database Update — append to metadata.csv
        # ==============================================================
        _update(job, "updating", "Updating catalog database...", 97)

        new_df = pd.DataFrame(metadata_rows)
        if METADATA_CSV.exists():
            existing_df = pd.read_csv(METADATA_CSV)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_csv(METADATA_CSV, index=False)

        # ==============================================================
        # STAGE 8: Model Training — retrain classifier
        # ==============================================================
        _update(job, "training", "Starting model training...", 97)

        # Capture training output for real-time epoch updates
        training_success = _run_training_with_progress(job)

        if not training_success:
            _update(job, "error", "Model training failed. Database was updated successfully — you can retrain manually.", 100)
            job["error"] = "Training failed. See server logs for details."
            return

        # ==============================================================
        # STAGE 9: Hot-swap model in the live search engine
        # ==============================================================
        _update(job, "reloading", "Loading new model...", 99)

        try:
            engine.reload()
        except Exception as e:
            print(f"  Warning: Hot-reload failed: {e}. Server restart needed.")

        # ==============================================================
        # STAGE 10: Complete!
        # ==============================================================
        num_new = len(metadata_rows)
        job["stats"] = {
            "new_items": num_new,
            "duplicates": duplicates,
            "total_database": len(combined_df),
        }
        _update(job, "complete", f"✓ Done! {num_new} new instruments added to database.", 100)
        job["is_complete"] = True

    except Exception as e:
        traceback.print_exc()
        job["error"] = str(e)
        job["stage"] = "error"
        job["message"] = f"Pipeline error: {str(e)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update(job: dict, stage: str, message: str, pct: int):
    """Update job status atomically."""
    job["stage"] = stage
    job["message"] = message
    job["progress_pct"] = min(pct, 100)


def _category_prefix(category: str) -> str:
    """Generate a short prefix from a category name for unique IDs."""
    category_lower = category.lower()
    mapping = {
        "surgical forceps": "forc",
        "surgical scissors": "scis",
        "surgical retractors": "retr",
        "clamps": "clam",
        "needle holders": "need",
        "ophthalmic instruments": "op",
        "medical rubber products": "rub",
        "hospital furniture": "furn",
        "hospital holloware": "holl",
        "height & weight scales": "scal",
        "autoclave & sterilizer": "auto",
    }
    for key, prefix in mapping.items():
        if key in category_lower:
            return prefix
    return "misc"


def _run_training_with_progress(job: dict) -> bool:
    """
    Run the training script and capture epoch output for real-time progress.
    Returns True on success, False on failure.
    """
    try:
        # Import the training function directly
        from train_classifier import train_model

        # Capture stdout to parse epoch progress
        old_stdout = sys.stdout
        captured = StringIO()

        class TeeOutput:
            """Write to both the real stdout and our capture buffer."""
            def __init__(self, real_stdout, capture_buffer):
                self.real = real_stdout
                self.capture = capture_buffer

            def write(self, text):
                self.real.write(text)
                self.capture.write(text)
                # Parse epoch lines for progress updates
                if "Epoch" in text and "/" in text:
                    try:
                        # Format: "Epoch 23/50 | Loss: ... | Acc: ..."
                        parts = text.split("|")[0].strip()
                        epoch_part = parts.split("Epoch")[1].strip()
                        current, total = epoch_part.split("/")
                        current = int(current)
                        total = int(total)
                        pct = 97 + int(2 * current / total)
                        _update(job, "training", f"Training epoch {current}/{total}...", pct)
                    except (IndexError, ValueError):
                        pass

            def flush(self):
                self.real.flush()
                self.capture.flush()

        sys.stdout = TeeOutput(old_stdout, captured)

        try:
            train_model()
            return True
        finally:
            sys.stdout = old_stdout

    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        return False


def start_pipeline_thread(job_id: str, pdf_path: str, engine) -> None:
    """Launch the pipeline in a daemon background thread."""
    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, pdf_path, engine),
        daemon=True,
        name=f"pipeline-{job_id}"
    )
    thread.start()
