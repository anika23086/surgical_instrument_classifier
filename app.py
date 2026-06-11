import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from search_engine import SurgicalInstrumentSearchEngine
from PIL import Image
import io

app = Flask(__name__)

# Initialize search engine (loads ResNet-50 backbone once)
PROJECT_DIR = Path("/Users/anika/Desktop/surgical_instrument_classifier")
engine = SurgicalInstrumentSearchEngine(
    cache_path=str(PROJECT_DIR / "dataset/classifier_resnet50.pt"),
    metadata_path=str(PROJECT_DIR / "dataset/metadata.csv"),
    mapping_path=str(PROJECT_DIR / "dataset/class_mapping.json")
)

# ─────────────────────────────────────────────────────────────
# Existing Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/catalog", methods=["GET"])
def get_catalog():
    if engine.metadata_df is None:
        return jsonify([])
    # Convert dataframe to records
    records = engine.metadata_df.to_dict(orient="records")
    return jsonify(records)

@app.route("/api/classify", methods=["POST"])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Read the file directly into memory
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Get matching items
        top_matches = engine.query_image(image, top_k=5)
        
        return jsonify({
            "success": True,
            "results": top_matches
        })
    except Exception as e:
        print(f"Error classifying image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/dataset/processed/<path:filename>")
def serve_processed_images(filename):
    return send_from_directory(PROJECT_DIR / "dataset/processed", filename)

# ─────────────────────────────────────────────────────────────
# New Pipeline Routes (Automated Catalog Ingestion)
# ─────────────────────────────────────────────────────────────

@app.route("/api/upload-catalog", methods=["POST"])
def upload_catalog():
    """
    Receive a PDF catalog file and start the extraction pipeline.
    Returns a job_id that can be polled for status.
    """
    if 'catalog' not in request.files:
        return jsonify({"error": "No catalog PDF file provided"}), 400

    file = request.files['catalog']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    try:
        from config import UPLOADS_DIR, GROQ_API_KEY
        from pipeline import create_job, start_pipeline_thread

        if not GROQ_API_KEY or GROQ_API_KEY in ("test_mock_groq_key", ""):
            return jsonify({"error": "Groq API key not configured. Please add your key in Settings first."}), 400

        # Save uploaded PDF
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        pdf_filename = file.filename
        pdf_path = UPLOADS_DIR / pdf_filename
        file.save(str(pdf_path))

        # Create job and start background pipeline
        job_id = create_job(pdf_filename)
        start_pipeline_thread(job_id, str(pdf_path), engine)

        return jsonify({"job_id": job_id, "filename": pdf_filename})

    except Exception as e:
        print(f"Error starting pipeline: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline-status/<job_id>", methods=["GET"])
def pipeline_status(job_id):
    """
    Poll for pipeline progress. Returns the current stage, message,
    progress percentage, completion flag, and any error.
    """
    from pipeline import get_job

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "stage": job["stage"],
        "message": job["message"],
        "progress_pct": job["progress_pct"],
        "is_complete": job["is_complete"],
        "error": job.get("error"),
        "stats": job.get("stats", {}),
    })


@app.route("/api/pipeline-preview/<job_id>", methods=["GET"])
def pipeline_preview(job_id):
    """
    Return the array of AI-extracted items for the review UI.
    Only available once the extraction stage is complete (stage='review').
    """
    from pipeline import get_job

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["stage"] not in ("review",):
        return jsonify({"error": "Items not ready for review yet", "stage": job["stage"]}), 400

    # Return items without internal fields
    preview_items = []
    for item in job["items"]:
        preview_items.append({
            "name": item.get("name", ""),
            "sku": item.get("sku", ""),
            "category": item.get("category", ""),
            "description": item.get("description", ""),
            "size": item.get("size", ""),
            "page": item.get("page", 0),
            "confidence": item.get("confidence", "low"),
            "image_base64": item.get("image_base64", ""),
            "raw_image_path": item.get("raw_image_path", ""),
        })

    return jsonify({
        "items": preview_items,
        "pdf_filename": job.get("pdf_filename", "")
    })


@app.route("/api/pipeline-approve/<job_id>", methods=["POST"])
def pipeline_approve(job_id):
    """
    Receive edited items from the user and resume the pipeline.
    The frontend sends the full edited array; deleted items are simply omitted.
    """
    from pipeline import get_job

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["stage"] != "review":
        return jsonify({"error": "Job is not awaiting review"}), 400

    data = request.get_json()
    if not data or "items" not in data:
        return jsonify({"error": "No items provided"}), 400

    approved_items = data["items"]

    # Merge user edits back with the original raw_image_path
    original_items = {item.get("raw_image_path", ""): item for item in job["items"]}
    for approved in approved_items:
        raw_path = approved.get("raw_image_path", "")
        if raw_path in original_items:
            # Keep the original raw_image_path, let user override name/sku/category
            approved["raw_image_path"] = raw_path

    job["approved_items"] = approved_items
    job["review_event"].set()  # Resume the pipeline thread

    return jsonify({"success": True, "message": f"Approved {len(approved_items)} items. Training starting..."})


@app.route("/api/settings", methods=["POST"])
def save_settings():
    """
    Save Groq API key to the .env file.
    """
    from config import ENV_FILE

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    groq_key = data.get("groq_api_key", "").strip()
    if not groq_key:
        return jsonify({"error": "Groq API key must be provided"}), 400

    try:
        # Read existing .env lines
        env_lines = []
        groq_found = False

        if ENV_FILE.exists():
            with open(ENV_FILE, "r") as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY="):
                        env_lines.append(f"GROQ_API_KEY={groq_key}\n")
                        groq_found = True
                    else:
                        env_lines.append(line)

        if not groq_found:
            env_lines.append(f"GROQ_API_KEY={groq_key}\n")

        with open(ENV_FILE, "w") as f:
            f.writelines(env_lines)

        # Update runtime config immediately
        import config
        os.environ["GROQ_API_KEY"] = groq_key
        config.GROQ_API_KEY = groq_key

        return jsonify({"success": True, "message": "Groq API key saved successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Check if Groq API key is configured."""
    from config import GROQ_API_KEY
    
    # Filter out mock keys
    has_groq = bool(GROQ_API_KEY) and GROQ_API_KEY not in ("test_mock_groq_key", "")
    
    return jsonify({
        "has_api_key": has_groq,
        "has_groq_key": has_groq,
        "provider": "groq",
    })


@app.route("/api/upload-image", methods=["POST"])
def upload_image():
    """
    Handle uploading a single raw image during the manual 'Add Item' review step.
    Saves the image to the raw dataset folder and returns its path.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        from config import RAW_DIR
        import uuid
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        ext = Path(file.filename).suffix or ".png"
        unique_filename = f"manual_{uuid.uuid4().hex[:12]}{ext}"
        dest_path = RAW_DIR / unique_filename
        file.save(str(dest_path))
        
        return jsonify({
            "success": True,
            "raw_image_path": f"dataset/raw/{unique_filename}"
        })
    except Exception as e:
        print(f"Error saving manually uploaded image: {e}")
        return jsonify({"error": str(e)}), 500


# Serve uploaded raw images for the review preview
@app.route("/dataset/raw/<path:filename>")
def serve_raw_images(filename):
    return send_from_directory(PROJECT_DIR / "dataset/raw", filename)


# Serve uploaded catalog PDFs for the split review panel
@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    from config import UPLOADS_DIR
    return send_from_directory(UPLOADS_DIR, filename)


if __name__ == "__main__":
    # Run on 5001 to avoid conflicts with typical macOS system service listeners (like AirPlay on 5000)
    app.run(host="0.0.0.0", port=5001, debug=True)
