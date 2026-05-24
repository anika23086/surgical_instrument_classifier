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

if __name__ == "__main__":
    # Run on 5001 to avoid conflicts with typical macOS system service listeners (like AirPlay on 5000)
    app.run(host="0.0.0.0", port=5001, debug=True)
