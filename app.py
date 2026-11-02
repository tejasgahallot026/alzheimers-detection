"""Flask web app that serves a trained MRI classifier.

The model is produced by notebooks/train_model.ipynb and exported to
artifacts/model.tflite with its class names in artifacts/labels.json.
If those files are missing the app refuses to predict (HTTP 503)
rather than returning a made-up result.
"""
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "artifacts" / "model.tflite"))
LABELS_PATH = Path(os.environ.get("LABELS_PATH", BASE_DIR / "artifacts" / "labels.json"))

# Must match the training notebook: images are 128 x 128.
IMG_WIDTH, IMG_HEIGHT = 128, 128
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DISCLAIMER = "Research demo only. Not a medical device and not for clinical decisions."

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alz-app")


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Turn uploaded bytes into a (1, 128, 128, 3) float32 batch with values 0-255.

    Scaling to the network's range happens inside the model itself,
    so training and serving cannot drift apart.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


class TFLitePredictor:
    """Thin wrapper around a TFLite interpreter."""

    def __init__(self, model_path: Path, labels_path: Path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # fall back to full TensorFlow if installed
            from tensorflow.lite import Interpreter  # type: ignore
        self.labels = json.loads(Path(labels_path).read_text())
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_index, batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def load_predictor():
    if not (MODEL_PATH.exists() and LABELS_PATH.exists()):
        log.warning("Model files not found at %s; predictions disabled.", MODEL_PATH)
        return None
    predictor = TFLitePredictor(MODEL_PATH, LABELS_PATH)
    log.info("Loaded model %s with classes %s", MODEL_PATH.name, predictor.labels)
    return predictor


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(predictor="auto") -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB uploads
    app.predictor = load_predictor() if predictor == "auto" else predictor

    @app.get("/")
    def index():
        return render_template("index.html", disclaimer=DISCLAIMER)

    @app.get("/health")
    def health():
        ready = app.predictor is not None
        return jsonify(status="ok" if ready else "model_missing", model_loaded=ready), (
            200 if ready else 503
        )

    @app.post("/predict")
    def predict():
        if app.predictor is None:
            return jsonify(error="Model is not loaded, so no prediction can be made."), 503

        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify(error="Attach an MRI image in the 'image' field."), 400
        if not allowed_file(file.filename):
            return jsonify(error="Upload a PNG or JPEG image."), 400

        try:
            batch = preprocess(file.read())
        except (UnidentifiedImageError, OSError):
            return jsonify(error="The file could not be read as an image."), 400

        start = time.perf_counter()
        probs = np.asarray(app.predictor.predict(batch), dtype=float)
        elapsed_ms = (time.perf_counter() - start) * 1000

        labels = app.predictor.labels
        top = int(np.argmax(probs))
        log.info("Predicted %s (%.3f) in %.0f ms", labels[top], probs[top], elapsed_ms)
        return jsonify(
            prediction=labels[top],
            confidence=round(float(probs[top]), 4),
            probabilities={label: round(float(p), 4) for label, p in zip(labels, probs)},
            inference_ms=round(elapsed_ms, 1),
            disclaimer=DISCLAIMER,
        )

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Image is larger than 5 MB."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
