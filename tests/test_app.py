"""API tests. They use a fake predictor, so no trained model is needed."""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import IMG_HEIGHT, IMG_WIDTH, create_app, preprocess  # noqa: E402

LABELS = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


class FakePredictor:
    labels = LABELS

    def __init__(self):
        self.last_shape = None

    def predict(self, batch):
        self.last_shape = batch.shape
        return np.array([0.1, 0.05, 0.8, 0.05], dtype=np.float32)


def make_png(size=(300, 300)):
    buf = io.BytesIO()
    Image.new("L", size, color=128).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_preprocess_shape_and_range():
    batch = preprocess(make_png().read())
    assert batch.shape == (1, IMG_HEIGHT, IMG_WIDTH, 3)
    assert batch.dtype == np.float32
    assert 0 <= batch.min() and batch.max() <= 255


def test_health_reports_model_loaded():
    client = create_app(predictor=FakePredictor()).test_client()
    res = client.get("/health")
    assert res.status_code == 200 and res.get_json()["model_loaded"] is True


def test_predict_returns_top_class_and_all_probabilities():
    fake = FakePredictor()
    client = create_app(predictor=fake).test_client()
    res = client.post("/predict", data={"image": (make_png(), "scan.png")},
                      content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 200
    assert body["prediction"] == "NonDemented"
    assert body["confidence"] == 0.8
    assert set(body["probabilities"]) == set(LABELS)
    assert fake.last_shape == (1, IMG_HEIGHT, IMG_WIDTH, 3)


def test_predict_without_model_refuses_instead_of_guessing():
    client = create_app(predictor=None).test_client()
    res = client.post("/predict", data={"image": (make_png(), "scan.png")},
                      content_type="multipart/form-data")
    assert res.status_code == 503
    assert client.get("/health").status_code == 503


def test_predict_rejects_missing_file():
    client = create_app(predictor=FakePredictor()).test_client()
    assert client.post("/predict").status_code == 400


def test_predict_rejects_wrong_file_type():
    client = create_app(predictor=FakePredictor()).test_client()
    res = client.post("/predict", data={"image": (io.BytesIO(b"hello"), "notes.txt")},
                      content_type="multipart/form-data")
    assert res.status_code == 400


def test_predict_rejects_corrupt_image():
    client = create_app(predictor=FakePredictor()).test_client()
    res = client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "scan.png")},
                      content_type="multipart/form-data")
    assert res.status_code == 400


def test_index_shows_disclaimer():
    client = create_app(predictor=FakePredictor()).test_client()
    assert b"Not a medical device" in client.get("/").data
