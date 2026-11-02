# Model artifacts

Produced by [`notebooks/train_model.ipynb`](../notebooks/train_model.ipynb):

- `model.tflite`: the trained MobileNetV2 classifier as a float16 TFLite model, loaded by `app.py`
- `labels.json`: class names in output order
- `metrics.json`: held-out test metrics, per-class scores and the confusion matrix
- `confusion_matrix.png`, `training_curves.png`: evaluation charts used in the main README
