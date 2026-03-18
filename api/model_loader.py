import os
import joblib

_model = None
_model_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pkl")

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(_model_path)
    return _model
