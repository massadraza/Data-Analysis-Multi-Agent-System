import json
import os
import tempfile

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data.preprocessing import load_and_preprocess
from src.models.trainer import train_model


@pytest.fixture(scope="session")
def iris_data():
    """Load and preprocess Iris data once for the entire test session."""
    return load_and_preprocess()


@pytest.fixture(scope="session")
def trained_model(iris_data):
    """Train a model once for the entire test session."""
    return train_model(iris_data["X_train"], iris_data["y_train"])


@pytest.fixture
def sample_features():
    """Sample valid Iris features (4 floats)."""
    return [5.1, 3.5, 1.4, 0.2]


@pytest.fixture
def mock_model_artifact(trained_model, tmp_path):
    """Save a trained model to a temporary path and return the path."""
    path = tmp_path / "model.pkl"
    joblib.dump(trained_model, path)
    return str(path)


@pytest.fixture
def mock_metrics_file(tmp_path):
    """Create a temporary metrics JSON file."""
    metrics = {
        "accuracy": 0.95,
        "classification_report": {"setosa": {"precision": 1.0}},
        "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]],
    }
    path = tmp_path / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f)
    return str(path)


@pytest.fixture
def test_client(trained_model, mock_metrics_file, monkeypatch):
    """FastAPI TestClient with a real trained model and mock metrics."""
    import api.model_loader as ml

    monkeypatch.setattr(ml, "_model", trained_model)
    monkeypatch.setattr("api.main.METRICS_PATH", mock_metrics_file)

    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)
