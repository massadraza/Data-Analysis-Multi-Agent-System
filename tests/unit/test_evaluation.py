import json
import os

import pytest

from src.evaluation.metrics import evaluate_model, save_metrics


class TestEvaluateModel:
    @pytest.mark.unit
    def test_returns_expected_keys(self, trained_model, iris_data):
        result = evaluate_model(
            trained_model, iris_data["X_test"], iris_data["y_test"], iris_data["target_names"]
        )
        assert set(result.keys()) == {"accuracy", "classification_report", "confusion_matrix"}

    @pytest.mark.unit
    def test_accuracy_is_float_between_0_and_1(self, trained_model, iris_data):
        result = evaluate_model(
            trained_model, iris_data["X_test"], iris_data["y_test"], iris_data["target_names"]
        )
        assert isinstance(result["accuracy"], float)
        assert 0.0 <= result["accuracy"] <= 1.0

    @pytest.mark.unit
    def test_confusion_matrix_shape(self, trained_model, iris_data):
        result = evaluate_model(
            trained_model, iris_data["X_test"], iris_data["y_test"], iris_data["target_names"]
        )
        cm = result["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    @pytest.mark.unit
    def test_classification_report_is_dict(self, trained_model, iris_data):
        result = evaluate_model(
            trained_model, iris_data["X_test"], iris_data["y_test"], iris_data["target_names"]
        )
        assert isinstance(result["classification_report"], dict)


class TestSaveMetrics:
    @pytest.mark.unit
    def test_save_creates_json(self, tmp_path):
        metrics = {"accuracy": 0.95}
        path = str(tmp_path / "metrics.json")
        save_metrics(metrics, path)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == metrics
