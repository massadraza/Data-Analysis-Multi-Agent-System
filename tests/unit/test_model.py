import tempfile
import os

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.trainer import train_model, save_model
from src.models.config import MODEL_CONFIG


class TestTrainModel:
    @pytest.mark.unit
    def test_returns_random_forest(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        assert isinstance(model, RandomForestClassifier)

    @pytest.mark.unit
    def test_model_uses_config(self, trained_model):
        assert trained_model.n_estimators == MODEL_CONFIG["n_estimators"]
        assert trained_model.random_state == MODEL_CONFIG["random_state"]

    @pytest.mark.unit
    def test_prediction_shape(self, trained_model, iris_data):
        preds = trained_model.predict(iris_data["X_test"])
        assert preds.shape == (len(iris_data["X_test"]),)

    @pytest.mark.unit
    def test_prediction_values_valid(self, trained_model, iris_data):
        preds = trained_model.predict(iris_data["X_test"])
        assert set(np.unique(preds)).issubset({0, 1, 2})

    @pytest.mark.unit
    def test_predict_proba_shape(self, trained_model, iris_data):
        proba = trained_model.predict_proba(iris_data["X_test"])
        assert proba.shape == (len(iris_data["X_test"]), 3)

    @pytest.mark.unit
    def test_predict_proba_sums_to_one(self, trained_model, iris_data):
        proba = trained_model.predict_proba(iris_data["X_test"])
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestSaveModel:
    @pytest.mark.unit
    def test_save_creates_file(self, trained_model, tmp_path):
        path = str(tmp_path / "model.pkl")
        result = save_model(trained_model, path)
        assert os.path.exists(result)

    @pytest.mark.unit
    def test_serialization_roundtrip(self, trained_model, iris_data, tmp_path):
        path = str(tmp_path / "model.pkl")
        save_model(trained_model, path)
        loaded = joblib.load(path)
        original_preds = trained_model.predict(iris_data["X_test"])
        loaded_preds = loaded.predict(iris_data["X_test"])
        np.testing.assert_array_equal(original_preds, loaded_preds)
