import numpy as np
import pytest

from src.data.preprocessing import load_and_preprocess


class TestLoadAndPreprocess:
    @pytest.mark.unit
    def test_returns_expected_keys(self, iris_data):
        expected = {"X_train", "X_test", "y_train", "y_test", "feature_names", "target_names", "scaler"}
        assert set(iris_data.keys()) == expected

    @pytest.mark.unit
    def test_feature_shape(self, iris_data):
        assert iris_data["X_train"].shape[1] == 4
        assert iris_data["X_test"].shape[1] == 4

    @pytest.mark.unit
    def test_default_split_ratio(self, iris_data):
        total = len(iris_data["X_train"]) + len(iris_data["X_test"])
        assert total == 150
        assert len(iris_data["X_test"]) == 30  # 20% of 150

    @pytest.mark.unit
    def test_custom_split_ratio(self):
        data = load_and_preprocess(test_size=0.3)
        assert len(data["X_test"]) == 45  # 30% of 150

    @pytest.mark.unit
    def test_features_are_scaled(self, iris_data):
        # Scaled features should have mean close to 0 for training set
        means = iris_data["X_train"].mean(axis=0)
        assert np.allclose(means, 0, atol=0.1)

    @pytest.mark.unit
    def test_target_names(self, iris_data):
        assert iris_data["target_names"] == ["setosa", "versicolor", "virginica"]

    @pytest.mark.unit
    def test_labels_are_valid(self, iris_data):
        assert set(np.unique(iris_data["y_train"])).issubset({0, 1, 2})
        assert set(np.unique(iris_data["y_test"])).issubset({0, 1, 2})
