import pytest


class TestHealthEndpoint:
    @pytest.mark.integration
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestPredictEndpoint:
    @pytest.mark.integration
    def test_predict_returns_200(self, test_client, sample_features):
        response = test_client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data

    @pytest.mark.integration
    def test_predict_returns_valid_class(self, test_client, sample_features):
        response = test_client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert data["prediction"] in ["setosa", "versicolor", "virginica"]

    @pytest.mark.integration
    def test_predict_probability_sums_to_one(self, test_client, sample_features):
        response = test_client.post("/predict", json={"features": sample_features})
        probs = response.json()["probability"]
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6

    @pytest.mark.integration
    def test_predict_invalid_features_returns_422(self, test_client):
        response = test_client.post("/predict", json={"features": [1.0, 2.0]})
        assert response.status_code == 422

    @pytest.mark.integration
    def test_predict_missing_body_returns_422(self, test_client):
        response = test_client.post("/predict", json={})
        assert response.status_code == 422


class TestMetricsEndpoint:
    @pytest.mark.integration
    def test_metrics_returns_200(self, test_client):
        response = test_client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data
        assert "confusion_matrix" in data

    @pytest.mark.integration
    def test_metrics_not_found(self, trained_model, monkeypatch):
        """Test 404 when metrics file doesn't exist."""
        import api.model_loader as ml

        monkeypatch.setattr(ml, "_model", trained_model)
        monkeypatch.setattr("api.main.METRICS_PATH", "/nonexistent/metrics.json")

        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 404
