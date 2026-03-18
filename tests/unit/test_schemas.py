import pytest
from pydantic import ValidationError

from api.schemas import PredictRequest, PredictResponse, HealthResponse


class TestPredictRequest:
    @pytest.mark.unit
    def test_valid_request(self):
        req = PredictRequest(features=[5.1, 3.5, 1.4, 0.2])
        assert req.features == [5.1, 3.5, 1.4, 0.2]

    @pytest.mark.unit
    def test_rejects_too_few_features(self):
        with pytest.raises(ValidationError):
            PredictRequest(features=[1.0, 2.0])

    @pytest.mark.unit
    def test_rejects_too_many_features(self):
        with pytest.raises(ValidationError):
            PredictRequest(features=[1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.mark.unit
    def test_rejects_missing_features(self):
        with pytest.raises(ValidationError):
            PredictRequest()

    @pytest.mark.unit
    def test_rejects_non_numeric(self):
        with pytest.raises(ValidationError):
            PredictRequest(features=["a", "b", "c", "d"])


class TestPredictResponse:
    @pytest.mark.unit
    def test_valid_response(self):
        resp = PredictResponse(prediction="setosa", probability=[0.9, 0.05, 0.05])
        assert resp.prediction == "setosa"


class TestHealthResponse:
    @pytest.mark.unit
    def test_valid_health(self):
        resp = HealthResponse(status="ok")
        assert resp.status == "ok"
