from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four Iris features: sepal_length, sepal_width, petal_length, petal_width",
    )


class PredictResponse(BaseModel):
    prediction: str
    probability: list[float]


class HealthResponse(BaseModel):
    status: str
