import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from .config import MODEL_CONFIG


def train_model(X_train, y_train):
    """Train a RandomForestClassifier with the configured hyperparameters."""
    model = RandomForestClassifier(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    return model


def save_model(model, path="artifacts/model.pkl"):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path
