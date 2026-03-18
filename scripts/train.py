#!/usr/bin/env python3
"""End-to-end training script for the Iris classification pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import load_and_preprocess
from src.models.trainer import train_model, save_model
from src.evaluation.metrics import evaluate_model, save_metrics


def main():
    print("Loading and preprocessing data...")
    data = load_and_preprocess()

    print("Training model...")
    model = train_model(data["X_train"], data["y_train"])

    print("Evaluating model...")
    metrics = evaluate_model(model, data["X_test"], data["y_test"], data["target_names"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    model_path = save_model(model, "artifacts/model.pkl")
    print(f"Model saved to {model_path}")

    metrics_path = save_metrics(metrics, "artifacts/metrics.json")
    print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    main()
