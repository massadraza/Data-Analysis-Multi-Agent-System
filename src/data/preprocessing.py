import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(test_size=0.2, random_state=42):
    """Load Iris dataset, scale features, and split into train/test sets."""
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": iris.feature_names,
        "target_names": target_names,
        "scaler": scaler,
    }
