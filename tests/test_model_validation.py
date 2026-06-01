import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/heart.csv"


@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_csv(DATA_PATH)

    # small sample for fast testing
    df = df.sample(n=120, random_state=42)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# 1. Model produces valid predictions (shape + type + classes)
# ------------------------------------------------------------
def test_model_predictions_valid(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # shape check
    assert preds.shape == (X_test.shape[0],)

    # valid class labels
    assert set(np.unique(preds)).issubset({0, 1})


# ------------------------------------------------------------
# 2. Model meets minimum performance threshold
# ------------------------------------------------------------
def test_model_minimum_accuracy(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    assert accuracy >= 0.70, f"Accuracy too low: {accuracy}"