import pandas as pd
import pytest


# ----------------------------
# Load real dataset once
# ----------------------------
def load_dataset():
    return pd.read_csv("data/raw/heart.csv")


# =========================================================
# 1. Required columns exist
# =========================================================
def test_required_columns_exist():
    df = load_dataset()

    required_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "target"
    ]

    for col in required_cols:
        assert col in df.columns


# =========================================================
# 2. Target variable contains only valid values
# =========================================================
def test_target_values_valid():
    df = load_dataset()

    unique_vals = set(df["target"].dropna().unique())

    assert unique_vals.issubset({0, 1})


# =========================================================
# 3. Numeric feature ranges are reasonable
# =========================================================
def test_numeric_feature_ranges():
    df = load_dataset()

    assert df["age"].between(0, 120).all()
    assert df["chol"].between(0, 800).all()
    assert df["trestbps"].between(0, 300).all()