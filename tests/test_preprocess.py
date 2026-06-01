import pandas as pd
import numpy as np
import pytest

from src.preprocess import clean_data, encode_categoricals, validate_input, split_data


def test_clean_data_handles_missing_values():
    df = pd.DataFrame({"a": [1, np.nan, 3], "target": [0, 1, 0]})
    cleaned = clean_data(df)
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_median_imputation():
    df = pd.DataFrame({"a": [1, np.nan, 3], "target": [0, 1, 0]})
    cleaned = clean_data(df)
    assert cleaned["a"].iloc[1] == 2


def test_encode_creates_dummies():
    df = pd.DataFrame({"color": ["red", "blue"], "target": [0, 1]})
    encoded = encode_categoricals(df)
    assert "color_red" in encoded.columns


def test_encode_does_not_modify_original():
    df = pd.DataFrame({"color": ["red", "blue"], "target": [0, 1]})
    original = df.copy()
    encode_categoricals(df)
    assert df.equals(original)


def test_validate_input_rejects_invalid():
    with pytest.raises(TypeError):
        validate_input([1, 2, 3])


def test_split_data_shapes():
    df = pd.DataFrame({
        "f1": range(10),
        "target": [0, 1] * 5
    })

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df)