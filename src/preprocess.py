import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# Input validation
# ----------------------------
def validate_input(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

# ----------------------------
# Load data
# ----------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ----------------------------
# Clean missing values
# ----------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_input(df)

    df = df.copy()

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

# ----------------------------
# Encode categoricals
# ----------------------------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    validate_input(df)

    df = df.copy()
    return pd.get_dummies(df)

# ----------------------------
# Split data
# ----------------------------
def split_data(df: pd.DataFrame, target_col: str = "target"):
    validate_input(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# ----------------------------
# Pipeline entrypoint
# ----------------------------
def main():
    df = load_data("data/raw/heart.csv")

    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train = encode_categoricals(X_train)
    X_test = encode_categoricals(X_test)

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
