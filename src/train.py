import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------------------------------
# MLflow safety config
# -------------------------------------------------
def setup_mlflow(config):
    """
    Ensures MLflow uses a CI-safe filesystem path.
    Prevents Windows-style paths like C:\ causing /C: errors in Linux CI.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        tracking_uri = "file:/tmp/mlruns"
    else:
        tracking_uri = "file:./mlruns"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

# -------------------------------------------------
# Load config
# -------------------------------------------------
def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------
# Load data
# -----------------------------
def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

# -----------------------------
# Main training loop
# -----------------------------
def main():

    config = load_config()
    setup_mlflow(config)

    X_train, X_test, y_train, y_test = load_data()

    n_estimators_list = config["model"]["n_estimators"]
    max_depth_list = config["model"]["max_depth"]
    MIN_ACCURACY = config["training"]["min_accuracy"]

    best_acc = 0.0

    for i in range(len(n_estimators_list)):

        n_estimators = n_estimators_list[i]
        max_depth = max_depth_list[i]

        with mlflow.start_run():

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config["training"]["random_state"]
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)

            best_acc = max(best_acc, acc)

            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            mlflow.sklearn.log_model(model, "model")

            print(f"Run {i+1} complete | accuracy={acc:.4f}")

    # -----------------------------
    # CI/CD gate
    # -----------------------------
    print("\n==============================")
    print(f"Best accuracy: {best_acc:.4f}")
    print("==============================")

    if best_acc < MIN_ACCURACY:
        raise SystemExit(
            f"Model failed CI check: {best_acc:.4f} < {MIN_ACCURACY}"
        )

    print("Model meets performance threshold")
    
if __name__ == "__main__":
    main()