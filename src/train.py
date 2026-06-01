import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

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

    # IMPORTANT: single tracking backend
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("heart-disease-experiment")

    X_train, X_test, y_train, y_test = load_data()

    n_estimators_list = [50, 100, 200, 300, 400]
    max_depth_list = [3, 5, 7, 10, None]

    for i in range(5):

        n_estimators = n_estimators_list[i]
        max_depth = max_depth_list[i]

        with mlflow.start_run():

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)

            # log params
            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            # log model
            mlflow.sklearn.log_model(model, "model")

            print(f"Run {i+1} complete")


if __name__ == "__main__":
    main()