import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import pandas as pd

def main():

    mlflow.set_tracking_uri("file:./mlruns")

    experiment_name = "heart-disease-experiment"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print("No experiment found. Run training first.")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        print("No runs found. Did you run training yet?")
        return

    cols = [
        "run_id",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "params.n_estimators",
        "params.max_depth"
    ]

    cols = [c for c in cols if c in runs.columns]
    runs = runs[cols]

    if "metrics.accuracy" not in runs.columns:
        print("No accuracy metric found.")
        return

    runs = runs.sort_values(by="metrics.accuracy", ascending=False)

    best = runs.iloc[0]

    print("\n BEST MODEL")
    print("=" * 40)
    print(best)

    print("\n ALL RUNS (ranked)")
    print("=" * 40)
    print(runs)

    print("\n Best Run ID:", best["run_id"])

if __name__ == "__main__":
    main()
