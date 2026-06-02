import os
import sys
import numpy as np
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# -----------------------------
# Config
# -----------------------------
DRIFT_THRESHOLD = 0.30
REPORT_DIR = "reports"

# -----------------------------
# Load reference data
# -----------------------------
def load_reference():
    return pd.read_csv("data/X_train.csv")

# -----------------------------
# Simulate production data (drifted)
# -----------------------------
def load_production(reference: pd.DataFrame):
    prod = reference.copy()

    for col in prod.columns:
        if pd.api.types.is_numeric_dtype(prod[col]):
            prod[col] = prod[col] * 1.05 + np.random.normal(
                0, 0.1, len(prod)
            )
        else:
            prod[col] = prod[col].sample(frac=1).reset_index(drop=True)

    return prod

# -----------------------------
# Run Evidently report
# -----------------------------
def run_drift(reference, production):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=production)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "drift_report.html")
    report.save_html(report_path)

    return report, report_path

# -----------------------------
# Extract drift score safely
# -----------------------------
def extract_drift_score(report):
    result = report.as_dict()

    try:
        return result["metrics"][0]["result"]["dataset_drift"]
    except Exception:
        return None

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading reference data...")
    reference = load_reference()

    print("Generating production data...")
    production = load_production(reference)

    print("Running drift detection...")
    report, report_path = run_drift(reference, production)

    drift_score = extract_drift_score(report)

    print("\n==============================")
    print(f"Drift score: {drift_score}")
    print("==============================")
    print(f"Report saved: {report_path}")

    if drift_score is None:
        print("Could not compute drift score")
        sys.exit(1)

    if drift_score > DRIFT_THRESHOLD:
        print("❌ Drift threshold exceeded")
        sys.exit(1)

    print("✅ Drift within threshold")
    sys.exit(0)

if __name__ == "__main__":
    main()
