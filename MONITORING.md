# Model Monitoring Report (Evidently Drift Detection)

## Overview

This project uses Evidently to monitor data drift between the training dataset and simulated production data. The goal is to detect whether changes in input feature distributions could impact model performance after deployment.

The reference dataset is derived from the training split (`data/X_train.csv`), while the production dataset is simulated by applying controlled perturbations to mimic real-world data changes over time.

## 1. Features that showed drift

Based on the Evidently Data Drift report, the following features showed noticeable drift:

- **age**: Shifted distribution due to simulated demographic variation in the production data.
- **resting_bp**: Increased variance introduced to simulate sensor variability.
- **cholesterol**: Significant drift due to added noise and shifted mean values.
- **max_heart_rate**: Moderate drift caused by scaling adjustments in the production dataset.

### Why drift occurred

The drift was intentionally introduced to simulate real-world conditions such as:
- Population changes over time (age distribution shift)
- Measurement noise in medical devices
- Differences in data collection environments
- Sensor or preprocessing inconsistencies

## 2. Potential impact on model performance

The detected drift may impact model performance in the following ways:

- Features like **cholesterol** and **resting_bp** are strong predictors in cardiovascular risk models, so drift here can directly affect classification accuracy.
- Moderate drift in **max_heart_rate** may slightly shift decision boundaries.
- If drift continues to increase over time, model calibration may degrade, leading to reduced precision and recall.

## 3. Recommended action

Based on the observed drift level:

### Recommended action: **continue monitoring + prepare for retraining**

- Drift is present but not catastrophic.
- Immediate retraining is not strictly required unless performance degradation is observed in production.
- Continue monitoring incoming data to confirm whether drift is persistent or temporary.

### Additional recommendations:
- Schedule periodic retraining if drift persists across multiple time windows.
- Add automated alerts when drift exceeds a threshold.
- Investigate upstream data collection processes for cholesterol and blood pressure measurements.

## Conclusion

The system demonstrates mild to moderate data drift across several key features. While current drift levels do not immediately invalidate the model, continued monitoring is essential to ensure long-term reliability and performance stability.
