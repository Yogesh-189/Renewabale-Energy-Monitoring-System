"""
Anomaly Detection — Stage 3
-----------------------------
Flags underperforming inverters by comparing ACTUAL output against what the
forecast model EXPECTED given the weather at that moment. A large negative
gap, sustained over several readings, means a real fault (soiling, shading,
wiring fault, inverter failure) rather than just normal weather variance.

This is the "automation" piece: this script's output feeds directly into
alerting (Stage 4 / the API layer).

Run:
    python3 src/anomaly_detection.py
"""

import os
import joblib
import pandas as pd
import numpy as np

from data_pipeline import build_dataset
from train_forecast_model import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "forecast_model.joblib")

# How far below the forecast (as a %) counts as underperformance
UNDERPERFORM_THRESHOLD = 0.35
# How many consecutive underperforming readings before it's a real alert
# (filters out single-reading sensor noise)
MIN_CONSECUTIVE = 2


def detect_anomalies(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = build_dataset()

    model = joblib.load(MODEL_PATH)

    df = df.dropna(subset=FEATURES).copy()
    df["PREDICTED_AC_POWER"] = model.predict(df[FEATURES])

    # Only evaluate daylight readings — nighttime zero-output is normal, not
    # a fault, and would otherwise swamp the results with false positives
    daylight = df[df["IRRADIATION"] > 0.05].copy()

    daylight["PERFORMANCE_RATIO"] = np.where(
        daylight["PREDICTED_AC_POWER"] > 1,
        daylight["AC_POWER"] / daylight["PREDICTED_AC_POWER"],
        1.0,
    )
    daylight["IS_UNDERPERFORMING"] = daylight["PERFORMANCE_RATIO"] < (1 - UNDERPERFORM_THRESHOLD)

    # Find sustained faults per inverter (consecutive underperforming rows)
    daylight = daylight.sort_values(["SOURCE_KEY", "DATE_TIME"])
    daylight["FAULT_STREAK"] = (
        daylight.groupby("SOURCE_KEY")["IS_UNDERPERFORMING"]
        .apply(lambda s: s.groupby((~s).cumsum()).cumsum())
        .reset_index(level=0, drop=True)
    )
    daylight["CONFIRMED_ALERT"] = daylight["FAULT_STREAK"] >= MIN_CONSECUTIVE

    alerts = daylight[daylight["CONFIRMED_ALERT"]][
        ["DATE_TIME", "SOURCE_KEY", "AC_POWER", "PREDICTED_AC_POWER", "PERFORMANCE_RATIO"]
    ]
    return daylight, alerts


if __name__ == "__main__":
    daylight, alerts = detect_anomalies()
    print(f"Daylight readings evaluated: {len(daylight)}")
    print(f"Confirmed underperformance alerts: {len(alerts)}")
    if len(alerts):
        print("\nSample alerts:")
        print(alerts.head(10).to_string(index=False))

        summary = alerts.groupby("SOURCE_KEY").size().sort_values(ascending=False)
        print("\nAlerts by inverter:")
        print(summary)
