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


def detect_anomalies(df: pd.DataFrame = None, model=None) -> pd.DataFrame:
    if df is None:
        df = build_dataset()

    # Callers that already hold a loaded model (the API caches one) can pass
    # it in; loading a ~25 MB joblib file on every call is needlessly slow.
    if model is None:
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


def group_incidents(alerts: pd.DataFrame, dataset_end, recent_hours: int = 24,
                    interval_minutes: int = 15) -> dict:
    """
    Collapses per-reading alerts into INCIDENTS.

    `alerts` has one row per confirmed 15-minute reading, so a single 3-hour
    outage on one inverter shows up as ~12 "alerts". For an operator (and a
    reviewer) the meaningful unit is the incident: one inverter, one
    continuous period of underperformance.

    Two readings belong to the same incident if they are at most two
    intervals (30 min) apart. An incident's start is backdated by one
    interval, because a reading only becomes a *confirmed* alert on the
    2nd consecutive bad reading (MIN_CONSECUTIVE = 2).

    An incident is "recent" if it ended within `recent_hours` of the last
    timestamp in the dataset - that is what separates a fault that needs
    attention now from one that is simply part of the history.
    """
    dataset_end = pd.Timestamp(dataset_end)
    empty = {
        "dataset_end": str(dataset_end), "recent_window_hours": recent_hours,
        "summary": {"alert_readings": 0, "incidents": 0, "inverters_affected": 0,
                    "inverters_recent": 0},
        "incidents": [],
    }
    if alerts is None or len(alerts) == 0:
        return empty

    a = alerts.sort_values(["SOURCE_KEY", "DATE_TIME"]).copy()
    a["DATE_TIME"] = pd.to_datetime(a["DATE_TIME"])
    new_incident = (
        a.groupby("SOURCE_KEY")["DATE_TIME"].diff() > pd.Timedelta(minutes=2 * interval_minutes)
    ) | a["SOURCE_KEY"].ne(a["SOURCE_KEY"].shift())
    a["INCIDENT_ID"] = new_incident.cumsum()

    incidents = []
    for _, g in a.groupby("INCIDENT_ID"):
        start = g["DATE_TIME"].iloc[0] - pd.Timedelta(minutes=interval_minutes)
        end = g["DATE_TIME"].iloc[-1]
        duration = int((end - start).total_seconds() // 60) + interval_minutes
        mean_ratio = float(g["PERFORMANCE_RATIO"].mean())
        incidents.append({
            "SOURCE_KEY": g["SOURCE_KEY"].iloc[0],
            "START": str(start),
            "END": str(end),
            "DURATION_MIN": duration,
            "CONFIRMED_READINGS": int(len(g)),
            "MEAN_ACTUAL": round(float(g["AC_POWER"].mean()), 1),
            "MEAN_EXPECTED": round(float(g["PREDICTED_AC_POWER"].mean()), 1),
            "MEAN_RATIO": round(float(g["PERFORMANCE_RATIO"].mean()), 3),
            "MIN_RATIO": round(float(g["PERFORMANCE_RATIO"].min()), 3),
            # An incident averaging under 10% of forecast is an OUTAGE (the inverter
            # produced essentially nothing), even if a few edge readings show a
            # little output. Judging by the maximum reading wrongly downgraded a
            # 3.5-hour near-zero event to "underperformance - check soiling".
            "SEVERITY": "TOTAL_OUTAGE" if mean_ratio < 0.10 else "UNDERPERFORMANCE",
            "IS_RECENT": bool(end >= dataset_end - pd.Timedelta(hours=recent_hours)),
        })
    incidents.sort(key=lambda i: i["END"], reverse=True)

    return {
        "dataset_end": str(dataset_end),
        "recent_window_hours": recent_hours,
        "summary": {
            "alert_readings": int(len(a)),
            "incidents": len(incidents),
            "inverters_affected": len({i["SOURCE_KEY"] for i in incidents}),
            "inverters_recent": len({i["SOURCE_KEY"] for i in incidents if i["IS_RECENT"]}),
        },
        "incidents": incidents,
    }


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

        report = group_incidents(alerts, build_dataset()["DATE_TIME"].max())
        print("\nGrouped into incidents:", report["summary"])
        for inc in report["incidents"][:8]:
            print(f"  {inc['SOURCE_KEY']}  {inc['START']} -> {inc['END']}  "
                  f"{inc['DURATION_MIN']} min  {inc['SEVERITY']}")
