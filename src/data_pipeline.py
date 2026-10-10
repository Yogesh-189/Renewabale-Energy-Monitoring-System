"""
Data Pipeline — Stage 1
------------------------
Loads the generation + weather CSVs (Kaggle "Solar Power Generation Data"
schema), merges them on DATE_TIME + PLANT_ID, cleans, and engineers features
used by the forecasting and anomaly-detection models.

Run directly to sanity-check the pipeline:
    python3 src/data_pipeline.py
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_raw(plant_prefix="Plant_1"):
    gen_path = os.path.join(DATA_DIR, f"{plant_prefix}_Generation_Data.csv")
    weather_path = os.path.join(DATA_DIR, f"{plant_prefix}_Weather_Sensor_Data.csv")

    gen = pd.read_csv(gen_path, parse_dates=["DATE_TIME"])
    weather = pd.read_csv(weather_path, parse_dates=["DATE_TIME"])
    return gen, weather


def merge_and_clean(gen: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    # Weather is one row per timestamp per plant (not per inverter) — merge
    # it onto every inverter's row for that timestamp.
    weather_slim = weather[
        ["DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    ]
    df = gen.merge(weather_slim, on=["DATE_TIME", "PLANT_ID"], how="left")

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Fill small weather gaps by interpolation (sensor dropouts happen a lot
    # in the real Kaggle dataset)
    for col in ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]:
        df[col] = df.groupby("SOURCE_KEY")[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    # Negative power readings are sensor noise, not real — clip to 0
    df["DC_POWER"] = df["DC_POWER"].clip(lower=0)
    df["AC_POWER"] = df["AC_POWER"].clip(lower=0)

    return df


def calibrate_known_dataset_quirks(df: pd.DataFrame, plant_prefix: str) -> pd.DataFrame:
    """
    Plant 1 of this specific Kaggle dataset ("Solar Power Generation Data")
    has a widely-documented quirk: its raw DC_POWER readings are reported
    at roughly 10x the scale of AC_POWER — almost certainly a unit/sensor
    calibration artifact from the original data collection, not a real
    ~10% conversion efficiency (real inverters run ~95-98%). Multiple
    public analyses of this exact dataset note the same scale mismatch.
    Without this correction, CONVERSION_EFFICIENCY computes to a
    physically implausible ~9-10% for every inverter.

    Rather than hardcoding this by filename (fragile — it would wrongly
    "fix" data that was never broken, e.g. the synthetic sample generator),
    we detect the quirk empirically: if the raw median AC/DC ratio across
    meaningful daylight readings is implausibly low, we calibrate; a
    healthy dataset (≈0.95-0.98 ratio) is left untouched.

    Only affects the DC_POWER column used for display/efficiency — not
    AC_POWER (the model's actual prediction target), so the trained
    forecasting model is unaffected either way.
    """
    df = df.copy()
    daylight = df[df["DC_POWER"] > 50]
    if len(daylight) == 0:
        return df
    raw_ratio = (daylight["AC_POWER"] / daylight["DC_POWER"]).median()

    if raw_ratio < 0.5:
        # Implausible for a real inverter — this is the known scale quirk
        print(f"[data_pipeline] Detected DC/AC scale quirk (median raw ratio "
              f"{raw_ratio:.3f}) — calibrating DC_POWER by /10 for realistic "
              f"conversion-efficiency figures.")
        df["DC_POWER"] = df["DC_POWER"] / 10.0

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["MINUTE"] = df["DATE_TIME"].dt.minute
    df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek
    df["MONTH"] = df["DATE_TIME"].dt.month

    # Inverter conversion efficiency (AC vs DC) — key health signal
    df["CONVERSION_EFFICIENCY"] = np.where(
        df["DC_POWER"] > 0, df["AC_POWER"] / df["DC_POWER"], np.nan
    )

    # Expected output given irradiation (simple physical baseline) — the gap
    # between this and actual AC_POWER feeds the anomaly detector later
    df["EXPECTED_AC_POWER"] = df["IRRADIATION"] * df["MODULE_TEMPERATURE"].clip(lower=0) * 25

    return df


def build_dataset(plant_prefix="Plant_1") -> pd.DataFrame:
    gen, weather = load_raw(plant_prefix)
    df = merge_and_clean(gen, weather)
    df = calibrate_known_dataset_quirks(df, plant_prefix)
    df = add_features(df)
    return df.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)


if __name__ == "__main__":
    df = build_dataset()
    print(df.shape)
    print(df.head())
    print("\nNulls per column:")
    print(df.isna().sum())
    out_path = os.path.join(DATA_DIR, "processed_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved processed dataset -> {out_path}")
