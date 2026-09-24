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
