"""
Forecasting Model — Stage 2
----------------------------
Trains a model to predict AC_POWER from weather + time-of-day features.
This is the "optimization" brain: once you can forecast output an hour or a
day ahead, you can schedule loads/battery charging around it.

Uses RandomForestRegressor (sklearn) so it runs anywhere with no extra
dependencies. Swap in XGBoost/LightGBM later for a small accuracy bump if
you have them installed — the feature set stays the same.

Run:
    python3 src/train_forecast_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_pipeline import build_dataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURES = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR",
    "MINUTE",
    "DAY_OF_WEEK",
    "MONTH",
]
TARGET = "AC_POWER"


def train():
    df = build_dataset()
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Time-ordered split (not random) — forecasting must be validated on
    # FUTURE data the model hasn't seen, not a random shuffle
    df_sorted = df.sort_values("DATE_TIME")
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Test MAE:  {mae:.2f} kW")
    print(f"Test R^2:  {r2:.4f}")

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "forecast_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved model -> {model_path}")

    return model, mae, r2


if __name__ == "__main__":
    train()
