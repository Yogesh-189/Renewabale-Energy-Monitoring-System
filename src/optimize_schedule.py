"""
Optimization Engine — Stage 5
--------------------------------
This is the "Optimization" half of the project title. It takes the
forecast model's output and turns it into an actionable schedule:
- WHEN to run flexible/heavy loads (pumps, EV charging, appliances)
- WHEN to charge a battery (if present) vs draw from the grid
- A simple self-consumption score: how much of the predicted solar output
  would be wasted if nothing is scheduled around it

This is intentionally rule-based (not another ML model) — optimization
logic should be transparent and explainable, especially for something
that controls real loads. The forecast model supplies the numbers;
this module supplies the decision logic.

Run:
    python3 src/optimize_schedule.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta

from data_pipeline import build_dataset
from train_forecast_model import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "forecast_model.joblib")


def get_forecast_curve(df: pd.DataFrame, model, hours_ahead: int = 24) -> pd.DataFrame:
    """Builds an hour-by-hour predicted TOTAL plant output curve for the
    next N hours, using historical same-hour averages as the weather
    stand-in (same approach as the API's /forecast endpoint)."""
    last_ts = df["DATE_TIME"].max()
    hourly_avg = df.groupby("HOUR")[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]].mean()
    n_inverters = df["SOURCE_KEY"].nunique()

    future_ts = [last_ts + timedelta(minutes=15 * (i + 1)) for i in range(hours_ahead * 4)]
    rows = []
    for ts in future_ts:
        avg = hourly_avg.loc[ts.hour]
        feat = pd.DataFrame([{
            "IRRADIATION": avg["IRRADIATION"],
            "AMBIENT_TEMPERATURE": avg["AMBIENT_TEMPERATURE"],
            "MODULE_TEMPERATURE": avg["MODULE_TEMPERATURE"],
            "HOUR": ts.hour,
            "MINUTE": ts.minute,
            "DAY_OF_WEEK": ts.dayofweek,
            "MONTH": ts.month,
        }])[FEATURES]
        pred_per_inverter = model.predict(feat)[0]
        rows.append({
            "DATE_TIME": ts,
            "PREDICTED_TOTAL_AC_POWER_KW": round(float(pred_per_inverter) * n_inverters / 1000, 2),
        })
    return pd.DataFrame(rows)


def recommend_schedule(
    forecast_df: pd.DataFrame,
    flexible_loads: list,
    battery_capacity_kwh: float = 50.0,
    battery_charge_rate_kw: float = 10.0,
) -> dict:
    """
    flexible_loads: list of dicts like
        {"name": "EV Charging", "power_kw": 7, "duration_hours": 3}
    Greedy strategy: place each load's required window at the highest
    contiguous predicted-output period still available (peak shaving /
    maximum self-consumption).
    """
    curve = forecast_df.copy().sort_values("DATE_TIME").reset_index(drop=True)
    curve["remaining_kw"] = curve["PREDICTED_TOTAL_AC_POWER_KW"]

    schedule = []
    for load in flexible_loads:
        steps_needed = int(load["duration_hours"] * 4)  # 15-min steps
        if steps_needed > len(curve):
            steps_needed = len(curve)

        # Find the contiguous window with the highest average predicted output
        best_start, best_avg = 0, -1
        for start in range(0, len(curve) - steps_needed + 1):
            window = curve["remaining_kw"].iloc[start:start + steps_needed]
            avg = window.mean()
            if avg > best_avg:
                best_avg = avg
                best_start = start

        window_slice = curve.iloc[best_start:best_start + steps_needed]
        start_time = window_slice["DATE_TIME"].iloc[0]
        end_time = window_slice["DATE_TIME"].iloc[-1] + timedelta(minutes=15)
        solar_covered_pct = min(100, round((best_avg / load["power_kw"]) * 100, 1)) if load["power_kw"] > 0 else 0

        schedule.append({
            "load": load["name"],
            "recommended_start": start_time.strftime("%Y-%m-%d %H:%M"),
            "recommended_end": end_time.strftime("%Y-%m-%d %H:%M"),
            "avg_predicted_solar_available_kw": round(best_avg, 2),
            "load_power_kw": load["power_kw"],
            "estimated_solar_coverage_pct": solar_covered_pct,
        })

        # Reduce remaining available capacity so the next load doesn't
        # double-book the same solar output
        curve.loc[best_start:best_start + steps_needed - 1, "remaining_kw"] = (
            curve.loc[best_start:best_start + steps_needed - 1, "remaining_kw"] - load["power_kw"]
        ).clip(lower=0)

    # Battery charging recommendation: charge during the single highest
    # predicted-output block that isn't already claimed by a flexible load
    charge_steps = int(np.ceil(battery_capacity_kwh / (battery_charge_rate_kw * 0.25)))
    charge_steps = min(charge_steps, len(curve))
    best_start, best_avg = 0, -1
    for start in range(0, len(curve) - charge_steps + 1):
        window = curve["remaining_kw"].iloc[start:start + charge_steps]
        avg = window.mean()
        if avg > best_avg:
            best_avg = avg
            best_start = start
    charge_window = curve.iloc[best_start:best_start + charge_steps]
    battery_plan = {
        "recommended_start": charge_window["DATE_TIME"].iloc[0].strftime("%Y-%m-%d %H:%M"),
        "recommended_end": (charge_window["DATE_TIME"].iloc[-1] + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M"),
        "avg_predicted_solar_available_kw": round(best_avg, 2),
    }

    total_predicted_kwh = round(curve["PREDICTED_TOTAL_AC_POWER_KW"].sum() * 0.25, 1)

    return {
        "total_predicted_generation_kwh_next_24h": total_predicted_kwh,
        "flexible_load_schedule": schedule,
        "battery_charge_plan": battery_plan,
    }


if __name__ == "__main__":
    df = build_dataset()
    model = joblib.load(MODEL_PATH)
    curve = get_forecast_curve(df, model, hours_ahead=24)

    # Example flexible loads — customize these for your demo
    loads = [
        {"name": "EV Charging", "power_kw": 7, "duration_hours": 3},
        {"name": "Water Pump", "power_kw": 2, "duration_hours": 2},
        {"name": "Washing Machine", "power_kw": 1.5, "duration_hours": 1},
    ]

    plan = recommend_schedule(curve, loads, battery_capacity_kwh=50, battery_charge_rate_kw=10)

    print(f"Predicted generation, next 24h: {plan['total_predicted_generation_kwh_next_24h']} kWh\n")
    print("Recommended load schedule (maximizes solar self-consumption):")
    for item in plan["flexible_load_schedule"]:
        print(f"  {item['load']:20s} -> {item['recommended_start']} to {item['recommended_end']}  "
              f"(solar avail: {item['avg_predicted_solar_available_kw']} kW, "
              f"coverage: {item['estimated_solar_coverage_pct']}%)")

    print(f"\nRecommended battery charge window: {plan['battery_charge_plan']['recommended_start']} "
          f"to {plan['battery_charge_plan']['recommended_end']} "
          f"(avg {plan['battery_charge_plan']['avg_predicted_solar_available_kw']} kW available)")
