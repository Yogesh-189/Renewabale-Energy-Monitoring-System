"""
Backend API — Stage 4
-----------------------
FastAPI service exposing the monitoring/optimization system to a dashboard
or any client.

NOTE: FastAPI isn't installed in this sandbox (no network access here to
pip install it), so this file is provided ready-to-run in your own
environment — it hasn't been executed in this session. To run it locally:

    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs.

Endpoints:
    GET  /health                       -> liveness check
    GET  /generation/latest            -> most recent readings per inverter
    GET  /forecast?hours=6             -> forecast for the next N hours
    GET  /anomalies                    -> current confirmed fault alerts
    GET  /performance/summary          -> daily yield + efficiency per inverter
"""

import os
import joblib
import pandas as pd
import numpy as np
import webbrowser
import threading
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_pipeline import build_dataset
from train_forecast_model import FEATURES
from anomaly_detection import detect_anomalies
from optimize_schedule import get_forecast_curve, recommend_schedule

app = FastAPI(title="Solar Monitoring & Optimization API", version="0.1.0")

# Allow the dashboard (running on a different port/origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "forecast_model.joblib")

_cache = {"df": None, "model": None}


def get_data():
    if _cache["df"] is None:
        _cache["df"] = build_dataset()
    return _cache["df"]


def get_model():
    if _cache["model"] is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model not trained yet — run train_forecast_model.py first")
        _cache["model"] = joblib.load(MODEL_PATH)
    return _cache["model"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def open_docs_on_start():
    # Auto-opens the API docs in your default browser 1.5s after the
    # server starts, so you don't have to type the URL every time.
    def _open():
        webbrowser.open("http://127.0.0.1:8000/docs")
    threading.Timer(1.5, _open).start()


@app.get("/generation/latest")
def latest_generation():
    df = get_data()
    latest_ts = df["DATE_TIME"].max()
    latest = df[df["DATE_TIME"] == latest_ts]
    return latest[["SOURCE_KEY", "DATE_TIME", "DC_POWER", "AC_POWER", "DAILY_YIELD"]].to_dict(orient="records")


@app.get("/weather/latest")
def weather_latest():
    """Latest ambient temperature, module temperature and irradiation
    reading — used by the dashboard's environmental condition cards."""
    df = get_data()
    latest_ts = df["DATE_TIME"].max()
    row = df[df["DATE_TIME"] == latest_ts].iloc[0]
    return {
        "DATE_TIME": str(latest_ts),
        "AMBIENT_TEMPERATURE": round(float(row["AMBIENT_TEMPERATURE"]), 2),
        "MODULE_TEMPERATURE": round(float(row["MODULE_TEMPERATURE"]), 2),
        "IRRADIATION": round(float(row["IRRADIATION"]), 4),
    }


@app.get("/forecast")
def forecast(hours: int = 6):
    """
    Naive-but-honest forward forecast: builds future timestamps, estimates
    irradiation from the historical same-time-of-day average (since we don't
    have a live weather feed here), and predicts AC power for each inverter.
    Swap the irradiation estimate for a live weather API call in production.
    """
    df = get_data()
    model = get_model()

    last_ts = df["DATE_TIME"].max()

    # historical average irradiation/temp by hour-of-day as a stand-in for
    # a live weather forecast feed
    hourly_avg = df.groupby("HOUR")[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]].mean()

    # If the next moment is nighttime, jump forward to the next daylight
    # window so the forecast is actually useful to look at (searches up to
    # 48h ahead as a safety cap)
    candidate = last_ts + timedelta(minutes=15)
    for _ in range(48 * 4):
        if hourly_avg.loc[candidate.hour]["IRRADIATION"] > 0.05:
            break
        candidate += timedelta(minutes=15)
    forecast_start = candidate

    future_ts = [forecast_start + timedelta(minutes=15 * i) for i in range(hours * 4)]

    rows = []
    for sk in df["SOURCE_KEY"].unique():
        for ts in future_ts:
            hr = ts.hour
            avg = hourly_avg.loc[hr]
            feat = pd.DataFrame([{
                "IRRADIATION": avg["IRRADIATION"],
                "AMBIENT_TEMPERATURE": avg["AMBIENT_TEMPERATURE"],
                "MODULE_TEMPERATURE": avg["MODULE_TEMPERATURE"],
                "HOUR": ts.hour,
                "MINUTE": ts.minute,
                "DAY_OF_WEEK": ts.dayofweek,
                "MONTH": ts.month,
            }])[FEATURES]
            pred = model.predict(feat)[0]
            rows.append({
                "SOURCE_KEY": sk,
                "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "PREDICTED_AC_POWER": round(float(pred), 2),
            })
    return {"forecast_starts_at": forecast_start.strftime("%Y-%m-%d %H:%M:%S"), "data": rows}


@app.get("/anomalies")
def anomalies():
    _, alerts = detect_anomalies(get_data())
    alerts = alerts.copy()
    alerts["DATE_TIME"] = alerts["DATE_TIME"].astype(str)
    return alerts.to_dict(orient="records")


@app.get("/performance/summary")
def performance_summary():
    df = get_data()
    latest_per_inverter = df.sort_values("DATE_TIME").groupby("SOURCE_KEY").tail(1)
    summary = latest_per_inverter[["SOURCE_KEY", "DAILY_YIELD", "TOTAL_YIELD"]].copy()
    avg_efficiency = df.groupby("SOURCE_KEY")["CONVERSION_EFFICIENCY"].mean().reset_index()
    summary = summary.merge(avg_efficiency, on="SOURCE_KEY")
    summary = summary.rename(columns={"CONVERSION_EFFICIENCY": "AVG_CONVERSION_EFFICIENCY"})
    return summary.to_dict(orient="records")


@app.get("/optimize/schedule")
def optimize_schedule():
    """
    Recommends when to run flexible loads and when to charge a battery,
    based on the 24h forecast, to maximize solar self-consumption.
    Example loads are hardcoded for the demo — in production these would
    come from a user-configurable settings page.
    """
    df = get_data()
    model = get_model()
    curve = get_forecast_curve(df, model, hours_ahead=24)

    loads = [
        {"name": "EV Charging", "power_kw": 7, "duration_hours": 3},
        {"name": "Water Pump", "power_kw": 2, "duration_hours": 2},
        {"name": "Washing Machine", "power_kw": 1.5, "duration_hours": 1},
    ]
    return recommend_schedule(curve, loads, battery_capacity_kwh=50, battery_charge_rate_kw=10)
