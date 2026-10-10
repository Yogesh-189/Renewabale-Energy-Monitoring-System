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
    GET  /generation/latest            -> most recent readings per inverter (historical dataset)
    GET  /weather/latest               -> most recent weather reading (historical dataset)
    GET  /forecast?hours=6             -> forecast for the next N hours
    GET  /anomalies                    -> confirmed underperformance readings (15-min rows)
    GET  /anomalies/incidents          -> the same, grouped into incidents (recent vs historical)
    GET  /forecast/live-weather        -> forecast from a REAL weather forecast (Open-Meteo)
    GET  /weather/diagnose             -> why can't Python reach Open-Meteo? step-by-step check
    GET  /performance/summary          -> daily yield + efficiency per inverter
    GET  /optimize/schedule            -> recommended load/battery schedule
    GET  /live/status                  -> real-time ingestion simulator status
    GET  /live/latest                  -> most recent LIVE reading per inverter (real clock time)
    GET  /live/history?minutes=10      -> recent live readings, for a live chart
"""

import os
import joblib
import pandas as pd
import numpy as np
import warnings
import webbrowser
import threading
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from data_pipeline import build_dataset
from train_forecast_model import FEATURES
from anomaly_detection import detect_anomalies, group_incidents
from optimize_schedule import get_forecast_curve, recommend_schedule, SITE_INSTALLED_KW
from live_ingestion import live_store
from weather_api import (fetch_live_weather_with_meta, build_live_feature_rows, fit_module_temp_offset,
                         diagnose_connectivity, DEFAULT_LAT, DEFAULT_LON)

# scikit-learn prints this harmless notice whenever a multi-threaded model is
# called from FastAPI's worker threads; it only clutters the terminal.
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

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
        model = joblib.load(MODEL_PATH)
        # The model was trained with n_jobs=-1. For the tiny batches the API
        # predicts (a few dozen rows) spinning up a thread pool per call costs
        # far more than the prediction itself, so inference runs single-threaded.
        model.n_jobs = 1
        _cache["model"] = model
    return _cache["model"]


def get_anomaly_result():
    """Anomaly detection scores every historical reading with the model - a few
    seconds of work - and the dataset never changes while the server runs, so
    compute it once and reuse it (it used to be recomputed on every request)."""
    if "anomaly" not in _cache:
        model = get_model()
        model.n_jobs = -1            # one big batch: parallel is a real win here
        try:
            daylight, alerts = detect_anomalies(get_data(), model)
        finally:
            model.n_jobs = 1
        _cache["anomaly"] = {
            "alerts": alerts,
            "daylight_readings": int(len(daylight)),
            "incidents": group_incidents(alerts, get_data()["DATE_TIME"].max()),
        }
    return _cache["anomaly"]


def get_module_temp_offset():
    if "module_offset" not in _cache:
        df = get_data()
        _cache["module_offset"] = fit_module_temp_offset(df)
    return _cache["module_offset"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def open_docs_on_start():
    # Auto-opens the API docs in your default browser 1.5s after the server
    # starts. Only for your own laptop: the Dockerfile / render.yaml set
    # OPEN_DOCS_ON_START=0 because a cloud server has no browser to open.
    if os.environ.get("OPEN_DOCS_ON_START", "1") != "1":
        return
    def _open():
        webbrowser.open("http://127.0.0.1:8000/docs")
    threading.Timer(1.5, _open).start()


@app.on_event("startup")
def start_live_ingestion():
    # Launches the real-time sensor simulator (src/live_ingestion.py) in a
    # background thread. It runs for as long as this API process is alive,
    # pushing a new reading every TICK_SECONDS using the actual current
    # time — this is the "real-time ingestion" stage of the roadmap.
    live_store.start()


@app.on_event("startup")
def warm_up_caches():
    # Load data + model and run anomaly detection once in the background, so
    # the first dashboard load is not stuck waiting on it.
    def _warm():
        try:
            if not os.path.exists(MODEL_PATH):
                # A fresh cloud deployment may not contain models/forecast_model.joblib
                # (large binary files are often left out of Git). Train it from the
                # dataset once; this takes a minute, and /forecast answers 503 until done.
                print("[api] no trained model found - training one from the dataset...")
                from train_forecast_model import train
                train()
            get_data(); get_model(); get_module_temp_offset(); get_anomaly_result()
            print("[api] caches warmed - dashboard requests are now fast")
        except Exception as e:
            print(f"[api] warm-up skipped: {e}")
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/live/status")
def live_status():
    """Whether the real-time ingestion simulator is running, and how many
    readings it has pushed so far — shown as the 'Streaming live' badge on
    the dashboard."""
    return live_store.status()


@app.get("/live/latest")
def live_latest():
    """Most recent live reading per (simulated) inverter, using the
    system's actual current time — unlike every other endpoint in this
    API, which reads the static 2020 Kaggle dataset."""
    return live_store.latest()


@app.get("/live/history")
def live_history(minutes: int = 10):
    """Every buffered live reading from the last N real minutes, for the
    dashboard's live-updating chart."""
    return live_store.history(minutes)


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

    future_ts = pd.DatetimeIndex([forecast_start + timedelta(minutes=15 * i) for i in range(hours * 4)])

    # Features depend only on the timestamp, not on the inverter, so predict
    # each timestamp ONCE in a single batch and reuse it for every inverter.
    weather = hourly_avg.reindex(future_ts.hour).fillna(0.0)
    feats = pd.DataFrame({
        "IRRADIATION": weather["IRRADIATION"].values,
        "AMBIENT_TEMPERATURE": weather["AMBIENT_TEMPERATURE"].values,
        "MODULE_TEMPERATURE": weather["MODULE_TEMPERATURE"].values,
        "HOUR": future_ts.hour,
        "MINUTE": future_ts.minute,
        "DAY_OF_WEEK": future_ts.dayofweek,
        "MONTH": future_ts.month,
    })[FEATURES]
    preds = model.predict(feats)

    rows = []
    for sk in df["SOURCE_KEY"].unique():
        for ts, pred in zip(future_ts, preds):
            rows.append({
                "SOURCE_KEY": sk,
                "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "PREDICTED_AC_POWER": round(float(pred), 2),
            })
    return {"forecast_starts_at": forecast_start.strftime("%Y-%m-%d %H:%M:%S"), "data": rows}


@app.get("/forecast/live-weather")
def forecast_live_weather(hours: int = 6, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON):
    """
    A REAL weather-driven forecast: pulls actual forecasted irradiance and
    temperature from the Open-Meteo API for (lat, lon), then runs it
    through the SAME trained model used everywhere else in this API.

    Unlike /forecast (which estimates future weather from this plant's own
    historical same-hour averages), this endpoint uses genuine forecasted
    weather for today/tomorrow — so timestamps here are real current
    dates, not the 2020 historical dataset dates seen elsewhere in this API.

    Requires internet access from wherever the API is running. Returns a
    503 with a clear message if the weather API can't be reached.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Latitude must be -90..90 and longitude -180..180")
    model = get_model()
    n_inverters = get_data()["SOURCE_KEY"].nunique()
    module_slope, module_intercept = get_module_temp_offset()

    try:
        weather_df, meta = fetch_live_weather_with_meta(lat=lat, lon=lon, hours=hours)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    feature_rows = build_live_feature_rows(weather_df, module_slope, module_intercept, hours=hours)
    preds = model.predict(pd.DataFrame(feature_rows)[FEATURES])      # one batch

    out = []
    for feat, pred_per_inverter in zip(feature_rows, preds):
        out.append({
            "DATE_TIME": feat["DATE_TIME"],
            "IRRADIATION": feat["IRRADIATION"],
            "AMBIENT_TEMPERATURE": feat["AMBIENT_TEMPERATURE"],
            "PREDICTED_TOTAL_AC_POWER": round(float(pred_per_inverter) * n_inverters, 2),
        })
    return {"location": {"lat": lat, "lon": lon}, "source": "Open-Meteo", "meta": meta, "data": out}


@app.get("/weather/diagnose")
def weather_diagnose():
    """Open this in the browser if the live-weather panel says 'Offline'. It
    tests DNS, the TCP connection and the HTTPS request separately and names
    the likely cause and fix."""
    return diagnose_connectivity()


@app.get("/anomalies")
def anomalies():
    """Every confirmed underperformance READING (one row per 15 minutes)."""
    alerts = get_anomaly_result()["alerts"].copy()
    alerts["DATE_TIME"] = alerts["DATE_TIME"].astype(str)
    return alerts.to_dict(orient="records")


@app.get("/anomalies/incidents")
def anomaly_incidents():
    """The same alerts grouped into INCIDENTS (one inverter, one continuous
    period), with a flag for which ones are recent vs. historical."""
    result = get_anomaly_result()
    return {**result["incidents"], "daylight_readings_evaluated": result["daylight_readings"]}


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
    plan = recommend_schedule(curve, loads, battery_capacity_kwh=50, battery_charge_rate_kw=10)
    plan["site_installed_kw"] = SITE_INSTALLED_KW
    return plan


# ---------------------------------------------------------------------------
# Serve the dashboard from the API itself, so a single deployed URL hosts the
# whole system (the dashboard then talks to the API on the same address).
# Locally you can still just double-click dashboard/index.html as before.
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")

if os.path.isdir(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/dashboard/")
