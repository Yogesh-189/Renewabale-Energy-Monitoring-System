# ☀️ Renewable Energy Monitoring & Optimization System

An AI-powered, cloud-ready system for monitoring solar plant performance, forecasting power output, automatically detecting faulty inverters, and recommending optimal load-scheduling — built on real-world solar generation data.

![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)

---

## 📌 Problem Statement

Solar plant operators often lack tools to monitor performance in real time, forecast expected output, or catch underperforming equipment automatically. This project closes that gap with a complete software pipeline — from raw sensor data to live dashboard — that monitors generation, predicts future output using machine learning, flags faulty inverters automatically, and recommends when to run flexible loads to maximize solar self-consumption.

## ✨ Features

- **📊 Live Monitoring** — real-time generation, yield, and efficiency per inverter
- **🔮 ML-Based Forecasting** — Random Forest model predicting next-hours power output (**R² = 0.979**)
- **🚨 Automatic Fault Detection** — flags inverters that sustainedly underperform their forecast, filtering out sensor noise
- **⚡ Load & Battery Optimization** — recommends the best time windows to run flexible loads (EV charging, pumps, etc.) and charge a battery, based on the forecast
- **🌐 REST API** — a full FastAPI backend exposing every feature as an independent, testable endpoint
- **📈 Interactive Dashboard** — a single-page live dashboard with auto-refreshing stats, charts, and tables

## 🏗️ Architecture

```
Kaggle Dataset → Data Pipeline → ┬→ Forecasting Model  ┬→ Optimization Engine → FastAPI Backend → Live Dashboard
  (solar + weather CSVs)         └→ Anomaly Detection   ┘
```

| Stage | File | Purpose |
|---|---|---|
| Data layer | `data/` | Real solar generation + weather sensor data (Kaggle "Solar Power Generation Data") |
| Data pipeline | `src/data_pipeline.py` | Loads, cleans, merges, and feature-engineers the raw CSVs |
| Forecasting model | `src/train_forecast_model.py` | Random Forest Regressor predicting AC power output |
| Anomaly detection | `src/anomaly_detection.py` | Flags inverters underperforming their forecast |
| Optimization engine | `src/optimize_schedule.py` | Recommends load/battery scheduling from the forecast |
| Backend API | `src/api.py` | FastAPI service exposing all of the above as REST endpoints |
| Dashboard | `dashboard/index.html` | Self-contained live visual dashboard |

## 📈 Results

| Metric | Value |
|---|---|
| Forecast model accuracy (R²) | **0.979** |
| Forecast model error (MAE) | 17.94 kW |
| Dominant forecast feature | Irradiation (99.6% importance) — physically correct |
| Inverters monitored | 22 |
| Daylight readings evaluated | 33,270 |
| Confirmed fault alerts | 68 |
| Optimization result (example) | EV charging, water pump, washing machine all scheduled at 100% solar coverage |

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn (Random Forest Regressor)
- **Data processing:** pandas, numpy
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, vanilla JS, Chart.js
- **Model persistence:** joblib

## 📂 Project Structure

```
solar-monitoring/
├── data/
│   ├── generate_sample_data.py      # synthetic data matching the real schema
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
├── src/
│   ├── data_pipeline.py             # Stage 1: load, clean, feature-engineer
│   ├── train_forecast_model.py      # Stage 2: trains the forecasting model
│   ├── anomaly_detection.py         # Stage 3: fault/underperformance alerts
│   ├── optimize_schedule.py         # Stage 4: load/battery scheduling logic
│   └── api.py                       # Stage 5: FastAPI backend
├── dashboard/
│   └── index.html                   # Stage 6: live visual dashboard
├── models/
│   └── forecast_model.joblib        # trained model artifact
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or newer
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yogesh-189/Renewabale-Energy-Monitoring-System.git
cd Renewable-Energy-Monitoring-System

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Get the dataset

Download **["Solar Power Generation Data"](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)** by anikannal from Kaggle, and place both CSVs into `data/` — or use the included generator for synthetic test data:

```bash
python data/generate_sample_data.py
```

### Run the pipeline

```bash
cd src
python data_pipeline.py          # clean & feature-engineer the data
python train_forecast_model.py   # train the forecasting model
python anomaly_detection.py      # test fault detection
python optimize_schedule.py      # test the optimization engine
```

### Launch the API + Dashboard

```bash
uvicorn api:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/docs` for the interactive API docs, and double-click `dashboard/index.html` (with the API still running) to view the live dashboard.

## 🔌 API Reference

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /generation/latest` | Most recent reading per inverter |
| `GET /weather/latest` | Most recent irradiation & temperature readings |
| `GET /forecast?hours=6` | ML-predicted output for the next N daylight hours |
| `GET /anomalies` | Current confirmed fault alerts |
| `GET /performance/summary` | Daily/total yield + efficiency per inverter |
| `GET /optimize/schedule` | Recommended load & battery charge windows |

## 🗺️ Roadmap

- [x] Data pipeline
- [x] ML forecasting model
- [x] Anomaly detection
- [x] Optimization engine
- [x] REST API
- [x] Live dashboard
- [ ] Cloud deployment (AWS / Azure)
- [ ] Real-time data ingestion (Kafka / MQTT) in place of static CSVs
- [ ] API authentication & security (JWT, HTTPS)

## 👥 Team

| Name | Roll Number |
|---|---|
| Aazam Khan | 20241ISE3002 |
| Ajmeera Pavan Das | 20241ISE3004 |
| Yogesh H | 20231ISE0122 |

**Project:** Mini Project (CSE7102) — Presidency University, School of Computer Science and Engineering


## ⚠️ Note on the Dataset

This project uses a real, publicly available historical dataset (May–June 2020 solar plant readings). Dashboard dates reflect the dataset's own timestamps, not the current date — this is expected behavior for a system built on historical data, and forecasts/optimizations are computed relative to the dataset's own timeline. With a live sensor feed, the same code would use real current timestamps automatically.
