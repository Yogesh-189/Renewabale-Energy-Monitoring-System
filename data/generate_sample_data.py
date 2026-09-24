"""
Generates synthetic solar plant data in the SAME SCHEMA as the popular Kaggle
dataset "Solar Power Generation Data" (Plant_1_Generation_Data.csv +
Plant_1_Weather_Sensor_Data.csv, by anikannal).

WHY THIS EXISTS
----------------
This lets us build and test the whole pipeline right now without waiting on a
manual Kaggle download. Once you download the real dataset from:
    https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
just drop the two CSVs into this /data folder with the same filenames
(Plant_1_Generation_Data.csv, Plant_1_Weather_Sensor_Data.csv) and every
downstream script works unchanged — the column names match exactly.

Columns produced:
  Generation_Data.csv:
    DATE_TIME, PLANT_ID, SOURCE_KEY, DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD
  Weather_Sensor_Data.csv:
    DATE_TIME, PLANT_ID, SOURCE_KEY, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

PLANT_ID = 4135001
N_INVERTERS = 4  # SOURCE_KEY = individual inverter/panel string
DAYS = 30
FREQ_MINUTES = 15

source_keys = [f"INV{i+1:02d}" for i in range(N_INVERTERS)]
start = datetime(2026, 8, 1, 0, 0)
n_steps = int((DAYS * 24 * 60) / FREQ_MINUTES)
timestamps = [start + timedelta(minutes=FREQ_MINUTES * i) for i in range(n_steps)]


def irradiation_curve(ts):
    """Bell-curve irradiation peaking at solar noon, zero at night."""
    hour = ts.hour + ts.minute / 60
    if hour < 6 or hour > 18.5:
        return 0.0
    x = (hour - 12.25) / 4.0
    base = np.exp(-x**2) * 0.9
    # cloud noise
    cloud = np.random.normal(0, 0.05)
    return max(0.0, base + cloud)


gen_rows = []
weather_rows = []
total_yield_tracker = {sk: np.random.uniform(500000, 900000) for sk in source_keys}
daily_yield_tracker = {sk: 0.0 for sk in source_keys}
last_day = timestamps[0].day

for ts in timestamps:
    if ts.day != last_day:
        daily_yield_tracker = {sk: 0.0 for sk in source_keys}
        last_day = ts.day

    irr = irradiation_curve(ts)
    ambient_temp = 24 + 6 * irr + np.random.normal(0, 0.8)
    module_temp = ambient_temp + 18 * irr + np.random.normal(0, 1.2)

    weather_rows.append({
        "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "PLANT_ID": PLANT_ID,
        "SOURCE_KEY": "WEATHER01",
        "AMBIENT_TEMPERATURE": round(ambient_temp, 3),
        "MODULE_TEMPERATURE": round(module_temp, 3),
        "IRRADIATION": round(irr, 5),
    })

    for sk in source_keys:
        # each inverter has a slightly different efficiency, and a 2% chance
        # of an underperformance fault this timestep (for anomaly testing)
        efficiency = np.random.uniform(0.92, 1.0)
        fault = np.random.random() < 0.02
        fault_mult = np.random.uniform(0.15, 0.5) if fault else 1.0

        dc_power = irr * 30000 * efficiency * fault_mult + np.random.normal(0, 15)
        dc_power = max(0.0, dc_power)
        ac_power = dc_power * 0.97  # inverter loss
        energy_kwh = ac_power * (FREQ_MINUTES / 60) / 1000

        daily_yield_tracker[sk] += energy_kwh
        total_yield_tracker[sk] += energy_kwh

        gen_rows.append({
            "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "PLANT_ID": PLANT_ID,
            "SOURCE_KEY": sk,
            "DC_POWER": round(dc_power, 3),
            "AC_POWER": round(ac_power, 3),
            "DAILY_YIELD": round(daily_yield_tracker[sk], 3),
            "TOTAL_YIELD": round(total_yield_tracker[sk], 3),
        })

gen_df = pd.DataFrame(gen_rows)
weather_df = pd.DataFrame(weather_rows)

gen_df.to_csv("C:/Users/Yoges/OneDrive/Desktop/1Project/Solar monitoring/solar-monitoring/Plant_1_Generation_Data.csv", index=False)
weather_df.to_csv("C:/Users/Yoges/OneDrive/Desktop/1Project/Solar monitoring/solar-monitoring/Plant_1_Weather_Sensor_Data.csv", index=False)

print(f"Generation rows: {len(gen_df)}")
print(f"Weather rows:    {len(weather_df)}")
print("Saved to C:/Users/Yoges/OneDrive/Desktop/1Project/Solar monitoring/solar-monitoring")
