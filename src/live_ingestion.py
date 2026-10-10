"""
Real-Time Data Ingestion — Stage 6
------------------------------------
This module replaces the "static CSV read" limitation from the roadmap with
a genuine real-time ingestion pipeline: a background thread that behaves
like a live IoT sensor feed, pushing a new reading for every inverter at a
fixed interval, using the ACTUAL current wall-clock time (not the 2020
historical dataset).

WHY A SIMULATOR
----------------
We don't have physical solar hardware wired up. In its place, this
generates PHYSICALLY REALISTIC readings (a real sunrise-to-sunset
irradiance curve, matching ambient/module temperature relationships) at
whatever the real current hour is right now, so the dashboard shows true
live behavior: zero output at night, rising through the morning, peaking
at solar noon — happening in real time, at your actual location's clock
time.

This is the same architectural role a message broker (Kafka/MQTT) plays in
the full roadmap: field devices -> ingestion -> a queryable live store.
Swapping this simulator for a real MQTT subscriber later requires no
changes to anything downstream (the API endpoints, the dashboard) because
they only depend on the LiveStore interface below, not on how readings
arrive.

Persistence: every reading is also appended to data/live_stream.csv, so
the stream survives an API restart (a lightweight stand-in for the
"Time-Series DB" layer in the roadmap).
"""

import os
import threading
import time
import csv
from datetime import datetime, timedelta, timezone
from collections import deque

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LIVE_CSV_PATH = os.path.join(DATA_DIR, "live_stream.csv")

N_INVERTERS = 6
SOURCE_KEYS = [f"LIVE-INV{i+1:02d}" for i in range(N_INVERTERS)]
TICK_SECONDS = 8          # how often (real seconds) a new batch of readings arrives
MAX_BUFFER_ROWS = 2000    # rolling in-memory window (per-inverter + weather rows)

CSV_FIELDS = [
    "DATE_TIME", "SOURCE_KEY", "DC_POWER", "AC_POWER",
    "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
]


# The simulated plant keeps its OWN local time. A cloud server runs on UTC, so
# using the server's clock made the feed show 'night' at 9 am in India.
# Default is India Standard Time (UTC+5:30); override with PLANT_UTC_OFFSET_MINUTES.
PLANT_TZ = timezone(timedelta(minutes=int(os.environ.get("PLANT_UTC_OFFSET_MINUTES", "330"))))


def _plant_now() -> datetime:
    """Current time at the simulated plant (timezone-aware)."""
    return datetime.now(PLANT_TZ)


def _irradiance_now(now: datetime) -> float:
    """Bell-curve irradiance peaking at solar noon (12:00), zero outside
    06:00-18:30 — the same physical model used by the offline sample
    generator, but driven by the real current clock instead of a fixed
    date range."""
    import math
    hour = now.hour + now.minute / 60 + now.second / 3600
    if hour < 6 or hour > 18.5:
        return 0.0
    x = (hour - 12.25) / 4.0
    return max(0.0, math.exp(-x * x) * 0.9)


class LiveStore:
    """Thread-safe rolling buffer of live sensor readings."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows = deque(maxlen=MAX_BUFFER_ROWS)
        self._ticks = 0
        self._started_at = None
        self._running = False
        self._thread = None
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(LIVE_CSV_PATH):
            with open(LIVE_CSV_PATH, "w", newline="") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def _tick(self):
        import random
        now = _plant_now()
        irr = _irradiance_now(now)
        ambient = 24 + 6 * irr + random.gauss(0, 0.6)
        module = ambient + 18 * irr + random.gauss(0, 1.0)

        new_rows = []
        for sk in SOURCE_KEYS:
            efficiency = random.uniform(0.93, 0.99)
            if irr < 0.01:
                dc_power = 0.0  # true night: no reverse current, exactly zero
            else:
                dc_power = max(0.0, irr * 6000 * efficiency + random.gauss(0, 8))
            ac_power = dc_power * 0.97
            row = {
                "DATE_TIME": now.strftime("%Y-%m-%d %H:%M:%S"),
                "SOURCE_KEY": sk,
                "DC_POWER": round(dc_power, 2),
                "AC_POWER": round(ac_power, 2),
                "AMBIENT_TEMPERATURE": round(ambient, 2),
                "MODULE_TEMPERATURE": round(module, 2),
                "IRRADIATION": round(irr, 4),
            }
            new_rows.append(row)

        with self._lock:
            for row in new_rows:
                self._rows.append(row)
            self._ticks += 1

        with open(LIVE_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            for row in new_rows:
                writer.writerow(row)

    def _run(self):
        while self._running:
            self._tick()
            time.sleep(TICK_SECONDS)

    def start(self):
        if self._running:
            return
        self._running = True
        self._started_at = _plant_now()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def status(self):
        with self._lock:
            return {
                "running": self._running,
                "started_at": self._started_at.strftime("%Y-%m-%d %H:%M:%S") if self._started_at else None,
                "ticks": self._ticks,
                "buffered_rows": len(self._rows),
                "inverters": SOURCE_KEYS,
                "tick_interval_seconds": TICK_SECONDS,
            }

    def latest(self):
        """Most recent reading per inverter."""
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return []
        latest_by_key = {}
        for row in rows:
            latest_by_key[row["SOURCE_KEY"]] = row
        return list(latest_by_key.values())

    def history(self, minutes: int = 10):
        """All buffered readings from the last N real minutes, for charting."""
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return []
        cutoff = _plant_now().replace(tzinfo=None) - timedelta(minutes=minutes)
        out = []
        for row in rows:
            ts = datetime.strptime(row["DATE_TIME"], "%Y-%m-%d %H:%M:%S")
            if ts >= cutoff:
                out.append(row)
        return out


# Single shared instance the API imports
live_store = LiveStore()
