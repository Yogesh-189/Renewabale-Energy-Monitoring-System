"""
Live Weather API Integration — Stage 7
-----------------------------------------
Replaces the "historical same-hour average" stand-in used by /forecast with
a REAL weather forecast, pulled from Open-Meteo (https://open-meteo.com) —
a free, no-API-key-required weather API that conveniently provides
'shortwave_radiation' (solar irradiance, W/m^2) directly, alongside
temperature.

This is the #1 item from this project's own "Future Work" section: predict
using a real satellite-derived weather forecast instead of a rolling
historical average.

WHY OPEN-METEO
---------------
Most weather APIs (OpenWeatherMap, etc.) require a signup + API key and
give you cloud cover %, not irradiance directly, so you'd have to estimate
solar radiation from cloud cover yourself. Open-Meteo's forecast endpoint
returns hourly shortwave_radiation directly, in the same physical quantity
(W/m^2) our model needs, with zero signup friction — ideal for a student
project that needs to just work.

UNIT NOTE: Open-Meteo gives irradiance in W/m^2. The training dataset's
IRRADIATION column is in kW/m^2 (roughly 0-1 scale). We divide by 1000
before feeding the model, matching the training data's units exactly.

MODULE TEMPERATURE: live weather APIs report ambient air temperature, not
solar panel surface temperature. We estimate module temperature using a
simple linear offset (module_temp = ambient_temp + slope * irradiance)
fitted from this project's own historical dataset (see
fit_module_temp_offset below), rather than a hardcoded guess — so it
reflects this specific plant's real thermal behavior.
"""

import json
import os
import socket
import ssl
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Default location: Bengaluru, India (change to your own plant's coordinates)
DEFAULT_LAT = 12.9716
DEFAULT_LON = 77.5946


def fit_module_temp_offset(df: pd.DataFrame):
    """Fits module_temp - ambient_temp = slope * irradiance + intercept
    from the historical dataset, so the live-weather forecast can estimate
    a realistic module temperature without ever having measured it."""
    daylight = df[df["IRRADIATION"] > 0.05]
    x = daylight["IRRADIATION"].values
    y = (daylight["MODULE_TEMPERATURE"] - daylight["AMBIENT_TEMPERATURE"]).values
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "weather_cache.json")
OPEN_METEO_HOST = "api.open-meteo.com"


def _ssl_context():
    """Python on Windows sometimes cannot find trusted root certificates
    (typical with antivirus / corporate HTTPS inspection), which makes every
    HTTPS call fail with CERTIFICATE_VERIFY_FAILED even though the browser
    works fine. If the 'certifi' package is installed we use its bundle."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_get_json(url: str, timeout: float = 10.0, retries: int = 1):
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_ssl_context()) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:  # keep the REAL error - it is what you need to debug
            last_err = e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


def _cache_key(lat, lon):
    return f"{round(float(lat), 2)},{round(float(lon), 2)}"


def _read_cache_file() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(lat, lon, data):
    """Keeps the last good response PER LOCATION, so switching location in the
    dashboard does not throw away the saved forecast of the previous one."""
    entries = _read_cache_file()
    entries[_cache_key(lat, lon)] = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"), "payload": data}
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f)
    except OSError:
        pass


def _load_cache(lat, lon):
    return _read_cache_file().get(_cache_key(lat, lon))


def _payload_to_df(data: dict, hours: int) -> pd.DataFrame:
    try:
        times = data["hourly"]["time"]
        radiation = data["hourly"]["shortwave_radiation"]
        temperature = data["hourly"]["temperature_2m"]
    except KeyError as e:
        raise RuntimeError(f"Weather API response missing expected field: {e}")
    df = pd.DataFrame({
        "DATE_TIME": pd.to_datetime(times),
        "IRRADIATION_W_M2": radiation,
        "AMBIENT_TEMPERATURE": temperature,
    })
    # Open-Meteo returns times in the LOCATION's local time (timezone=auto).
    # Compare them with "now" in that same zone - using this computer's clock
    # would pick the wrong hours for any location outside its own timezone.
    offset = data.get("utc_offset_seconds")
    if offset is None:
        now = pd.Timestamp.now()
    else:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(seconds=int(offset))
    df = df[df["DATE_TIME"] >= now.floor("h")]
    df = df.head(hours + 2)  # small buffer for the 15-min interpolation below
    if df.empty:
        raise RuntimeError("Weather data contained no upcoming hours")
    return df.reset_index(drop=True)


def fetch_live_weather_with_meta(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, hours: int = 24):
    """
    Returns (DataFrame, meta). Tries the live Open-Meteo API first; if the
    network call fails but an earlier successful response is cached on disk,
    falls back to that and says so in `meta` instead of failing silently.
    Raises RuntimeError (with the underlying cause) if neither works.
    """
    params = (
        f"latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,temperature_2m"
        f"&forecast_days=2&timezone=auto"
    )
    url = f"{OPEN_METEO_URL}?{params}"
    try:
        data = _http_get_json(url)
        df = _payload_to_df(data, hours)
        _save_cache(lat, lon, data)
        return df, {"source": "live", "fetched_at": datetime.now().isoformat(timespec="seconds"),
                    "timezone": data.get("timezone"), "utc_offset_seconds": data.get("utc_offset_seconds")}
    except RuntimeError as live_err:
        cached = _load_cache(lat, lon)
        if cached:
            try:
                df = _payload_to_df(cached["payload"], hours)
                return df, {"source": "cache", "fetched_at": cached["fetched_at"],
                            "timezone": cached["payload"].get("timezone"),
                            "utc_offset_seconds": cached["payload"].get("utc_offset_seconds"),
                            "warning": f"Live request failed ({live_err}); showing last saved forecast."}
            except RuntimeError:
                pass
        raise RuntimeError(f"Could not get weather from Open-Meteo - {live_err}")


def fetch_live_weather(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, hours: int = 24):
    """Backwards-compatible wrapper that returns only the DataFrame."""
    return fetch_live_weather_with_meta(lat, lon, hours)[0]


def diagnose_connectivity() -> dict:
    """Step-by-step check of WHY Python cannot reach Open-Meteo, so the
    failure can be named exactly rather than guessed at."""
    report = {"python": sys.version.split()[0], "openssl": ssl.OPENSSL_VERSION,
              "proxies_seen_by_python": urllib.request.getproxies(), "steps": []}

    def step(name, fn):
        try:
            detail = fn()
            report["steps"].append({"step": name, "ok": True, "detail": detail})
            return True
        except Exception as e:
            report["steps"].append({"step": name, "ok": False,
                                    "detail": f"{type(e).__name__}: {e}"})
            return False

    try:
        import certifi
        report["certifi"] = certifi.where()
    except ImportError:
        report["certifi"] = "NOT INSTALLED  (fix: pip install certifi)"

    dns_ok = step("1. DNS lookup of api.open-meteo.com",
                  lambda: socket.getaddrinfo(OPEN_METEO_HOST, 443)[0][4][0])
    tcp_ok = dns_ok and step("2. TCP connect to port 443",
                             lambda: (socket.create_connection((OPEN_METEO_HOST, 443), timeout=8).close(), "connected")[1])
    https_ok = tcp_ok and step("3. HTTPS request (small test forecast)",
                               lambda: "HTTP OK, keys: " + ",".join(_http_get_json(
                                   f"{OPEN_METEO_URL}?latitude={DEFAULT_LAT}&longitude={DEFAULT_LON}"
                                   f"&hourly=temperature_2m&forecast_days=1", retries=0).keys()))

    if https_ok:
        cause, fix = "None - Open-Meteo is reachable from Python.", "Nothing to fix. Refresh the dashboard."
    elif not dns_ok:
        cause = "Python cannot resolve api.open-meteo.com (DNS)."
        fix = ("Network/DNS filtering - common on college Wi-Fi. Try a mobile hotspot, or set a public DNS "
               "(8.8.8.8 / 1.1.1.1) on your network adapter.")
    elif not tcp_ok:
        cause = "DNS works but outbound port 443 is blocked or timing out for Python."
        fix = ("A firewall or proxy is blocking it. Allow python.exe in Windows Defender Firewall / antivirus, "
               "or use a different network. If you need a proxy: set HTTPS_PROXY before starting uvicorn.")
    else:
        failed = report["steps"][-1]["detail"]
        if "CERTIFICATE" in failed.upper() or "SSL" in failed.upper():
            cause = "TLS certificate verification fails inside Python (browser is unaffected)."
            fix = "Run: pip install certifi   then restart uvicorn. If it persists, antivirus HTTPS-scanning is intercepting."
        else:
            cause, fix = "Connected, but the HTTPS request failed: " + failed, "Check the error above."
    report["likely_cause"], report["suggested_fix"] = cause, fix
    report["reachable"] = bool(https_ok)
    return report


def build_live_feature_rows(weather_df: pd.DataFrame, module_slope: float, module_intercept: float, hours: int):
    """
    Interpolates the hourly weather forecast to 15-minute steps (matching
    the model's training granularity) and builds one feature-row dict per
    timestamp, ready to feed straight into the trained model.
    """
    weather_df = weather_df.set_index("DATE_TIME")
    start = weather_df.index[0]
    end = start + timedelta(hours=hours)
    fifteen_min_index = pd.date_range(start=start, end=end, freq="15min")

    irr_interp = weather_df["IRRADIATION_W_M2"].reindex(
        weather_df.index.union(fifteen_min_index)
    ).interpolate(method="time").reindex(fifteen_min_index)
    temp_interp = weather_df["AMBIENT_TEMPERATURE"].reindex(
        weather_df.index.union(fifteen_min_index)
    ).interpolate(method="time").reindex(fifteen_min_index)

    rows = []
    for ts, irr_w, temp in zip(fifteen_min_index, irr_interp.values, temp_interp.values):
        irr_kw = max(0.0, float(irr_w) / 1000.0)  # W/m^2 -> kW/m^2, matching training units
        module_temp = float(temp) + module_slope * irr_kw + module_intercept
        rows.append({
            "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "IRRADIATION": round(irr_kw, 4),
            "AMBIENT_TEMPERATURE": round(float(temp), 2),
            "MODULE_TEMPERATURE": round(module_temp, 2),
            "HOUR": ts.hour,
            "MINUTE": ts.minute,
            "DAY_OF_WEEK": ts.dayofweek,
            "MONTH": ts.month,
        })
    return rows
