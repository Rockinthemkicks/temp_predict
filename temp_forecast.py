#!/usr/bin/env python3
"""
temp_forecast.py — Multi-method temperature forecasting tool

Implements the five forecasting strategies from ATMS-313 "Temperature
Forecasting Techniques", AND the course's final synthesis step (slide 38,
"How Do I Forecast?"): don't just average every method blindly -- figure out
whether the pattern is stagnant or changing, weight the methods that fit
that pattern more heavily, and treat highs and lows separately.

  1. Persistence            - tomorrow = today
  2. Delta Method           - tomorrow = today's actual + (model's day2-day1 delta)
  3. Analogous Thickness    - find an upstream point with matching 1000-500mb
                               thickness & MSLP 24h ago; use its temperature
  4. MOS Error Advection    - average today's MOS bias across one or more
                               upstream stations; apply it to tomorrow's MOS
  5. Dewpoint Method (lows) - tonight's low ~= today's afternoon dewpoint - (0-3F)

Synthesis (new): rather than a flat average, the tool
  - classifies the pattern as stagnant / moderate change / rapidly changing,
    from the Delta Method's day-over-day model change and the 500mb steering
    wind speed, and weights Persistence/Delta down (and Analogous Thickness/
    MOS Error Advection up) as the pattern gets less stagnant -- this is
    exactly the judgement call the course slides ask you to make manually;
  - discounts the Analogous Thickness point when its elevation differs a lot
    from the target's, per the course's explicit elevation caveat;
  - discounts the Dewpoint Method when tomorrow morning's forecast wind/cloud
    cover breaks its clear-sky, light-wind assumption;
  - averages MOS error across ALL upstream stations you give it, per the
    slide 28 instruction "don't just look at one station, want average."

Data sources:
  - Open-Meteo (api.open-meteo.com)            -> obs/forecast, pressure levels
  - Iowa Environmental Mesonet MOS API (IEM)    -> real NWS MOS bulletins
  - Iowa Environmental Mesonet ASOS API (IEM)   -> real station observations

No API key required for either service.

Usage:
    python3 temp_forecast.py --lat 41.98 --lon -87.90 --station KORD

    # average MOS error across several upstream stations (recommended):
    python3 temp_forecast.py --lat 41.98 --lon -87.90 --station KORD \\
        --upstream-station KOMA,KDSM,KFSD

    # run the built-in list of major US cities and print a compact table
    # of tomorrow's regime-weighted high/low for each (edit DEFAULT_CITIES
    # in this file to change the list):
    python3 temp_forecast.py --all-cities

macOS notes: run with `python3`, not `python`. If you installed Python from
python.org (not Homebrew) and get an SSL/certificate error, run
"/Applications/Python 3.x/Install Certificates.command" once, or
`pip3 install --upgrade certifi`.

See README.md for full option list and method-by-method explanation of the
assumptions baked into each forecast.
"""

import argparse
import contextlib
import io
import json
import math
import os
import re
import smtplib
import ssl
import sys
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    ZoneInfo = None

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
IEM_MOS_URL = "https://mesonet.agron.iastate.edu/api/1/mos.json"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

VALID_MOS_MODELS = {"AVN", "GFS", "ETA", "NAM", "NBS", "NBE", "ECM", "LAV", "MEX"}

# Regime -> per-method confidence weight (0-1) used in the weighted synthesis.
# "analog" and "mos_error_advection" get MORE weight as the pattern becomes
# less stagnant; "persistence" gets LESS. This encodes the judgement call
# slide 38 asks the forecaster to make by hand.
REGIME_WEIGHTS = {
    "stagnant":          {"persistence": 1.0,  "delta": 1.0, "analog": 0.4,  "mos_error_advection": 0.5},
    "moderate change":   {"persistence": 0.4,  "delta": 1.0, "analog": 0.85, "mos_error_advection": 0.9},
    "rapidly changing":  {"persistence": 0.15, "delta": 0.6, "analog": 1.0,  "mos_error_advection": 1.0},
}

# Slide-derived alternative to REGIME_WEIGHTS (see backtest_accuracy.py's
# module docstring for the reasoning per method/regime). Never validated on
# held-out data -- kept here for comparison, not because it's known better.
REGIME_WEIGHTS_ALT = {
    "stagnant":          {"persistence": 1.0, "delta": 1.0,  "analog": 0.3, "mos_error_advection": 0.5},
    "moderate change":   {"persistence": 0.3, "delta": 1.0,  "analog": 1.0, "mos_error_advection": 0.8},
    "rapidly changing":  {"persistence": 0.1, "delta": 0.25, "analog": 0.5, "mos_error_advection": 1.0},
}

# Empirically fit via fit_regime_weights.py on 120 backtest days at KORD
# (Jun-Sep, GFS and NBS MOS models combined, upstream stations KRFD/KDVN/
# KMLI for MOS Error Advection), grid-searching every method's weight per
# regime -- including raw_mos, an optional 5th candidate outside the
# course's five specified methods (see method_raw_mos) -- to minimize MAE,
# then validated on a held-out slice those weights never trained on.
# Fitted MAE beat both tables above by 15-44% on that holdout, per regime:
#   moderate change: Delta 0.40 + Raw MOS 1.00 (dominant) + Analog 0.05 --
#     raw, uncorrected MOS outperformed its own bias-corrected version here.
#   rapidly changing: MOS Error Advection alone -- makes physical sense,
#     since advecting yesterday's error pattern into tomorrow is most
#     valuable exactly when the pattern is actively changing.
# "stagnant" had only 6 train days (below the 15-day reliability floor the
# fitting script uses), so it's left on REGIME_WEIGHTS unchanged rather
# than guessed at.
#
# CAUTION: single station, two MOS models, one (warm) season. This is
# real, validated evidence for KORD in summer -- it is not yet evidence
# this generalizes to other stations, terrain, or winter frontal patterns.
# Treat as an experiment to compare against, not a proven replacement,
# until backtested more broadly.
REGIME_WEIGHTS_FITTED = {
    "stagnant":          {"persistence": 1.0,  "delta": 1.0,  "analog": 0.4,  "mos_error_advection": 0.5,  "raw_mos": 0.0},
    "moderate change":   {"persistence": 0.00, "delta": 0.40, "analog": 0.05, "mos_error_advection": 0.00, "raw_mos": 1.0},
    "rapidly changing":  {"persistence": 0.00, "delta": 0.00, "analog": 0.00, "mos_error_advection": 1.0,  "raw_mos": 0.0},
}

REGIME_WEIGHTS_TABLES = {
    "current": REGIME_WEIGHTS,
    "alt": REGIME_WEIGHTS_ALT,
    "fitted": REGIME_WEIGHTS_FITTED,
}

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# "Today" anchoring
# ----------------------------------------------------------------------
# The methods below treat "today" as a UTC day (00Z-24Z). That's fine when the
# tool runs before 00Z, but a run in the local evening (e.g. 7 PM Denver =
# 01Z the NEXT UTC day) would see a brand-new, nearly empty UTC day and
# forecast the wrong "tomorrow". effective_now() fixes this: if the UTC date
# has already rolled past the user's local calendar date, it returns 23:59Z
# of the local date instead, so "today" is the just-finished UTC day (which
# covers the full local afternoon) and "tomorrow" is the correct next day.
# Override the zone with the TEMP_FORECAST_TZ environment variable.
LOCAL_TZ_NAME = os.environ.get("TEMP_FORECAST_TZ", "America/Denver")


def local_today():
    """The user's local calendar date (default America/Denver)."""
    now_utc = datetime.now(timezone.utc)
    try:
        tz = ZoneInfo(LOCAL_TZ_NAME)
    except Exception:
        tz = timezone(timedelta(hours=-7))  # fixed-offset fallback (MST)
    return now_utc.astimezone(tz).date()


def effective_now():
    """Timezone-aware UTC 'now', held back to 23:59Z of the local date when
    the UTC date has already moved ahead of it (see comment above)."""
    now_utc = datetime.now(timezone.utc)
    ldate = local_today()
    if now_utc.date() > ldate:
        return datetime.combine(ldate, datetime.min.time(), tzinfo=timezone.utc) \
            + timedelta(hours=23, minutes=59)
    return now_utc


def c_to_f(c):
    if c is None:
        return None
    return c * 9.0 / 5.0 + 32.0


def kmh_to_mph(v):
    if v is None:
        return None
    return v * 0.621371


def destination_point(lat, lon, bearing_deg, distance_km):
    """Given a start point, a bearing (degrees, meteorological convention:
    direction the wind is blowing FROM), and a distance, return the lat/lon
    of the point that lies in that direction. This is how we build
    'upstream' search candidates for the analogous-thickness technique."""
    R = 6371.0088
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    d_r = distance_km / R
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def nearest_index(times, target_dt):
    """Index into an hourly ISO-8601 time list closest to target_dt.

    Open-Meteo returns naive timestamps (e.g. '2026-09-03T00:00') that
    represent the requested timezone (UTC in this tool), so target_dt must
    also be a naive UTC datetime -- never timezone-aware -- or the
    subtraction below raises TypeError."""
    best_i, best_diff = 0, None
    for i, t in enumerate(times):
        tt = datetime.fromisoformat(t)
        diff = abs((tt - target_dt).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_i = i
    return best_i


# ----------------------------------------------------------------------
# Open-Meteo access
# ----------------------------------------------------------------------


def fetch_open_meteo(lat, lon, hourly_vars, past_days=2, forecast_days=3,
                      model=None, timezone_mode="UTC"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": timezone_mode,
    }
    if model:
        params["models"] = model
    r = requests.get(OPEN_METEO_URL, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def fetch_open_meteo_multi(lat_list, lon_list, hourly_vars, past_days=1,
                            forecast_days=2, timezone_mode="UTC"):
    """Batch request: Open-Meteo accepts comma-separated lat/lon lists and
    returns one result object per location (a list if >1 location)."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lat_list),
        "longitude": ",".join(f"{v:.4f}" for v in lon_list),
        "hourly": ",".join(hourly_vars),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": timezone_mode,
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=40)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]
    return data


def fetch_daily_maxmin(lat, lon, model=None):
    """Today's and tomorrow's model-forecast daily high/low, from the same
    run -- used by the Delta Method."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "forecast_days": 3,
        "timezone": "auto",
    }
    if model:
        params["models"] = model
    r = requests.get(OPEN_METEO_URL, params=params, timeout=25)
    r.raise_for_status()
    return r.json()["daily"]


def fetch_overnight_conditions(lat, lon):
    """Rough proxy for tonight's radiative-cooling conditions, used to weight
    the Dewpoint Method: average wind speed & cloud cover over a fixed UTC
    window (03Z-12Z tomorrow) that covers early-morning hours for most of the
    continental US. This is a coarse stand-in for the station's actual local
    pre-dawn window, in keeping with the rest of the tool's UTC-based
    simplification (see nearest_index docstring)."""
    data = fetch_open_meteo(lat, lon, ["wind_speed_10m", "cloud_cover"],
                             past_days=0, forecast_days=2, timezone_mode="UTC")
    h = data["hourly"]
    now = effective_now().replace(tzinfo=None)
    tomorrow = now.date() + timedelta(days=1)
    win_start = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=3)
    win_end = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=12)
    winds, clouds = [], []
    for i, t in enumerate(h["time"]):
        tt = datetime.fromisoformat(t)
        if win_start <= tt <= win_end:
            if h["wind_speed_10m"][i] is not None:
                winds.append(h["wind_speed_10m"][i])
            if h["cloud_cover"][i] is not None:
                clouds.append(h["cloud_cover"][i])
    if not winds or not clouds:
        return None
    avg_wind_kmh = sum(winds) / len(winds)
    avg_cloud_pct = sum(clouds) / len(clouds)
    return {
        "avg_wind_speed_mph": round(kmh_to_mph(avg_wind_kmh), 1),
        "avg_cloud_cover_pct": round(avg_cloud_pct, 0),
        "window_utc": f"{win_start.isoformat()}Z - {win_end.isoformat()}Z",
    }


# ----------------------------------------------------------------------
# IEM ASOS (real station observations) access
# ----------------------------------------------------------------------


def fetch_asos_today(station, tz_offset_hours=0):
    """Pull today's raw hourly temperature/dewpoint obs for a station so we
    can compute an actual (not model-analysis) high/low and afternoon
    dewpoint. tz_offset_hours lets you request 'local day' instead of UTC
    day; left at 0 (UTC day) by default since that's unambiguous."""
    now = effective_now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    params = {
        "station": station,
        "data": "tmpf,dwpf",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "hour1": 0, "minute1": 0,
        "year2": now.year, "month2": now.month, "day2": now.day,
        "hour2": now.hour, "minute2": now.minute,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
    }
    r = requests.get(IEM_ASOS_URL, params=params, timeout=25)
    r.raise_for_status()
    lines = [ln for ln in r.text.strip().splitlines()
             if ln and not ln.startswith("station") and not ln.startswith("#")]
    temps, dewpoints = [], []
    for ln in lines:
        parts = ln.split(",")
        if len(parts) < 4:
            continue
        _, valid, tmpf, dwpf = parts[0], parts[1], parts[2], parts[3]
        try:
            if tmpf not in ("M", ""):
                temps.append((valid, float(tmpf)))
        except ValueError:
            pass
        try:
            if dwpf not in ("M", ""):
                dewpoints.append((valid, float(dwpf)))
        except ValueError:
            pass
    return temps, dewpoints


def todays_actual_high_low(station):
    temps, _ = fetch_asos_today(station)
    if not temps:
        return None, None
    vals = [t[1] for t in temps]
    return max(vals), min(vals)


def afternoon_dewpoint(station):
    """Max dewpoint between 15Z-21Z today, a reasonable proxy for the
    afternoon max dewpoint used in the dewpoint/low-temperature method."""
    _, dewpoints = fetch_asos_today(station)
    afternoon = []
    for valid, val in dewpoints:
        try:
            hour = int(valid.split(" ")[1].split(":")[0])
        except (IndexError, ValueError):
            continue
        if 15 <= hour <= 21:
            afternoon.append(val)
    if afternoon:
        return max(afternoon)
    # fall back to the day's max dewpoint if nothing in the afternoon window
    return max((v for _, v in dewpoints), default=None)


# ----------------------------------------------------------------------
# IEM MOS access
# ----------------------------------------------------------------------


def fetch_mos(station, model="GFS", runtime=None):
    if model.upper() not in VALID_MOS_MODELS:
        raise ValueError(f"MOS model must be one of {sorted(VALID_MOS_MODELS)}")
    params = {"station": station, "model": model.upper()}
    if runtime:
        params["runtime"] = runtime
    r = requests.get(IEM_MOS_URL, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def mos_tmax_tmin_for_date(mos_json, date_str):
    """Pull the MOS tmp/n_x max & min style fields for a given local date
    (YYYY-MM-DD) out of an IEM mos.json payload. IEM stores raw MOS fields;
    'tmp' is the hourly temperature guidance, so day max/min is derived
    across that date's rows."""
    rows = [r for r in mos_json.get("data", []) if r.get("ftime", "").startswith(date_str)]
    temps = [r["tmp"] for r in rows if r.get("tmp") is not None]
    if not temps:
        return None, None
    return max(temps), min(temps)


# ----------------------------------------------------------------------
# Method 1: Persistence
# ----------------------------------------------------------------------


def method_persistence(station):
    hi, lo = todays_actual_high_low(station)
    return {
        "method": "Persistence",
        "forecast_high_f": hi,
        "forecast_low_f": lo,
        "basis": "Today's actual high/low so far, carried unchanged to tomorrow.",
        "caveat": ("Only appropriate for a stagnant pattern: negligible temperature "
                   "advection, similar cloud/snow cover expected tomorrow. If today's "
                   "obs are incomplete (run before end of day), the high/low shown may "
                   "still rise/fall further."),
    }


# ----------------------------------------------------------------------
# Method 2: Delta Method
# ----------------------------------------------------------------------


def method_delta(lat, lon, station, model=None):
    daily = fetch_daily_maxmin(lat, lon, model=model)
    if len(daily["time"]) < 2:
        raise RuntimeError("Not enough model days returned for delta method")
    today_model_hi, tomorrow_model_hi = daily["temperature_2m_max"][0], daily["temperature_2m_max"][1]
    today_model_lo, tomorrow_model_lo = daily["temperature_2m_min"][0], daily["temperature_2m_min"][1]
    delta_hi = tomorrow_model_hi - today_model_hi
    delta_lo = tomorrow_model_lo - today_model_lo

    actual_hi, actual_lo = todays_actual_high_low(station)
    forecast_hi = forecast_lo = None
    if actual_hi is not None:
        forecast_hi = actual_hi + delta_hi
    if actual_lo is not None:
        forecast_lo = actual_lo + delta_lo

    return {
        "method": "Delta Method",
        "model": model or "best-match (Open-Meteo default blend)",
        "model_delta_high_f": round(delta_hi, 1),
        "model_delta_low_f": round(delta_lo, 1),
        "todays_actual_high_f": actual_hi,
        "todays_actual_low_f": actual_lo,
        "forecast_high_f": round(forecast_hi, 1) if forecast_hi is not None else None,
        "forecast_low_f": round(forecast_lo, 1) if forecast_lo is not None else None,
        "basis": ("Same model run's forecast change from today->tomorrow, applied to "
                   "today's ACTUAL high/low (not the model's forecast for today)."),
        "caveat": ("Only valid if weather conditions are reasonably similar today vs. "
                   "tomorrow (same regime, cloud cover, snow cover). Avoid across "
                   "frontal passages or big airmass changes."),
    }


# ----------------------------------------------------------------------
# Method 3: Analogous Thickness
# ----------------------------------------------------------------------

THICKNESS_VARS = [
    "geopotential_height_1000hPa",
    "geopotential_height_500hPa",
    "pressure_msl",
    "wind_direction_500hPa",
    "wind_speed_500hPa",
    "temperature_2m",
]


def find_upstream_analog(lat, lon, valid_hour_utc, search_distances_km=(150, 300, 450, 600, 800, 1000, 1300)):
    """
    Core of the Analogous Thickness Technique:
      1. Get target's FORECAST 1000-500mb thickness & MSLP at tomorrow's
         valid_hour_utc, and the steering (500mb) wind direction/speed right now.
      2. Walk upstream along that wind direction at several distances.
      3. At each candidate point, get today's (24h earlier) thickness/MSLP
         at the same hour, plus its elevation.
      4. Whichever candidate's (thickness, MSLP) most closely matches the
         target's forecast tomorrow is the analog point -- its temperature
         today at that hour becomes the analog forecast for the target
         tomorrow.
    Elevation and steering wind speed are carried back so the caller can
    discount the result per the course's explicit "not well suited for
    different elevations" caveat, and so the synthesis step can gauge
    advection strength for the pattern-regime classification.
    """
    # All datetimes here are naive-UTC on purpose (see nearest_index docstring).
    now = effective_now().replace(tzinfo=None)
    target_date = now.date()
    tomorrow_valid = datetime.combine(target_date, datetime.min.time()) \
        + timedelta(days=1, hours=valid_hour_utc)
    today_same_hour = tomorrow_valid - timedelta(days=1)

    home = fetch_open_meteo(lat, lon, THICKNESS_VARS, past_days=1, forecast_days=2,
                             timezone_mode="UTC")
    h = home["hourly"]
    home_elevation = home.get("elevation")
    idx_tomorrow = nearest_index(h["time"], tomorrow_valid)
    idx_now = nearest_index(h["time"], now)

    target_thick = h["geopotential_height_500hPa"][idx_tomorrow] - h["geopotential_height_1000hPa"][idx_tomorrow]
    target_mslp = h["pressure_msl"][idx_tomorrow]
    steering_dir = h["wind_direction_500hPa"][idx_now]  # direction wind is FROM, right now
    steering_speed = h["wind_speed_500hPa"][idx_now]    # km/h, Open-Meteo default unit

    if target_thick is None or target_mslp is None or steering_dir is None:
        raise RuntimeError("Open-Meteo did not return complete pressure-level data for this location/model.")

    cand_lats, cand_lons, cand_dists = [], [], []
    for dist in search_distances_km:
        clat, clon = destination_point(lat, lon, steering_dir, dist)
        cand_lats.append(clat)
        cand_lons.append(clon)
        cand_dists.append(dist)

    batch = fetch_open_meteo_multi(cand_lats, cand_lons, THICKNESS_VARS,
                                    past_days=1, forecast_days=1, timezone_mode="UTC")

    best = None
    for i, loc in enumerate(batch):
        hh = loc["hourly"]
        idx = nearest_index(hh["time"], today_same_hour)
        thick = hh["geopotential_height_500hPa"][idx] - hh["geopotential_height_1000hPa"][idx]
        mslp = hh["pressure_msl"][idx]
        temp = hh["temperature_2m"][idx]
        if thick is None or mslp is None or temp is None:
            continue
        # Normalize the two error terms roughly onto the same scale before
        # combining: ~1 dam of thickness and ~1 hPa of MSLP are comparable
        # in significance for this purpose.
        score = abs(thick - target_thick) + abs(mslp - target_mslp)
        record = {
            "distance_km": cand_dists[i],
            "lat": round(cand_lats[i], 3),
            "lon": round(cand_lons[i], 3),
            "elevation_m": loc.get("elevation"),
            "thickness_gpm": round(thick, 1),
            "mslp_hpa": round(mslp, 1),
            "temp_f": round(c_to_f(temp), 1),
            "score": round(score, 2),
        }
        if best is None or score < best["score"]:
            best = record

    return {
        "target_forecast_thickness_gpm": round(target_thick, 1),
        "target_forecast_mslp_hpa": round(target_mslp, 1),
        "steering_direction_from_deg": round(steering_dir, 0),
        "steering_speed_kmh": round(steering_speed, 1) if steering_speed is not None else None,
        "home_elevation_m": home_elevation,
        "tomorrow_valid_utc": tomorrow_valid.isoformat() + "Z",
        "today_analog_valid_utc": today_same_hour.isoformat() + "Z",
        "best_analog": best,
    }


# Confidence discount for how far upstream the search had to go to find a
# match. Added after backtesting (30 days, KORD) found Analog's error
# growing with pick distance even on days where elevation was well-matched:
#   <=450 km:  mean error -1.9F, MAE 3.9F   (roughly half the search's picks)
#   600-800km: mean error -3.8F, MAE 4.3F
#   >=1000km:  mean error -5.8F, MAE 5.8F
# The mechanism wasn't pinned down (tested and ruled out: search maxing out
# the distance grid; tested but inconclusive either way: urban/rural land
# cover via reverse geocoding, which mostly returned uninformative county-
# level labels rather than real land-use data) -- but the distance
# correlation itself held up across two independent backtest runs with
# real sample size, so it's used here as an empirical discount regardless
# of cause. Thresholds follow the search grid's own distance steps (see
# DEFAULT_ANALOG_DISTANCES_KM-equivalent search_distances_km above).
DISTANCE_CONFIDENCE_NEAR_KM = 450
DISTANCE_CONFIDENCE_MID_KM = 800


def method_analogous_thickness(lat, lon, valid_hour_utc=18, role="high"):
    result = find_upstream_analog(lat, lon, valid_hour_utc)
    best = result["best_analog"]

    elevation_diff = None
    elev_confidence_mult = 1.0
    elev_note = None
    if best and result.get("home_elevation_m") is not None and best.get("elevation_m") is not None:
        elevation_diff = abs(result["home_elevation_m"] - best["elevation_m"])
        if elevation_diff > 600:
            elev_confidence_mult = 0.25
            elev_note = (f"Analog point is {round(elevation_diff)} m different in elevation -- "
                         "the course explicitly flags this technique as unreliable across large "
                         "elevation differences, so this result is heavily discounted.")
        elif elevation_diff > 300:
            elev_confidence_mult = 0.5
            elev_note = (f"Analog point is {round(elevation_diff)} m different in elevation -- "
                         "moderately discounted per the technique's elevation caveat.")
        else:
            elev_note = f"Analog point elevation is close to target ({round(elevation_diff)} m diff) -- no discount applied."

    distance_confidence_mult = 1.0
    distance_note = None
    if best and best.get("distance_km") is not None:
        dist = best["distance_km"]
        if dist > DISTANCE_CONFIDENCE_MID_KM:
            distance_confidence_mult = 0.25
            distance_note = (f"Analog point is {dist} km upstream -- backtesting found accuracy "
                             "degrades substantially past ~800 km, so this result is heavily discounted.")
        elif dist > DISTANCE_CONFIDENCE_NEAR_KM:
            distance_confidence_mult = 0.5
            distance_note = (f"Analog point is {dist} km upstream -- backtesting found accuracy "
                             "degrading noticeably beyond ~450 km, so this result is moderately discounted.")
        else:
            distance_note = f"Analog point is {dist} km upstream -- within the range backtesting found reliable, no discount applied."

    confidence_mult = elev_confidence_mult * distance_confidence_mult

    label = "Analogous Thickness (High, ~18Z)" if role == "high" else "Analogous Thickness (Low, ~12Z)"

    return {
        "method": label,
        "role": role,
        "valid_time_utc": result["tomorrow_valid_utc"],
        "target_forecast_thickness_gpm": result["target_forecast_thickness_gpm"],
        "target_forecast_mslp_hpa": result["target_forecast_mslp_hpa"],
        "steering_wind_from_deg": result["steering_direction_from_deg"],
        "steering_wind_speed_kmh": result["steering_speed_kmh"],
        "analog_point": best,
        "elevation_diff_m": round(elevation_diff) if elevation_diff is not None else None,
        "elevation_confidence_multiplier": elev_confidence_mult,
        "elevation_note": elev_note,
        "distance_confidence_multiplier": distance_confidence_mult,
        "distance_note": distance_note,
        "confidence_multiplier": confidence_mult,
        "forecast_temp_f": best["temp_f"] if best else None,
        "basis": ("Upstream point (along current 500mb steering flow) whose thickness "
                   "and MSLP today most closely match the target's forecast thickness/"
                   "MSLP at the same clock hour tomorrow; that point's current "
                   "temperature is used as the analog forecast. Run at ~18Z for the "
                   "high and ~12Z for the low, per the course's guidance on which "
                   "check-time to use for each (slide 26)."),
        "caveat": ("Approximate: assumes similar cloud/snow cover, and only searches "
                   "along a single bearing at coarse distance steps. Best in moderate, "
                   "steady advection regimes -- avoid near rapidly strengthening/"
                   "weakening systems. Elevation mismatch (see elevation_note) and pick "
                   "distance (see distance_note) are both now factored into the "
                   "confidence weighting used in the final synthesis."),
    }


# ----------------------------------------------------------------------
# Method 4a: Raw MOS (NOT one of the course's five specified methods)
# ----------------------------------------------------------------------


def method_raw_mos(station, model="GFS", target_date=None):
    """Uncorrected MOS guidance for the target station -- no upstream
    stations, no bias correction, just the model's own MOS bulletin.

    This is NOT one of the course's five specified methods. It's included
    as an optional 5th synthesis contributor after backtesting (120 days
    across two MOS models at KORD, see fit_regime_weights.py --include-raw-
    mos) found it earning real, holdout-validated weight in some pattern
    regimes -- including regimes where it beat its own bias-corrected
    version (MOS Error Advection). It only affects the synthesis when the
    active REGIME_WEIGHTS table gives it nonzero weight for the current
    regime; REGIME_WEIGHTS and REGIME_WEIGHTS_ALT both omit it entirely
    (defaulting to 0 -- see synthesize_forecast's w_for), so this changes
    nothing unless --regime-weights fitted is selected."""
    tomorrow = effective_now().date() + timedelta(days=1)
    if target_date is None:
        target_date = tomorrow.isoformat()

    try:
        mos = fetch_mos(station, model=model)
        hi, lo = mos_tmax_tmin_for_date(mos, target_date)
    except Exception as e:
        return {"method": "Raw MOS", "model": model, "error": str(e)}

    if hi is None:
        return {"method": "Raw MOS", "model": model,
                "error": f"No {model} MOS data available for {target_date}."}

    return {
        "method": "Raw MOS", "model": model,
        "forecast_high_f": hi, "forecast_low_f": lo,
        "basis": f"Uncorrected {model} MOS guidance for {target_date}, taken directly with no "
                 "bias correction and no upstream-station comparison.",
        "caveat": ("Not one of the course's five specified methods -- included only because "
                   "backtesting found it earns real weight in some pattern regimes, sometimes "
                   "outperforming its own bias-corrected version (MOS Error Advection). Has no "
                   "effect unless the active regime-weights table gives it nonzero weight."),
    }


# ----------------------------------------------------------------------
# Method 4: MOS Error Advection
# ----------------------------------------------------------------------


def method_mos_error_advection(station, upstream_stations, model="GFS", target_date=None):
    if not upstream_stations:
        return {
            "method": "MOS Error Advection",
            "skipped": True,
            "reason": ("Requires one or more --upstream-station entries (comma-separated), "
                       "ICAO MOS sites upstream of the target, in the direction the airmass "
                       "is coming from. Run the Analogous Thickness method first to find a "
                       "candidate lat/lon, then look up the nearest MOS-reporting station(s) "
                       "to that point. The course recommends averaging several stations "
                       "rather than using just one."),
        }

    today = effective_now().date()
    tomorrow = today + timedelta(days=1)
    if target_date is None:
        target_date = tomorrow.isoformat()
    today_str = today.isoformat()

    target_mos = fetch_mos(station, model=model)
    target_mos_hi, target_mos_lo = mos_tmax_tmin_for_date(target_mos, target_date)

    station_results = []
    errors_hi, errors_lo = [], []
    for up in upstream_stations:
        try:
            up_mos = fetch_mos(up, model=model)
            up_mos_hi, up_mos_lo = mos_tmax_tmin_for_date(up_mos, today_str)
            up_actual_hi, up_actual_lo = todays_actual_high_low(up)
            if up_mos_hi is None or up_actual_hi is None:
                station_results.append({"station": up, "error": "Missing MOS or observation data for today."})
                continue
            err_hi = up_actual_hi - up_mos_hi
            err_lo = (up_actual_lo - up_mos_lo) if (up_actual_lo is not None and up_mos_lo is not None) else None
            station_results.append({
                "station": up,
                "mos_high_f": up_mos_hi,
                "actual_high_f": up_actual_hi,
                "error_high_f": round(err_hi, 1),
                "error_low_f": round(err_lo, 1) if err_lo is not None else None,
            })
            errors_hi.append(err_hi)
            if err_lo is not None:
                errors_lo.append(err_lo)
        except Exception as e:
            station_results.append({"station": up, "error": str(e)})

    if not errors_hi or target_mos_hi is None:
        return {
            "method": "MOS Error Advection",
            "skipped": True,
            "reason": "No usable upstream station errors, or missing target MOS forecast.",
            "station_results": station_results,
        }

    avg_error_hi = sum(errors_hi) / len(errors_hi)
    avg_error_lo = sum(errors_lo) / len(errors_lo) if errors_lo else None

    forecast_hi = target_mos_hi + avg_error_hi
    forecast_lo = (target_mos_lo + avg_error_lo) if (target_mos_lo is not None and avg_error_lo is not None) else None

    return {
        "method": "MOS Error Advection",
        "model": model,
        "upstream_stations": upstream_stations,
        "station_results": station_results,
        "avg_upstream_error_high_f": round(avg_error_hi, 1),
        "avg_upstream_error_low_f": round(avg_error_lo, 1) if avg_error_lo is not None else None,
        "error_spread_high_f": round(max(errors_hi) - min(errors_hi), 1) if len(errors_hi) > 1 else 0.0,
        "target_mos_high_f": target_mos_hi,
        "target_mos_low_f": target_mos_lo,
        "forecast_high_f": round(forecast_hi, 1),
        "forecast_low_f": round(forecast_lo, 1) if forecast_lo is not None else None,
        "basis": (f"Average of today's {model} MOS error (actual - MOS) across "
                  f"{len(errors_hi)} upstream station(s), applied as a correction to "
                  "tomorrow's MOS forecast at the target station."),
        "caveat": ("Only valid if upstream and target regions have similar cloud cover, "
                   "precipitation, snow cover, and elevation. Errors are averaged across "
                   "all upstream stations you supply -- pass several (comma-separated) "
                   "for a more reliable areal average, per the course's guidance."),
    }


# ----------------------------------------------------------------------
# Method 5: Dewpoint method for lows
# ----------------------------------------------------------------------


def method_dewpoint_low(station, lat, lon, subtract_f=2.0):
    dp = afternoon_dewpoint(station)
    if dp is None:
        return {"method": "Dewpoint Method", "skipped": True, "reason": "No dewpoint obs available."}

    confidence_weight = 1.0
    confidence_note = "Overnight wind/cloud forecast unavailable -- assuming favorable conditions."
    overnight = None
    try:
        overnight = fetch_overnight_conditions(lat, lon)
    except Exception:
        overnight = None

    if overnight:
        wind = overnight["avg_wind_speed_mph"]
        cloud = overnight["avg_cloud_cover_pct"]
        if wind < 8 and cloud < 40:
            confidence_weight = 1.0
            confidence_note = (f"Favorable: light wind ({wind} mph), mostly clear ({cloud}% cloud) "
                                "-- the method's assumptions likely hold.")
        elif wind < 15 and cloud < 70:
            confidence_weight = 0.6
            confidence_note = (f"Marginal: wind {wind} mph, cloud cover {cloud}% -- some "
                                "radiative-cooling loss expected; moderately discounted.")
        else:
            confidence_weight = 0.2
            confidence_note = (f"Unfavorable: wind {wind} mph and/or cloud cover {cloud}% -- "
                                "the method's clear-sky/light-wind assumption is violated; "
                                "heavily discounted in the synthesis.")

    return {
        "method": "Dewpoint Method",
        "afternoon_dewpoint_f": round(dp, 1),
        "subtracted_f": subtract_f,
        "forecast_low_f": round(dp - subtract_f, 1),
        "overnight_conditions": overnight,
        "confidence_weight": confidence_weight,
        "confidence_note": confidence_note,
        "basis": "Tonight's low ~= today's afternoon dewpoint minus 0-3F (default 2F).",
        "caveat": ("Assumes good radiative cooling (clear skies, light wind), roughly "
                   "constant dewpoint overnight, and minimal temperature advection. Not "
                   "valid on windy, cloudy, or frontal nights -- see confidence_note."),
    }


# ----------------------------------------------------------------------
# Pattern regime assessment (drives the weighted synthesis)
# ----------------------------------------------------------------------


def assess_pattern_regime(model_delta_hi_f, model_delta_lo_f, steering_speed_kmh):
    """Classify today->tomorrow as stagnant / moderate change / rapidly
    changing, using two signals the course explicitly calls out as
    indicators that persistence/delta-style methods start breaking down:
    the SIZE of the model's own day-over-day temperature change, and the
    speed of the 500mb steering flow (fast flow = more advection)."""

    def severity_from_delta(d):
        if d is None:
            return 0
        d = abs(d)
        if d < 3:
            return 0
        if d <= 7:
            return 1
        return 2

    def severity_from_wind(w):
        if w is None:
            return 0
        if w < 30:   # ~<19 mph: fairly slack steering flow
            return 0
        if w <= 60:  # ~19-37 mph: moderate flow
            return 1
        return 2     # >~37 mph: strong flow, likely active pattern

    sev = max(
        severity_from_delta(model_delta_hi_f),
        severity_from_delta(model_delta_lo_f),
        severity_from_wind(steering_speed_kmh),
    )
    label = {0: "stagnant", 1: "moderate change", 2: "rapidly changing"}[sev]

    return {
        "regime": label,
        "severity": sev,
        "model_delta_high_f": model_delta_hi_f,
        "model_delta_low_f": model_delta_lo_f,
        "steering_wind_speed_kmh": steering_speed_kmh,
        "steering_wind_speed_mph": round(kmh_to_mph(steering_speed_kmh), 1) if steering_speed_kmh is not None else None,
        "basis": ("Classified from the Delta Method's day-over-day model temperature "
                  "change and the current 500mb steering wind speed -- larger changes in "
                  "either signal a less-stagnant pattern (frontal passage/advection), per "
                  "the course's persistence/delta caveats."),
    }


# ----------------------------------------------------------------------
# Report / CLI
# ----------------------------------------------------------------------


def synthesize_forecast(report, weights_table=None):
    """Combine whichever methods produced usable numbers into a single
    weighted best-estimate high/low. Unlike a plain average, each method's
    contribution is weighted by:
      - how well it fits the assessed pattern regime (persistence/delta vs.
        analog/MOS-error-advection -- see REGIME_WEIGHTS, or pass
        weights_table to use REGIME_WEIGHTS_ALT / REGIME_WEIGHTS_FITTED /
        a custom table instead), and
      - method-specific confidence adjustments (elevation mismatch for
        Analogous Thickness; overnight wind/cloud for the Dewpoint Method).
    This mirrors the course's own instruction to reason about which methods
    actually apply today rather than trusting all of them equally."""

    if weights_table is None:
        weights_table = REGIME_WEIGHTS
    regime_info = report.get("pattern_regime") or {}
    weights = weights_table.get(regime_info.get("regime"), {})

    def w_for(key):
        # Fall back to equal weighting (old behavior) if regime is unknown --
        # except raw_mos, which defaults to 0 (excluded) when a table has no
        # opinion on it, since it's an optional 5th contributor outside the
        # course's five specified methods, not one of the original four.
        return weights.get(key, 0.0 if key == "raw_mos" else 1.0)

    high_contributors = []
    low_contributors = []

    for key in ["persistence", "delta", "mos_error_advection", "raw_mos"]:
        d = report.get(key, {})
        if d.get("error") or d.get("skipped"):
            continue
        hi = d.get("forecast_high_f")
        lo = d.get("forecast_low_f")
        weight = w_for(key)
        if hi is not None and weight > 0:
            high_contributors.append((d.get("method", key), hi, weight))
        if lo is not None and weight > 0:
            low_contributors.append((d.get("method", key), lo, weight))

    # Analogous Thickness now runs twice: once at ~18Z for the high, once at
    # ~12Z for the low (per the course's own guidance on which check-time to
    # use for each), so each contributes directly to its matching bucket
    # rather than being guessed from the valid hour.
    at_hi = report.get("analogous_thickness", {})
    if not (at_hi.get("error") or at_hi.get("skipped")):
        t = at_hi.get("forecast_temp_f")
        # confidence_multiplier combines the elevation-mismatch discount with
        # the pick-distance discount (both computed in method_analogous_thickness).
        weight = w_for("analog") * at_hi.get("confidence_multiplier", 1.0)
        if t is not None and weight > 0:
            high_contributors.append((at_hi.get("method", "analogous_thickness"), t, weight))

    at_lo = report.get("analogous_thickness_low", {})
    if not (at_lo.get("error") or at_lo.get("skipped")):
        t = at_lo.get("forecast_temp_f")
        weight = w_for("analog") * at_lo.get("confidence_multiplier", 1.0)
        if t is not None and weight > 0:
            low_contributors.append((at_lo.get("method", "analogous_thickness_low"), t, weight))

    dp = report.get("dewpoint_low", {})
    if not (dp.get("error") or dp.get("skipped")):
        lo = dp.get("forecast_low_f")
        weight = dp.get("confidence_weight", 1.0)
        if lo is not None and weight > 0:
            low_contributors.append((dp.get("method", "dewpoint_low"), lo, weight))

    def summarize(contributors):
        if not contributors:
            return None
        vals = [v for _, v, _ in contributors]
        ws = [w for _, _, w in contributors]
        weighted_val = sum(v * w for v, w in zip(vals, ws)) / sum(ws)
        return {
            "value_f": round(weighted_val, 1),
            "n_methods": len(vals),
            "methods": [f"{name} (weight {w:.2f})" for name, _, w in contributors],
            "range_f": [round(min(vals), 1), round(max(vals), 1)],
            "spread_f": round(max(vals) - min(vals), 1),
        }

    return {
        "high": summarize(high_contributors),
        "low": summarize(low_contributors),
        "weights_used": weights or "equal (regime unknown -- delta/analog data unavailable)",
    }


def build_report(args):
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "station": args.station,
              "lat": args.lat, "lon": args.lon}

    try:
        report["persistence"] = method_persistence(args.station)
    except Exception as e:
        report["persistence"] = {"method": "Persistence", "error": str(e)}

    try:
        report["delta"] = method_delta(args.lat, args.lon, args.station, model=args.model)
    except Exception as e:
        report["delta"] = {"method": "Delta Method", "error": str(e)}

    try:
        report["analogous_thickness"] = method_analogous_thickness(args.lat, args.lon, args.valid_hour, role="high")
    except Exception as e:
        report["analogous_thickness"] = {"method": "Analogous Thickness (High)", "error": str(e)}

    try:
        report["analogous_thickness_low"] = method_analogous_thickness(
            args.lat, args.lon, args.low_valid_hour, role="low")
    except Exception as e:
        report["analogous_thickness_low"] = {"method": "Analogous Thickness (Low)", "error": str(e)}

    upstream_stations = [s.strip() for s in args.upstream_station.split(",") if s.strip()] \
        if args.upstream_station else []
    try:
        report["mos_error_advection"] = method_mos_error_advection(
            args.station, upstream_stations, model=args.mos_model)
    except Exception as e:
        report["mos_error_advection"] = {"method": "MOS Error Advection", "error": str(e)}

    try:
        report["raw_mos"] = method_raw_mos(args.station, model=args.mos_model)
    except Exception as e:
        report["raw_mos"] = {"method": "Raw MOS", "error": str(e)}

    try:
        report["dewpoint_low"] = method_dewpoint_low(
            args.station, args.lat, args.lon, subtract_f=args.dewpoint_subtract)
    except Exception as e:
        report["dewpoint_low"] = {"method": "Dewpoint Method", "error": str(e)}

    delta_d = report.get("delta", {})
    analog_d = report.get("analogous_thickness", {})
    report["pattern_regime"] = assess_pattern_regime(
        delta_d.get("model_delta_high_f"),
        delta_d.get("model_delta_low_f"),
        analog_d.get("steering_wind_speed_kmh"),
    )

    weights_name = getattr(args, "regime_weights", "current") or "current"
    weights_table = REGIME_WEIGHTS_TABLES.get(weights_name, REGIME_WEIGHTS)
    report["synthesis"] = synthesize_forecast(report, weights_table=weights_table)
    report["regime_weights_used"] = weights_name

    return report


# ----------------------------------------------------------------------
# Batch mode: run the same forecast across a list of cities
# ----------------------------------------------------------------------

# Built-in city list -- edit freely, or pass your own via --cities-json.
DEFAULT_CITIES = [
    {"name": "Los Angeles, CA",    "lat": 34.0522, "lon": -118.2437, "station": "KLAX"},
    {"name": "New York City, NY",  "lat": 40.7128, "lon": -74.0060,  "station": "KNYC"},
    {"name": "Miami, FL",          "lat": 25.7617, "lon": -80.1918,  "station": "KMIA"},
    {"name": "Chicago, IL",        "lat": 41.8781, "lon": -87.6298,  "station": "KORD"},
    {"name": "Dallas, TX",         "lat": 32.7767, "lon": -96.7970,  "station": "KDFW"},
    {"name": "Austin, TX",         "lat": 30.2672, "lon": -97.7431,  "station": "KAUS"},
    {"name": "Philadelphia, PA",   "lat": 39.9526, "lon": -75.1652,  "station": "KPHL"},
    {"name": "Houston, TX",        "lat": 29.7604, "lon": -95.3698,  "station": "KIAH"},
    {"name": "Denver, CO",         "lat": 39.7392, "lon": -104.9903, "station": "KDEN"},
    {"name": "Phoenix, AZ",        "lat": 33.4484, "lon": -112.0740, "station": "KPHX"},
    {"name": "Atlanta, GA",        "lat": 33.7490, "lon": -84.3880,  "station": "KATL"},
    {"name": "San Francisco, CA",  "lat": 37.7749, "lon": -122.4194, "station": "KSFO"},
    {"name": "Minneapolis, MN",    "lat": 44.9778, "lon": -93.2650,  "station": "KMSP"},
    {"name": "Washington, DC",     "lat": 38.9072, "lon": -77.0369,  "station": "KDCA"},
    {"name": "Oklahoma City, OK",  "lat": 35.4676, "lon": -97.5164,  "station": "KOKC"},
    {"name": "Seattle, WA",        "lat": 47.6062, "lon": -122.3321, "station": "KSEA"},
    {"name": "Las Vegas, NV",      "lat": 36.1699, "lon": -115.1398, "station": "KLAS"},
    {"name": "Boston, MA",         "lat": 42.3601, "lon": -71.0589,  "station": "KBOS"},
    {"name": "New Orleans, LA",    "lat": 29.9511, "lon": -90.0715,  "station": "KMSY"},
    {"name": "San Antonio, TX",    "lat": 29.4241, "lon": -98.4936,  "station": "KSAT"},
]


def run_all_cities(cities=None, model=None, mos_model="GFS", valid_hour=18, low_valid_hour=12,
                    dewpoint_subtract=2.0, upstream_station=None, regime_weights="current"):
    """Run build_report() for each city in the list and return a list of
    {name, station, lat, lon, report} dicts. Network/API failures for one
    city never abort the rest -- they show up as an 'error' key on that
    city's report and print as n/a in the summary table."""
    cities = cities if cities is not None else DEFAULT_CITIES
    results = []
    for city in cities:
        ns = argparse.Namespace(
            lat=city["lat"], lon=city["lon"], station=city["station"],
            model=model, mos_model=mos_model, valid_hour=valid_hour, low_valid_hour=low_valid_hour,
            dewpoint_subtract=dewpoint_subtract, upstream_station=upstream_station,
            regime_weights=regime_weights,
        )
        try:
            print(f"Starting {city['name']}...", file=sys.stderr, flush=True)
            report = build_report(ns)
            print(f"  finished {city['name']}", file=sys.stderr, flush=True)
        except Exception as e:
            report = {"error": str(e)}
        results.append({
            "name": city["name"], "station": city["station"],
            "lat": city["lat"], "lon": city["lon"], "report": report,
        })
    return results


def print_batch_summary(results):
    weights_name = next((r["report"].get("regime_weights_used") for r in results
                         if not r["report"].get("error")), "current")
    print("=" * 78)
    print(f"TOMORROW'S MOST LIKELY HIGH / LOW  (generated {datetime.now(timezone.utc).isoformat()})")
    print(f"{len(results)} cities -- each value is the regime-weighted combination of every")
    print("method that produced a usable number for that city (see README for how the")
    print(f"weighting works). regime_weights: {weights_name}. Run the tool for a single "
          "city to see the full breakdown.")
    print("=" * 78)
    print(f"{'City':<22}{'Station':<9}{'High':>7}{'Low':>7}   {'Pattern'}")
    print("-" * 78)
    for r in results:
        rep = r["report"]
        if rep.get("error"):
            print(f"{r['name']:<22}{r['station']:<9}{'n/a':>7}{'n/a':>7}   ERROR: {rep['error']}")
            continue
        syn = rep.get("synthesis", {})
        high, low = syn.get("high"), syn.get("low")
        regime = rep.get("pattern_regime", {}).get("regime", "n/a")
        hi_str = f"{high['value_f']:.0f}\u00b0F" if high else "n/a"
        lo_str = f"{low['value_f']:.0f}\u00b0F" if low else "n/a"
        print(f"{r['name']:<22}{r['station']:<9}{hi_str:>7}{lo_str:>7}   {regime}")
    print("=" * 78)


def print_human_report(report):
    def line(*a):
        print(*a)

    line("=" * 70)
    line(f"TEMPERATURE FORECAST REPORT  (generated {report['generated_utc']})")
    line(f"Station: {report['station']}   Lat/Lon: {report['lat']}, {report['lon']}")
    line("=" * 70)

    for key in ["persistence", "delta", "analogous_thickness", "analogous_thickness_low",
                "mos_error_advection", "raw_mos", "dewpoint_low"]:
        d = report.get(key, {})
        line()
        line(f"--- {d.get('method', key)} ---")
        if d.get("error"):
            line(f"  ERROR: {d['error']}")
            continue
        if d.get("skipped"):
            line(f"  SKIPPED: {d['reason']}")
            continue
        for k, v in d.items():
            if k in ("method",):
                continue
            line(f"  {k}: {v}")

    regime = report.get("pattern_regime", {})
    line()
    line("=" * 70)
    line("PATTERN REGIME ASSESSMENT")
    line("=" * 70)
    line(f"  Regime: {regime.get('regime', 'unknown')}")
    line(f"  Model day-over-day delta (high/low): "
         f"{regime.get('model_delta_high_f')}\u00b0F / {regime.get('model_delta_low_f')}\u00b0F")
    line(f"  500mb steering wind speed: {regime.get('steering_wind_speed_mph')} mph")
    line(f"  {regime.get('basis', '')}")

    syn = report.get("synthesis", {})
    high, low = syn.get("high"), syn.get("low")

    line()
    line("=" * 70)
    weights_name = report.get("regime_weights_used", "current")
    line(f"MOST LIKELY FORECAST (weighted by pattern regime, elevation, and overnight "
         f"conditions -- regime_weights: {weights_name})")
    line("=" * 70)
    if high:
        line(f"  High: {high['value_f']}\u00b0F  "
             f"(weighted from {high['n_methods']} method(s): {', '.join(high['methods'])}; "
             f"range {high['range_f'][0]}-{high['range_f'][1]}\u00b0F)")
    else:
        line("  High: no methods produced a usable number -- check the errors above.")
    if low:
        line(f"  Low:  {low['value_f']}\u00b0F  "
             f"(weighted from {low['n_methods']} method(s): {', '.join(low['methods'])}; "
             f"range {low['range_f'][0]}-{low['range_f'][1]}\u00b0F)")
    else:
        line("  Low:  no methods produced a usable number -- check the errors above.")

    if (high and high["spread_f"] >= 5) or (low and low["spread_f"] >= 5):
        line()
        line("  NOTE: methods disagree by 5\u00b0F or more even after weighting -- that's a sign")
        line("  worth double-checking the regime call above against the actual synoptic")
        line("  picture (maps, satellite, radar) rather than trusting the number blindly.")

    line()
    line("This is a regime-weighted combination, not a plain average: methods that fit")
    line("today's pattern (see PATTERN REGIME ASSESSMENT) count more; Analogous Thickness")
    line("is discounted if its analog point's elevation differs a lot from the target's;")
    line("the Dewpoint Method is discounted if tomorrow morning looks windy or cloudy.")
    line("It's still a starting point, not gospel -- per the course's own process, check")
    line("MOS against this, and reason about whether your first guess or MOS is more")
    line("likely wrong today.")
    line("=" * 70)


# ----------------------------------------------------------------------
# Timing check: is now a good time to run the forecast?
# ----------------------------------------------------------------------

MODEL_CYCLES_UTC = [0, 6, 12, 18]
CYCLE_AVAILABILITY_BUFFER_HOURS = 4  # rough wait after a cycle time before guidance is fully posted
DEWPOINT_WINDOW_START_UTC = 15
DEWPOINT_WINDOW_END_UTC = 21


def timing_assessment(lat=None, lon=None, station=None, cities=None, now=None):
    """Assess whether right now is a good time to run the tool, based on:
      - freshness of the most recent GFS/NAM-style model cycle (00/06/12/18Z)
        vs. the ~4h it typically takes Open-Meteo/IEM to post it,
      - whether the tool's fixed 15Z-21Z 'afternoon dewpoint' window (used by
        the Dewpoint Method) has started/finished yet today, and
      - a rough, longitude-based estimate of whether each target's local
        afternoon has happened yet (Persistence/Delta read today's actual
        high, which isn't final until the local afternoon peak has passed).
    This is deliberately approximate -- civil time zones don't track
    longitude exactly -- but it's enough to flag "too early" runs."""
    now = now or datetime.now(timezone.utc)
    info, warnings = [], []

    # --- Model cycle freshness ---
    past_cycles = [c for c in MODEL_CYCLES_UTC if c <= now.hour]
    if past_cycles:
        last_cycle = max(past_cycles)
        hours_since_cycle = (now.hour - last_cycle) + now.minute / 60.0
    else:
        last_cycle = MODEL_CYCLES_UTC[-1]  # yesterday's 18Z
        hours_since_cycle = (now.hour + 24 - last_cycle) + now.minute / 60.0
    hours_since_cycle = round(hours_since_cycle, 1)
    cycle_fresh = hours_since_cycle >= CYCLE_AVAILABILITY_BUFFER_HOURS

    info.append(f"Current time: {now.strftime('%H:%M UTC')} on {now.date().isoformat()}")
    info.append(f"Most recent model cycle: {last_cycle:02d}Z ({hours_since_cycle}h ago)")
    if cycle_fresh:
        info.append(f"That's past the ~{CYCLE_AVAILABILITY_BUFFER_HOURS}h buffer -- this cycle's "
                     "guidance should be fully posted to Open-Meteo/IEM by now.")
    else:
        warnings.append(f"Only {hours_since_cycle}h since the {last_cycle:02d}Z model cycle -- "
                         f"Open-Meteo/IEM may still be ingesting it. Consider waiting until "
                         f"~{CYCLE_AVAILABILITY_BUFFER_HOURS}h after a cycle time "
                         "(00Z/06Z/12Z/18Z) for fresher data.")

    # --- Fixed UTC dewpoint window used by the Dewpoint Method ---
    if now.hour < DEWPOINT_WINDOW_START_UTC:
        warnings.append(f"It's before {DEWPOINT_WINDOW_START_UTC}Z -- today's "
                         f"{DEWPOINT_WINDOW_START_UTC}Z-{DEWPOINT_WINDOW_END_UTC}Z afternoon-dewpoint "
                         "window hasn't started, so the Dewpoint Method will have no reading yet for "
                         "any station whose local afternoon falls in that window.")
    elif now.hour < DEWPOINT_WINDOW_END_UTC:
        warnings.append(f"The {DEWPOINT_WINDOW_START_UTC}Z-{DEWPOINT_WINDOW_END_UTC}Z afternoon-dewpoint "
                         "window is still in progress -- the Dewpoint Method may be using a partial "
                         "reading if you run right now.")
    else:
        info.append(f"The {DEWPOINT_WINDOW_START_UTC}Z-{DEWPOINT_WINDOW_END_UTC}Z afternoon-dewpoint "
                     "window has already completed for today.")
    info.append(f"Note: this window is a FIXED UTC range, not adjusted per station -- for "
                 "Mountain/Pacific stations it actually falls in their late morning, not their "
                 "afternoon, so the Dewpoint Method is structurally weaker for western stations "
                 "regardless of when you run the tool.")

    # --- Rough per-location local-afternoon completeness (for Persistence/Delta) ---
    # IMPORTANT: the tool's ASOS pull only covers the CURRENT UTC calendar day
    # (00Z of today through now) -- see fetch_asos_today. So what matters isn't
    # "what does the local clock read right now" (that's always answerable and
    # doesn't tell you what's in today's UTC-dated data), it's "has enough of
    # TODAY's UTC calendar day elapsed to include the local ~3pm afternoon peak
    # at this longitude." For continental US longitudes, local 3pm maps to a
    # UTC hour-of-day between roughly 19 and 23, comfortably below 24, so this
    # threshold never has to wrap into a second day -- comparing now's elapsed
    # UTC hours directly against it (no modulo) is the correct check. Using a
    # mod-24 "local clock time" comparison instead silently gives a false
    # "complete" reading in the hours right after a fresh UTC day starts (e.g.
    # US evenings), because the local clock reads "evening" while the actual
    # afternoon peak it's describing occurred under YESTERDAY's UTC date --
    # outside the window the tool actually queries.
    targets = []
    if cities:
        targets = [(c["name"], c["lon"]) for c in cities]
    elif lon is not None:
        targets = [(station or "target location", lon)]

    now_elapsed_hours = now.hour + now.minute / 60.0
    incomplete = []
    for name, lon_v in targets:
        threshold_utc_hour = 15.0 - (lon_v / 15.0)  # UTC hour-of-day for local ~3pm
        if now_elapsed_hours < threshold_utc_hour:
            incomplete.append((name, round(threshold_utc_hour, 1)))

    if targets:
        if incomplete:
            names = ", ".join(f"{n} (needs {h}Z)" for n, h in incomplete[:8])
            more = f" and {len(incomplete) - 8} more" if len(incomplete) > 8 else ""
            warnings.append("Today's UTC calendar day hasn't yet reached these locations' local "
                             "~3pm-equivalent UTC hour, so Persistence/Delta's 'today's actual high' "
                             f"isn't in the data yet for: {names}{more}.")
        else:
            info.append("Today's UTC calendar day has already passed all target location(s)' "
                         "local ~3pm-equivalent UTC hour -- today's actual high should be in the data.")

    good_time = cycle_fresh and now.hour >= DEWPOINT_WINDOW_END_UTC and not incomplete

    return {
        "now_utc": now.isoformat(),
        "last_cycle_utc": last_cycle,
        "hours_since_cycle": hours_since_cycle,
        "cycle_fresh": cycle_fresh,
        "dewpoint_window_state": (
            "not_started" if now.hour < DEWPOINT_WINDOW_START_UTC
            else "in_progress" if now.hour < DEWPOINT_WINDOW_END_UTC
            else "complete"
        ),
        "incomplete_local_afternoon": incomplete,
        "good_time_to_run": good_time,
        "info": info,
        "warnings": warnings,
    }


def print_timing_report(assessment):
    print("=" * 70)
    print("TIMING CHECK -- is now a good time to run the forecast?")
    print("=" * 70)
    for line in assessment["info"]:
        print(f"  {line}")
    print()
    if assessment["warnings"]:
        print("  WARNINGS:")
        for w in assessment["warnings"]:
            print(f"  - {w}")
    else:
        print("  No timing concerns detected.")
    print()
    if assessment["good_time_to_run"]:
        print("  VERDICT: good time to run -- model cycle is fresh, the dewpoint window has")
        print("  completed, and today's UTC date has already passed target location(s)'")
        print("  local ~3pm-equivalent hour.")
    else:
        print("  VERDICT: you can still run now, but expect some methods to be less reliable")
        print("  (see warnings above). Because the tool resets its 'today' data at UTC")
        print("  midnight, there's a genuinely narrow nightly window where EVERY continental")
        print("  US city's local afternoon has landed in today's UTC-dated data but that data")
        print("  hasn't rolled over yet: roughly 23:10-23:59 UTC (~7-8pm Eastern during EDT,")
        print("  ~6-7pm Eastern during EST -- Seattle/San Francisco set the limiting threshold).")
        print("  For a single, more easterly city that window opens earlier -- see this")
        print("  city's/these cities' own threshold(s) above.")
    print("=" * 70)


# ----------------------------------------------------------------------
# Email delivery (for scheduled/unattended runs, e.g. via cron)
# ----------------------------------------------------------------------

# Personal default recipient -- override any time with --email-to.
DEFAULT_EMAIL_TO = "megan.christina.geyer@gmail.com"


def render_human_report_to_string(report):
    """Capture print_human_report()'s output as a string, for use as an
    email body, without duplicating its formatting logic."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_human_report(report)
    return buf.getvalue()


def format_batch_email_body(results):
    lines = []
    lines.append(f"Tomorrow's most likely high/low -- generated {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"{len(results)} cities -- regime-weighted combination of every method that")
    lines.append("produced a usable number for that city.")
    lines.append("")
    lines.append(f"{'City':<22}{'Station':<9}{'High':>7}{'Low':>7}   Pattern")
    lines.append("-" * 70)
    for r in results:
        rep = r["report"]
        if rep.get("error"):
            lines.append(f"{r['name']:<22}{r['station']:<9}{'n/a':>7}{'n/a':>7}   ERROR: {rep['error']}")
            continue
        syn = rep.get("synthesis", {})
        high, low = syn.get("high"), syn.get("low")
        regime = rep.get("pattern_regime", {}).get("regime", "n/a")
        hi_str = f"{high['value_f']:.0f}F" if high else "n/a"
        lo_str = f"{low['value_f']:.0f}F" if low else "n/a"
        lines.append(f"{r['name']:<22}{r['station']:<9}{hi_str:>7}{lo_str:>7}   {regime}")
    lines.append("-" * 70)

    # Diagnostics: for any city with an n/a high or low, list which methods
    # failed and why, so the cause is visible without digging through logs.
    method_keys = ["persistence", "delta", "analogous_thickness", "analogous_thickness_low",
                   "mos_error_advection", "raw_mos", "dewpoint_low"]
    diag = []
    for r in results:
        rep = r["report"]
        if rep.get("error"):
            continue
        syn = rep.get("synthesis", {})
        if syn.get("high") and syn.get("low"):
            continue
        problems = []
        for k in method_keys:
            d = rep.get(k, {})
            if d.get("error"):
                problems.append(f"    {d.get('method', k)}: {str(d['error'])[:160]}")
        if problems:
            diag.append(f"  {r['name']} ({r['station']}):")
            diag.extend(problems)
    if diag:
        lines.append("")
        lines.append("DIAGNOSTICS (methods that failed for cities with an n/a):")
        lines.extend(diag)

    lines.append("")
    lines.append("Run with --check-timing to see whether this run happened at a reliable time")
    lines.append("of day for the Persistence/Delta/Dewpoint methods (they depend on today's")
    lines.append("actual observations being complete; Analogous Thickness does not).")
    return "\n".join(lines)


def _check_ascii_credential(name, value):
    """Catch a common macOS gotcha before it becomes a cryptic
    UnicodeEncodeError deep inside smtplib: pasting a password (or the
    quotes around it) from an app with 'smart quotes' enabled (Notes,
    TextEdit, Pages, etc.) silently swaps straight quotes (" ') for curly
    ones (\u201c \u201d \u2018 \u2019), which then get treated as literal
    characters INSIDE the value by the shell -- corrupting it. (The other
    common culprit, non-breaking spaces copied from Google's App Passwords
    page, is already stripped out before this check runs -- see
    send_email.)"""
    for i, ch in enumerate(value):
        if ord(ch) > 127:
            raise RuntimeError(
                f"{name} contains a non-standard character ({ch!r} at position {i}) that's "
                "not valid here. This usually means a curly/smart quote ended up inside the "
                "value -- typically from pasting a password (or the quotes around it) out of "
                "Notes, TextEdit, Pages, or similar, which silently swaps straight quotes for "
                "curly ones. Fix: retype the export/crontab line directly in Terminal, and "
                "either skip quotes entirely around the password (App Passwords have no "
                "spaces or punctuation, so quotes aren't needed) or type the quote characters "
                "yourself in Terminal rather than pasting them."
            )


def send_email(subject, body, to_addr, from_addr=None, app_password=None):
    """Send a plain-text email via Gmail's SMTP relay. Requires a Gmail App
    Password (NOT the account's normal login password) -- generate one at
    https://myaccount.google.com/apppasswords after enabling 2-Step
    Verification. Never hardcode the password in this file; it's read from
    the GMAIL_APP_PASSWORD environment variable (or the app_password
    argument, for programmatic use)."""
    from_addr = from_addr or os.environ.get("GMAIL_ADDRESS") or to_addr
    app_password = app_password or os.environ.get("GMAIL_APP_PASSWORD")
    if not app_password:
        raise RuntimeError(
            "No Gmail App Password found. Set the GMAIL_APP_PASSWORD environment variable "
            "to an App Password generated at https://myaccount.google.com/apppasswords "
            "(requires 2-Step Verification to be enabled on that Google account)."
        )
    # Google's App Passwords page displays the password with spaces every 4
    # characters purely for readability (the real value has none). Copying
    # directly off that page often carries those "spaces" over as non-
    # breaking spaces (U+00A0) rather than regular ones, which end up
    # embedded in the value. Strip ALL whitespace (regular and otherwise)
    # before use -- this is always safe since a real App Password never
    # contains whitespace.
    app_password = re.sub(r"\s+", "", app_password)
    _check_ascii_credential("GMAIL_APP_PASSWORD", app_password)
    _check_ascii_credential("--email-from / GMAIL_ADDRESS", from_addr)
    _check_ascii_credential("--email-to", to_addr)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    # Prefer certifi's CA bundle when available: on macOS, Python installed
    # from python.org doesn't wire itself up to the system's trusted roots,
    # which otherwise surfaces as ssl.SSLCertVerificationError here. certifi
    # ships its own bundle and sidesteps that regardless of how Python was
    # installed; fall back to the system default if certifi isn't present.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(from_addr, app_password)
        server.sendmail(from_addr, [to_addr], msg.as_string())


def main():
    p = argparse.ArgumentParser(description="Multi-method temperature forecasting tool")
    p.add_argument("--lat", type=float, default=None, help="Target latitude")
    p.add_argument("--lon", type=float, default=None, help="Target longitude")
    p.add_argument("--station", default=None, help="Target ICAO station id (e.g. KORD) for obs/MOS")
    p.add_argument("--all-cities", action="store_true",
                    help="Ignore --lat/--lon/--station and instead run the forecast for the "
                         "built-in list of major US cities (see DEFAULT_CITIES in the script), "
                         "printing a compact high/low table for tomorrow instead of a full report.")
    p.add_argument("--check-timing", action="store_true",
                    help="Print guidance on whether right now is a good time to run the forecast "
                         "(model-cycle freshness, dewpoint-window completeness, local-afternoon "
                         "completeness), then exit without fetching any forecast data. Combine "
                         "with --all-cities to check across the whole city list, or with "
                         "--lat/--lon for a single location.")
    p.add_argument("--model", default=None,
                    help="Open-Meteo model id for the Delta Method, e.g. gfs_seamless, "
                         "nam_conus, icon_seamless (default: Open-Meteo's best-match blend)")
    p.add_argument("--mos-model", default="GFS", choices=sorted(VALID_MOS_MODELS),
                    help="MOS model for MOS Error Advection (default GFS)")
    p.add_argument("--upstream-station", default=None,
                    help="One or more ICAO MOS stations upstream of the target, comma-separated "
                         "(e.g. KOMA,KDSM,KFSD). Errors are averaged across all of them, per the "
                         "course's guidance not to rely on a single station. Run without this "
                         "first, use the Analogous Thickness output's analog lat/lon to pick "
                         "candidates, then re-run.")
    p.add_argument("--valid-hour", type=int, default=18,
                    help="UTC hour tomorrow to forecast for Analogous Thickness's HIGH estimate (default 18Z)")
    p.add_argument("--low-valid-hour", type=int, default=12,
                    help="UTC hour tomorrow morning to forecast for Analogous Thickness's LOW "
                         "estimate (course recommends 12Z, default 12)")
    p.add_argument("--dewpoint-subtract", type=float, default=2.0,
                    help="Degrees F to subtract from afternoon dewpoint for low forecast (0-3, default 2)")
    p.add_argument("--regime-weights", default="current", choices=sorted(REGIME_WEIGHTS_TABLES),
                    help="Which REGIME_WEIGHTS table drives the final synthesis: 'current' (default, "
                         "hand-derived), 'alt' (slide-derived, unvalidated), or 'fitted' (empirically "
                         "fit + holdout-validated on 90 days at KORD, Jun-Sep only -- see "
                         "REGIME_WEIGHTS_FITTED's comment for caveats before using this elsewhere).")
    p.add_argument("--json", action="store_true", help="Output raw JSON instead of a formatted report")
    p.add_argument("--email", action="store_true",
                    help="Email the results (in addition to printing them) via Gmail's SMTP relay. "
                         "Requires the GMAIL_APP_PASSWORD environment variable to be set to a Gmail "
                         "App Password (see README.md for setup). Intended for unattended/cron runs.")
    p.add_argument("--email-to", default=os.environ.get("TEMP_FORECAST_EMAIL_TO", DEFAULT_EMAIL_TO),
                    help=f"Recipient address for --email (default: {DEFAULT_EMAIL_TO}, or "
                         "$TEMP_FORECAST_EMAIL_TO if set)")
    p.add_argument("--email-from", default=os.environ.get("GMAIL_ADDRESS"),
                    help="Sending Gmail address for --email (default: $GMAIL_ADDRESS env var, "
                         "falling back to --email-to if unset -- i.e. sending to yourself)")
    args = p.parse_args()

    if args.check_timing:
        cities = DEFAULT_CITIES if args.all_cities else None
        assessment = timing_assessment(lat=args.lat, lon=args.lon, station=args.station, cities=cities)
        if args.json:
            print(json.dumps(assessment, indent=2, default=str))
        else:
            print_timing_report(assessment)
        return

    if not args.all_cities and (args.lat is None or args.lon is None or not args.station):
        p.error("--lat, --lon, and --station are required unless --all-cities is given")

    try:
        if args.all_cities:
            results = run_all_cities(
                model=args.model, mos_model=args.mos_model, valid_hour=args.valid_hour,
                low_valid_hour=args.low_valid_hour, dewpoint_subtract=args.dewpoint_subtract,
                upstream_station=args.upstream_station, regime_weights=args.regime_weights,
            )
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            else:
                print_batch_summary(results)
            if args.email:
                today = local_today().isoformat()
                send_email(
                    subject=f"Tomorrow's forecast high/low -- {len(results)} cities ({today})",
                    body=format_batch_email_body(results),
                    to_addr=args.email_to, from_addr=args.email_from,
                )
        else:
            report = build_report(args)
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print_human_report(report)
            if args.email:
                today = local_today().isoformat()
                send_email(
                    subject=f"Tomorrow's forecast for {report['station']} ({today})",
                    body=render_human_report_to_string(report),
                    to_addr=args.email_to, from_addr=args.email_from,
                )
    except requests.exceptions.SSLError:
        print("SSL certificate error while contacting the weather APIs.", file=sys.stderr)
        print("On macOS with a python.org install, this usually means Python's own CA bundle", file=sys.stderr)
        print("was never set up. Fix with one of:", file=sys.stderr)
        print('  open "/Applications/Python 3.x/Install Certificates.command"', file=sys.stderr)
        print("  pip3 install --upgrade certifi", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"Network error contacting the weather APIs: {e}", file=sys.stderr)
        print("Check your internet connection, or that a firewall/VPN isn't blocking", file=sys.stderr)
        print("api.open-meteo.com or mesonet.agron.iastate.edu.", file=sys.stderr)
        sys.exit(1)
    except ssl.SSLCertVerificationError:
        print("SSL certificate error while connecting to Gmail's SMTP server.", file=sys.stderr)
        print("On macOS with a python.org install, this usually means Python's own CA bundle", file=sys.stderr)
        print("was never set up. Fix with one of:", file=sys.stderr)
        print('  open "/Applications/Python 3.x/Install Certificates.command"', file=sys.stderr)
        print("  pip3 install --upgrade certifi", file=sys.stderr)
        sys.exit(1)
    except smtplib.SMTPAuthenticationError:
        print("Gmail rejected the login. Make sure GMAIL_APP_PASSWORD is a 16-character App", file=sys.stderr)
        print("Password (not your normal Gmail password) generated at", file=sys.stderr)
        print("https://myaccount.google.com/apppasswords, and that GMAIL_ADDRESS/--email-from", file=sys.stderr)
        print("matches the Google account that password belongs to.", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
