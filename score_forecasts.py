#!/usr/bin/env python3
"""
score_forecasts.py -- grade temp_forecast.py against what actually happened.

Runs every morning on GitHub Actions (.github/workflows/score.yml):
  1. Reads every saved forecast in forecasts/<target date>.csv (written each
     night by `temp_forecast.py --all-cities --save-forecasts forecasts`).
  2. For each finished day not yet graded, pulls the official NWS Daily
     Climate Report (CLI) high/low for each station from Iowa Environmental
     Mesonet -- the same official number record books and weather markets use.
  3. Appends the results to scores/history.csv and emails a scorecard:
     the newly graded days plus running averages.

A day whose official report isn't published yet is simply retried the next
morning, so a missed or failed run catches itself up automatically.

Local use:  python3 score_forecasts.py            (prints, no email)
            python3 score_forecasts.py --email    (needs GMAIL_ADDRESS + GMAIL_APP_PASSWORD)
"""

import argparse
import csv
import json
import os
import smtplib
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
FORECAST_DIR = ROOT / "forecasts"
HISTORY_CSV = ROOT / "scores" / "history.csv"
LOCAL_TZ = ZoneInfo(os.environ.get("TEMP_FORECAST_TZ", "America/Denver"))
DEFAULT_TO = "megan.christina.geyer@gmail.com"

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py?station={station}&year={year}"
PAUSE_BETWEEN_REQUESTS = 1.5  # seconds -- IEM returns 429 if hit too fast
BIG_MISS = 8                  # degrees F; flagged in the scorecard

HISTORY_FIELDS = ["target_date", "city", "station", "pattern",
                  "fc_high", "act_high", "err_high", "fc_low", "act_low", "err_low"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_int(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None  # blank, "M" (missing), "T" (trace), None


def fmt(v):
    return "n/a" if v is None else f"{v}F"


def fmt_err(v):
    return "-" if v is None else f"{v:+d}"


def mean_abs(vals):
    vals = [abs(v) for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


# --------------------------------------------------------------------------- #
# Load saved forecasts and past scores
# --------------------------------------------------------------------------- #
def load_forecasts():
    """{target_date: [row, ...]} from forecasts/*.csv"""
    out = {}
    for path in sorted(FORECAST_DIR.glob("*.csv")):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            out[rows[0]["target_date"]] = rows
    return out


def load_history():
    if not HISTORY_CSV.exists():
        return []
    with open(HISTORY_CSV, newline="") as f:
        return list(csv.DictReader(f))


def append_history(rows):
    if not rows:
        return
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in HISTORY_FIELDS})


# --------------------------------------------------------------------------- #
# Fetch actuals (NWS CLI via Iowa Environmental Mesonet), with retry
# --------------------------------------------------------------------------- #
def fetch_json(url, retries=4):
    delay = 5
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "temp_predict-scorer/1.0"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except URLError:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def get_actuals(stations, years, errors):
    """{station: {YYYY-MM-DD: (high, low)}}"""
    actuals = defaultdict(dict)
    for station in sorted(stations):
        for year in sorted(years):
            try:
                data = fetch_json(IEM_CLI_URL.format(station=station, year=year))
            except Exception as e:
                errors.append(f"{station}: could not fetch official report ({e})")
                continue
            finally:
                time.sleep(PAUSE_BETWEEN_REQUESTS)
            for rec in data.get("results", []):
                if rec.get("valid"):
                    actuals[station][rec["valid"]] = (to_int(rec.get("high")), to_int(rec.get("low")))
    return actuals


# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #
def score_day(rows, actuals):
    """Return (scored_rows, unscored_city_names) for one target day."""
    tdate = rows[0]["target_date"]
    scored, missing = [], []
    for r in rows:
        actual = actuals.get(r["station"], {}).get(tdate)
        if not actual or (actual[0] is None and actual[1] is None):
            missing.append(r["city"])
            continue
        ahi, alo = actual
        fhi, flo = to_int(r["fc_high"]), to_int(r["fc_low"])
        scored.append({
            "target_date": tdate, "city": r["city"], "station": r["station"],
            "pattern": r.get("pattern", ""),
            "fc_high": fhi, "act_high": ahi,
            "err_high": fhi - ahi if fhi is not None and ahi is not None else None,
            "fc_low": flo, "act_low": alo,
            "err_low": flo - alo if flo is not None and alo is not None else None,
        })
    return scored, missing


def day_table(scored, missing):
    tdate = date.fromisoformat(scored[0]["target_date"])
    lines = [
        f"{tdate:%a %b %d, %Y}",
        f"{'City':<19}{'High':>5}{'act':>5}{'miss':>5}  {'Low':>5}{'act':>5}{'miss':>5}  notes",
        "-" * 64,
    ]
    for r in scored:
        notes = []
        if r["fc_high"] is not None and r["fc_low"] is not None and r["fc_high"] < r["fc_low"]:
            notes.append("HIGH<LOW")
        if r["err_high"] is not None and abs(r["err_high"]) >= BIG_MISS:
            notes.append("big high miss")
        if r["err_low"] is not None and abs(r["err_low"]) >= BIG_MISS:
            notes.append("big low miss")
        lines.append(
            f"{r['city'][:18]:<19}{fmt(r['fc_high']):>5}{fmt(r['act_high']):>5}{fmt_err(r['err_high']):>5}  "
            f"{fmt(r['fc_low']):>5}{fmt(r['act_low']):>5}{fmt_err(r['err_low']):>5}  {', '.join(notes)}"
        )
    mh, nh = mean_abs([r["err_high"] for r in scored])
    ml, nl = mean_abs([r["err_low"] for r in scored])
    errs = [r[k] for r in scored for k in ("err_high", "err_low") if r[k] is not None]
    within3 = sum(1 for e in errs if abs(e) <= 3)
    lines.append("-" * 64)
    if mh is not None:
        lines.append(f"Avg high miss {mh:.1f}F ({nh} cities)")
    if ml is not None:
        lines.append(f"Avg low miss  {ml:.1f}F ({nl} cities)")
    if errs:
        lines.append(f"Within 3F: {within3} of {len(errs)}")
    if missing:
        lines.append(f"No official report yet: {', '.join(missing)} (will retry tomorrow)")
    return "\n".join(lines)


def running_summary(history):
    if not history:
        return ""
    today = datetime.now(LOCAL_TZ).date()
    out = ["RUNNING ACCURACY"]
    for label, days in (("Last 7 days", 7), ("Last 30 days", 30), ("All time", None)):
        rows = [h for h in history
                if days is None or date.fromisoformat(h["target_date"]) >= today - timedelta(days=days)]
        mh, nh = mean_abs([to_int(h["err_high"]) for h in rows])
        ml, nl = mean_abs([to_int(h["err_low"]) for h in rows])
        if mh is None and ml is None:
            continue
        hi = f"high {mh:.1f}F" if mh is not None else "high n/a"
        lo = f"low {ml:.1f}F" if ml is not None else "low n/a"
        out.append(f"  {label:<13} avg miss: {hi}, {lo}  ({nh + nl} values)")

    # Bias: is the model running warm or cold overall?
    bias_h = [to_int(h["err_high"]) for h in history if to_int(h["err_high"]) is not None]
    bias_l = [to_int(h["err_low"]) for h in history if to_int(h["err_low"]) is not None]
    if bias_h and bias_l:
        bh, bl = sum(bias_h) / len(bias_h), sum(bias_l) / len(bias_l)
        out.append(f"  Bias (all time): highs {bh:+.1f}F, lows {bl:+.1f}F  (+ = forecasts run warm)")

    # Worst cities by combined average miss (needs a few days of data)
    by_city = defaultdict(list)
    for h in history:
        for k in ("err_high", "err_low"):
            v = to_int(h[k])
            if v is not None:
                by_city[h["city"]].append(abs(v))
    ranked = sorted(((sum(v) / len(v), c) for c, v in by_city.items() if len(v) >= 4), reverse=True)
    if ranked:
        out.append("  Hardest cities so far: " +
                   ", ".join(f"{c} ({m:.1f}F)" for m, c in ranked[:3]))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Email
# --------------------------------------------------------------------------- #
def send_email(subject, body, to_addr, user, password):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
        s.login(user, password)
        s.send_message(msg)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Grade saved forecasts against official NWS highs/lows.")
    ap.add_argument("--email", action="store_true", help="email the scorecard")
    ap.add_argument("--rescore", action="store_true",
                    help="ignore scores/history.csv and regrade every saved day (doesn't write history)")
    args = ap.parse_args()

    forecasts = load_forecasts()
    history = [] if args.rescore else load_history()
    graded = {h["target_date"] for h in history}
    today = datetime.now(LOCAL_TZ).date()

    to_grade = {d: rows for d, rows in forecasts.items()
                if d not in graded and date.fromisoformat(d) < today}

    errors, sections, new_rows = [], [], []
    if to_grade:
        stations = {r["station"] for rows in to_grade.values() for r in rows}
        years = {date.fromisoformat(d).year for d in to_grade}
        actuals = get_actuals(stations, years, errors)
        for d in sorted(to_grade):
            scored, missing = score_day(to_grade[d], actuals)
            if not scored:
                continue  # nothing published yet for this day -- retry tomorrow
            if missing and date.fromisoformat(d) >= today - timedelta(days=1):
                # Some reports not out yet for yesterday: wait one more day
                # rather than locking in a partial grade.
                sections.append(day_table(scored, missing) + "\n(partial -- will regrade tomorrow)")
                continue
            sections.append(day_table(scored, missing))
            new_rows.extend(scored)

    if not args.rescore:
        append_history(new_rows)
        history = load_history()
    else:
        history = new_rows

    if sections:
        body = "\n\n".join(sections)
    else:
        body = "No new days to grade this morning (official reports not out yet, or no saved forecasts)."
    summary = running_summary(history)
    if summary:
        body += "\n\n" + summary
    body += ("\n\nmiss = forecast minus actual (+ means forecast was too warm). "
             "Actuals: official NWS Daily Climate Report via Iowa Environmental Mesonet.")
    if errors:
        body += "\n\nFETCH PROBLEMS:\n  " + "\n  ".join(errors)

    print(body)

    if args.email:
        user = os.environ.get("GMAIL_ADDRESS")
        password = os.environ.get("GMAIL_APP_PASSWORD")
        to_addr = os.environ.get("TEMP_FORECAST_EMAIL_TO") or DEFAULT_TO
        if not (user and password):
            sys.exit("Email requested but GMAIL_ADDRESS / GMAIL_APP_PASSWORD are not set.")
        send_email(f"Forecast scorecard -- {today:%b %d}", body, to_addr, user, password)
        print(f"\nEmailed scorecard to {to_addr}")


if __name__ == "__main__":
    main()
