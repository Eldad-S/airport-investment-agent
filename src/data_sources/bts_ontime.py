"""BTS Reporting Carrier On-Time Performance (TranStats PREZIP monthly files).

Source: https://www.transtats.bts.gov/  (public, no authentication). One zip per month with
one row per scheduled domestic flight operated by a *reporting carrier* (carriers with
>= 0.5% of domestic scheduled passenger revenue). Each flight is reported once by the
operating carrier, so codeshare marketing duplicates are not present by design; we still
de-duplicate on the flight key defensively.

We aggregate the flight-level rows into two small tables that are committed to the repo:
  * airport-month operations / delay metrics
  * route-month (origin, dest) flight counts with distance - used for long-haul shares
"""
from __future__ import annotations

import logging
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.data_sources.http import DataSourceError, download

log = logging.getLogger(__name__)

URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
SOURCE_ID = "bts_ontime"

USECOLS = [
    "FlightDate", "Year", "Month", "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "CRSArrTime", "DepDelayMinutes", "DepDel15", "TaxiOut",
    "TaxiIn", "ArrDelayMinutes", "ArrDel15", "Cancelled", "Diverted", "Distance", "NASDelay",
]
FLIGHT_KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
              "Origin", "Dest", "CRSDepTime"]
_IATA_RE = re.compile(r"^[A-Z0-9]{3}$")


def normalize_code(code: object) -> str | None:
    """Normalize an airport identifier to upper-case 3-character IATA/FAA form, else None."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    s = str(code).strip().upper()
    if s.startswith("K") and len(s) == 4 and s[1:].isalpha():  # ICAO 'KLAX' -> 'LAX'
        s = s[1:]
    return s if _IATA_RE.match(s) else None


def month_path(year: int, month: int) -> Path:
    return RAW_DIR / SOURCE_ID / f"ontime_{year}_{month:02d}.zip"


def fetch_month(year: int, month: int, force: bool = False) -> Path:
    return download(URL_TEMPLATE.format(year=year, month=month), month_path(year, month), force=force)


def read_month(zip_path: Path) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(name) as f:
                return pd.read_csv(f, usecols=USECOLS, encoding="latin-1", low_memory=False,
                                   dtype={"Reporting_Airline": str, "Origin": str, "Dest": str})
    except (zipfile.BadZipFile, StopIteration, ValueError) as e:
        raise DataSourceError(f"Corrupt or unexpected BTS file {zip_path.name}: {e}") from e


def clean_flights(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize identifiers, drop invalid rows and duplicate flight records. Returns QA stats."""
    stats = {"rows_raw": int(len(df))}
    df = df.copy()
    df["Origin"] = df["Origin"].map(normalize_code)
    df["Dest"] = df["Dest"].map(normalize_code)
    invalid = df["Origin"].isna() | df["Dest"].isna()
    stats["rows_invalid_airport_code"] = int(invalid.sum())
    df = df[~invalid]
    before = len(df)
    df = df.drop_duplicates(subset=FLIGHT_KEY)
    stats["rows_duplicate_removed"] = int(before - len(df))
    for c in ("Cancelled", "Diverted"):
        df[c] = df[c].fillna(0).astype(int)
    stats["rows_clean"] = int(len(df))
    return df, stats


def _hour(hhmm: pd.Series) -> pd.Series:
    h = (pd.to_numeric(hhmm, errors="coerce") // 100) % 24
    return h


def aggregate_airport_month(df: pd.DataFrame) -> pd.DataFrame:
    """Airport-month metrics. Departure metrics by Origin, arrival metrics by Dest.

    Delay metrics are computed over *operated* flights (not cancelled). NAS-attributed delay
    minutes are attributed to the arrival airport, matching BTS's own airport delay-cause
    reporting convention.
    """
    year, month = int(df["Year"].iloc[0]), int(df["Month"].iloc[0])
    operated = df[df["Cancelled"] == 0]

    dep = df.groupby("Origin").agg(scheduled_departures=("Cancelled", "size"),
                                   cancelled_departures=("Cancelled", "sum"))
    dep_op = operated.groupby("Origin").agg(
        operated_departures=("Cancelled", "size"),
        dep_del15_count=("DepDel15", "sum"),
        dep_del15_obs=("DepDel15", "count"),
        dep_delay_minutes_sum=("DepDelayMinutes", "sum"),
        taxi_out_sum=("TaxiOut", "sum"),
        taxi_out_obs=("TaxiOut", "count"),
    )
    arr = df.groupby("Dest").agg(scheduled_arrivals=("Cancelled", "size"))
    arr_op = operated.groupby("Dest").agg(
        arr_del15_count=("ArrDel15", "sum"),
        arr_del15_obs=("ArrDel15", "count"),
        nas_delay_minutes_sum=("NASDelay", "sum"),
    )

    # Peak-hour schedule intensity: scheduled movements (deps + arrs) in the busiest local
    # hour of each day, averaged over days. Uses scheduled (CRS) times, so it measures
    # schedule concentration rather than realized throughput.
    mv = pd.concat([
        pd.DataFrame({"airport": df["Origin"], "date": df["FlightDate"], "hour": _hour(df["CRSDepTime"])}),
        pd.DataFrame({"airport": df["Dest"], "date": df["FlightDate"], "hour": _hour(df["CRSArrTime"])}),
    ])
    hourly = mv.groupby(["airport", "date", "hour"]).size()
    daily_peak = hourly.groupby(level=["airport", "date"]).max()
    peak = daily_peak.groupby(level="airport").agg(["sum", "count"])
    peak.columns = ["daily_peak_movements_sum", "days_observed"]

    out = (dep.join(dep_op, how="outer").join(arr, how="outer").join(arr_op, how="outer")
           .join(peak, how="outer").fillna(0))
    out.index.name = "airport"
    out = out.reset_index()
    out.insert(1, "year", year)
    out.insert(2, "month", month)
    int_cols = [c for c in out.columns if c.endswith(("_count", "_obs", "departures", "arrivals", "days_observed"))]
    out[int_cols] = out[int_cols].astype(int)
    return out


def aggregate_route_month(df: pd.DataFrame) -> pd.DataFrame:
    year, month = int(df["Year"].iloc[0]), int(df["Month"].iloc[0])
    g = df.groupby(["Origin", "Dest"]).agg(
        scheduled=("Cancelled", "size"),
        cancelled=("Cancelled", "sum"),
        diverted=("Diverted", "sum"),
        bts_distance_miles=("Distance", "median"),
    ).reset_index().rename(columns={"Origin": "origin", "Dest": "dest"})
    g.insert(0, "year", year)
    g.insert(1, "month", month)
    return g


def process_month(year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = month_path(year, month)
    raw = read_month(path)
    flights, stats = clean_flights(raw)
    stats.update({
        "year": year, "month": month, "file": path.name,
        "url": URL_TEMPLATE.format(year=year, month=month),
        "file_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
    })
    return aggregate_airport_month(flights), aggregate_route_month(flights), stats
