"""Deterministic airport metrics computed from the processed snapshot.

Definitions (all over a full calendar year unless stated):
  scheduled_departures   flights scheduled to depart (reporting carriers, domestic)
  operated_departures    scheduled minus cancelled (diverted flights did depart, so count)
  cancel_rate            cancelled / scheduled
  dep_delay_rate         operated departures with departure delay >= 15 min / operated
                         departures with a recorded delay value
  avg_dep_delay_min      mean departure delay in minutes (early departures count as 0)
  avg_taxi_out_min       mean gate-out to wheels-off minutes (surface congestion proxy)
  arr_delay_rate         arrivals >= 15 min late / operated arrivals with a recorded value
  nas_delay_per_arrival  minutes of delay BTS attributes to the National Airspace System
                         (traffic volume, airport operations, ATC, non-extreme weather) per
                         operated arrival
  peak_hour_movements    average over days of the busiest scheduled hour (deps + arrs)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import BASELINE_YEAR, CURRENT_YEAR, PRIOR_YEAR

SUM_COLS = ["scheduled_departures", "cancelled_departures", "operated_departures",
            "dep_del15_count", "dep_del15_obs", "dep_delay_minutes_sum", "taxi_out_sum",
            "taxi_out_obs", "scheduled_arrivals", "arr_del15_count", "arr_del15_obs",
            "nas_delay_minutes_sum", "daily_peak_movements_sum", "days_observed"]


def safe_div(num, den):
    """Element-wise division returning NaN (never inf / 0) when the denominator is 0 or missing."""
    num = pd.Series(num, dtype="float64") if not isinstance(num, pd.Series) else num.astype("float64")
    den = pd.Series(den, dtype="float64") if not isinstance(den, pd.Series) else den.astype("float64")
    return num / den.where(den > 0)


def ops_metrics(airport_month: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate airport-month rows for one calendar year into rate metrics per airport."""
    df = airport_month[airport_month["year"] == year]
    g = df.groupby("airport")[SUM_COLS].sum()
    g["months_covered"] = df.groupby("airport")["month"].nunique()
    out = pd.DataFrame(index=g.index)
    out["scheduled_departures"] = g["scheduled_departures"]
    out["operated_departures"] = g["operated_departures"]
    out["cancel_rate"] = safe_div(g["cancelled_departures"], g["scheduled_departures"])
    out["dep_delay_rate"] = safe_div(g["dep_del15_count"], g["dep_del15_obs"])
    out["avg_dep_delay_min"] = safe_div(g["dep_delay_minutes_sum"], g["dep_del15_obs"])
    out["avg_taxi_out_min"] = safe_div(g["taxi_out_sum"], g["taxi_out_obs"])
    out["arr_delay_rate"] = safe_div(g["arr_del15_count"], g["arr_del15_obs"])
    out["nas_delay_per_arrival"] = safe_div(g["nas_delay_minutes_sum"], g["arr_del15_obs"])
    out["peak_hour_movements"] = safe_div(g["daily_peak_movements_sum"], g["days_observed"])
    out["months_covered"] = g["months_covered"]
    out.index.name = "iata"
    return out


def growth(current, prior):
    """Fractional growth (current / prior - 1). NaN when prior is missing or zero."""
    return safe_div(current, prior) - 1


def build_airport_table(ds, current_year: int = CURRENT_YEAR, prior_year: int = PRIOR_YEAR) -> pd.DataFrame:
    """One row per airport with reference info, passenger and operations metrics for a
    consistent pair of periods (current vs prior calendar year)."""
    cur = ops_metrics(ds.airport_month, current_year)
    pri = ops_metrics(ds.airport_month, prior_year)[["operated_departures", "months_covered"]]
    pri = pri.rename(columns={"operated_departures": "operated_departures_prior",
                              "months_covered": "months_covered_prior"})
    ref = ds.airports.set_index("iata")[["name", "city", "state", "lat", "lon", "airport_type"]]
    enpl_cols = {f"enpl_cy{current_year}": "enplanements", f"enpl_cy{prior_year}": "enplanements_prior",
                 f"enpl_cy{BASELINE_YEAR}": "enplanements_2019"}
    enpl = ds.enplanements.set_index("iata")[["hub", *[c for c in enpl_cols if c in ds.enplanements]]]
    enpl = enpl.rename(columns=enpl_cols)

    t = ref.join(enpl, how="left").join(cur, how="left").join(pri, how="left")
    t = t[t["enplanements"].notna() | t["operated_departures"].notna()]
    t["hub"] = t["hub"].fillna("N")
    t["enplanement_growth"] = growth(t["enplanements"], t["enplanements_prior"])
    t["recovery_vs_2019"] = safe_div(t["enplanements"], t["enplanements_2019"])
    t["departure_growth"] = growth(t["operated_departures"], t["operated_departures_prior"])
    # Partial-year coverage would make YoY growth misleading: blank it rather than guess.
    partial = (t["months_covered"].fillna(0) < 12) | (t["months_covered_prior"].fillna(0) < 12)
    t.loc[partial, "departure_growth"] = np.nan
    t.index.name = "iata"
    return t


def monthly_trend(ds, iata: str) -> pd.DataFrame:
    df = ds.airport_month[ds.airport_month["airport"] == iata].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["dep_delay_rate"] = safe_div(df["dep_del15_count"], df["dep_del15_obs"]).values
    df["avg_taxi_out_min"] = safe_div(df["taxi_out_sum"], df["taxi_out_obs"]).values
    return df.sort_values("date")[["date", "year", "month", "scheduled_departures", "operated_departures",
                                   "dep_delay_rate", "avg_taxi_out_min"]]


# ---------------------------------------------------------------------------------------
# Long-haul share
# ---------------------------------------------------------------------------------------
EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in statute miles. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


@dataclass
class LongHaulResult:
    airport: str
    year: int
    threshold_miles: float
    scheduled_departures: int
    cancelled_excluded: int
    missing_coordinates_excluded: int
    eligible_departures: int
    long_haul_departures: int
    share_pct: float | None
    max_distance_discrepancy_miles: float | None
    top_long_haul_routes: pd.DataFrame
    sensitivity: dict[float, float | None]
    months_covered: int


def long_haul_share(ds, iata: str, threshold_miles: float, year: int = CURRENT_YEAR,
                    sensitivity_thresholds: tuple[float, ...] = (1500.0, 2000.0, 2485.0, 3000.0)) -> LongHaulResult:
    """Share of eligible departing flights whose great-circle distance >= threshold.

    Denominator ("eligible departures"): scheduled departures from the airport in the period,
      minus cancelled flights (they never flew),
      minus flights whose destination has no known coordinates (distance not computable).
    Diverted flights are kept: they departed for the scheduled destination.
    Duplicate flight records were removed at ingestion on the flight key
    (date, carrier, flight number, origin, destination, scheduled departure time).
    Distance is computed from OurAirports coordinates; BTS's own distance field is used only
    as a cross-check.
    """
    r = ds.route_month[(ds.route_month["origin"] == iata) & (ds.route_month["year"] == year)]
    months = int(r["month"].nunique())
    r = r.groupby("dest").agg(scheduled=("scheduled", "sum"), cancelled=("cancelled", "sum"),
                              bts_distance_miles=("bts_distance_miles", "median")).reset_index()
    coords = ds.airports.set_index("iata")[["lat", "lon"]]
    empty = pd.DataFrame(columns=["dest", "flights", "distance_miles"])
    if iata not in coords.index or r.empty or pd.isna(coords.loc[iata, "lat"]):
        return LongHaulResult(iata, year, threshold_miles, int(r["scheduled"].sum()) if not r.empty else 0,
                              int(r["cancelled"].sum()) if not r.empty else 0, 0, 0, 0, None, None,
                              empty, {}, months)
    o = coords.loc[iata]
    r = r.join(coords, on="dest")
    r["distance_miles"] = haversine_miles(o["lat"], o["lon"], r["lat"], r["lon"])
    r["flights"] = r["scheduled"] - r["cancelled"]
    has_coords = r["distance_miles"].notna()
    eligible = r[has_coords]
    denom = int(eligible["flights"].sum())

    def share(th: float) -> float | None:
        return None if denom == 0 else float(eligible.loc[eligible["distance_miles"] >= th, "flights"].sum() / denom * 100)

    num = int(eligible.loc[eligible["distance_miles"] >= threshold_miles, "flights"].sum())
    disc = (eligible["distance_miles"] - eligible["bts_distance_miles"]).abs()
    top = (eligible[eligible["distance_miles"] >= threshold_miles]
           .sort_values("flights", ascending=False)[["dest", "flights", "distance_miles"]].head(10))
    return LongHaulResult(
        airport=iata, year=year, threshold_miles=threshold_miles,
        scheduled_departures=int(r["scheduled"].sum()),
        cancelled_excluded=int(r["cancelled"].sum()),
        missing_coordinates_excluded=int(r.loc[~has_coords, "flights"].sum()),
        eligible_departures=denom, long_haul_departures=num,
        share_pct=share(threshold_miles),
        max_distance_discrepancy_miles=None if disc.dropna().empty else float(disc.max()),
        top_long_haul_routes=top.reset_index(drop=True),
        sensitivity={th: share(th) for th in sorted(set(sensitivity_thresholds) | {threshold_miles})},
        months_covered=months,
    )


def fmt_pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{digits}f}%"


def fmt_signed_pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{digits}f}%"


def fmt_num(x, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    return f"{x:,.{digits}f}"
