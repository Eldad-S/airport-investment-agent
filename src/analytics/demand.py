"""Demand and capacity-constraint evidence - deliberately honest about unmet demand.

Five layers are kept separate:
  1. Observed traffic            (FAA enplanements, BTS operated departures) - observed
  2. Growth & seasonality        (YoY, recovery vs 2019, monthly peak/trough) - derived
  3. Capacity-constraint evidence (delay, taxi-out, NAS delay, peak intensity) - proxies
  4. Potential unmet demand      qualitative pressure level from proxies - inference
  5. Actual unmet demand         NOT quantifiable with this dataset - we say so

The system never outputs a numeric unmet-demand estimate unless the evidence listed in
REQUIRED_DATA_UNMET_DEMAND is present (it is not, in the current snapshot).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from src.config import CONTEXT_DIR, CURRENT_YEAR, PRIOR_YEAR
from src.analytics.metrics import fmt_num, fmt_pct, fmt_signed_pct, monthly_trend
from src.analytics.scoring import SUBMETRIC_LABELS

REQUIRED_DATA_UNMET_DEMAND = [
    ("BTS T-100 segment data (seats, passengers, load factor by route/month)",
     "High load factors on full schedules indicate spilled demand. Public download from TranStats."),
    ("BTS DB1B origin-destination ticket sample (fares)",
     "Fare premiums vs comparable markets indicate supply-constrained routes."),
    ("FAA ASPM / OPSNET (airport acceptance rates, ground-delay programs by hour)",
     "Shows how often scheduled demand exceeds declared runway capacity. ASPM requires an FAA account."),
    ("Airport-declared capacity: gates, terminal design-hour passengers, runway capacity",
     "Needed to compute utilization (demand / capacity) instead of delay proxies."),
    ("Airline schedule requests vs slots / gates granted (OAG / Cirium, airport records)",
     "Directly measures rejected or deferred service."),
    ("Catchment / leakage studies (passengers using alternative airports)",
     "Measures diverted demand."),
    ("FAA Terminal Area Forecast (TAF)",
     "Official forecast of enplanements and operations to compare against capacity."),
]

REQUIRED_DATA_INVESTMENT = [
    ("Project capital cost and phasing", "Needed for any return calculation."),
    ("Airport financials - FAA CATS (Form 127): revenue per passenger, cost per enplanement, debt",
     "Public; needed to estimate incremental revenue and financing capacity."),
    ("Airline use & lease agreements (residual vs compensatory) and PFC levels",
     "Determine who captures the upside of new capacity."),
    ("Regulatory / environmental constraints (NEPA, noise agreements, land availability)",
     "Can cap capacity regardless of demand (e.g., SNA settlement agreement)."),
    ("Demand forecast and airline commitment letters", "De-risk utilization assumptions."),
]


def load_context_notes() -> dict[str, list[dict]]:
    p = CONTEXT_DIR / "airport_notes.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


@dataclass
class DemandAssessment:
    iata: str
    observed: dict[str, str]
    growth: dict[str, str]
    constraint_evidence: list[str]
    pressure_level: str
    pressure_rationale: str
    hypotheses: list[dict] = field(default_factory=list)
    unmet_demand_estimate: float | None = None  # always None with the current data
    unmet_demand_statement: str = ""
    required_data: list[tuple[str, str]] = field(default_factory=lambda: list(REQUIRED_DATA_UNMET_DEMAND))


def assess_demand(ds, table: pd.DataFrame, scored: pd.DataFrame, iata: str) -> DemandAssessment:
    row = table.loc[iata]
    srow = scored.loc[iata] if iata in scored.index else None

    observed = {
        f"Passenger enplanements CY{CURRENT_YEAR} (FAA, all carriers)": fmt_num(row.get("enplanements")),
        f"Operated departures CY{CURRENT_YEAR} (BTS reporting carriers)": fmt_num(row.get("operated_departures")),
    }
    tr = monthly_trend(ds, iata)
    cur = tr[tr["year"] == CURRENT_YEAR]
    seasonality = (cur["operated_departures"].max() / cur["operated_departures"].min()
                   if len(cur) == 12 and cur["operated_departures"].min() > 0 else None)
    growth = {
        f"Enplanement growth CY{CURRENT_YEAR} vs CY{PRIOR_YEAR}": fmt_signed_pct(row.get("enplanement_growth")),
        f"Enplanements vs 2019 (recovery)": fmt_pct(row.get("recovery_vs_2019"), 0),
        f"Departure growth CY{CURRENT_YEAR} vs CY{PRIOR_YEAR}": fmt_signed_pct(row.get("departure_growth")),
        "Seasonality (busiest / quietest month departures)": "n/a" if seasonality is None else f"{seasonality:.2f}x",
    }

    evidence: list[str] = []
    cap = None
    if srow is not None:
        cap = srow["score_capacity_pressure"]
        evidence.append(f"Capacity-pressure (congestion proxy) score {cap:.0f}/100 relative to "
                        f"{len(scored)} peer hub airports.")
        for sub, col, f in (("dep_delay_rate", "dep_delay_rate", fmt_pct),
                            ("avg_taxi_out", "avg_taxi_out_min", lambda v: f"{v:.1f} min"),
                            ("nas_delay_per_flight", "nas_delay_per_arrival", lambda v: f"{v:.1f} min/arrival"),
                            ("peak_hour_movements", "peak_hour_movements", lambda v: f"{v:.0f} movements")):
            p = srow[f"pct_{sub}"]
            if pd.notna(p) and pd.notna(row.get(col)):
                label = SUBMETRIC_LABELS[sub]
                evidence.append(f"{label[0].upper() + label[1:]}: {f(row[col])} (P{p:.0f} among peers)")
    eg, dg = row.get("enplanement_growth"), row.get("departure_growth")
    if pd.notna(eg) and pd.notna(dg) and eg - dg > 0.02:
        evidence.append(f"Passengers grew faster than flights ({fmt_signed_pct(eg)} vs {fmt_signed_pct(dg)}): "
                        "consistent with larger aircraft / fuller flights, which can be a response to gate or "
                        "slot limits - or simply airline fleet strategy. (Enplanements cover all carriers; "
                        "departures cover reporting carriers only.)")
    rec = row.get("recovery_vs_2019")
    if pd.notna(rec) and rec < 0.95:
        evidence.append(f"Passengers are still {fmt_pct(1 - rec, 0)} below 2019 - realized traffic is below a "
                        "level the airport has already handled, which argues against a pure physical-capacity cap "
                        "on passengers (constraints may be hourly / airfield rather than annual).")

    if srow is None:
        level, why = "Insufficient evidence", "Airport is outside the scored peer set (not an FAA hub or too few reporting-carrier flights)."
    else:
        demand_strong = (srow["pct_enplanements"] >= 60) or (pd.notna(srow["pct_enplanement_growth"]) and srow["pct_enplanement_growth"] >= 60)
        if cap >= 70 and demand_strong:
            level, why = "Elevated", "high congestion-proxy readings coincide with large or growing passenger demand"
        elif cap >= 70 or (cap >= 50 and demand_strong):
            level, why = "Moderate", "some congestion-proxy signals, but demand and constraint signals do not both point strongly the same way"
        else:
            level, why = "Low", "congestion proxies are near or below the peer median"

    statement = ("Actual unmet (rejected, suppressed or diverted) demand cannot be quantified from realized "
                 "traffic alone: flown passengers and flights only show demand that was served. This "
                 "assessment therefore reports proxy-based pressure, not a number. The datasets listed below "
                 "would be needed to estimate it.")
    return DemandAssessment(iata=iata, observed=observed, growth=growth, constraint_evidence=evidence,
                            pressure_level=level, pressure_rationale=why,
                            hypotheses=load_context_notes().get(iata, []),
                            unmet_demand_statement=statement)
