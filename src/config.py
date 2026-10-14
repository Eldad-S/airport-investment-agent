"""Central configuration: paths, analysis period, thresholds and default scoring weights.

Everything that is an *assumption* (thresholds, weights, peer-set filters) lives here so it
can be inspected and changed in one place. Values are prototype assumptions, not industry
standards - see docs/architecture.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # large downloads, gitignored
PROCESSED_DIR = DATA_DIR / "processed"  # small committed snapshot the app reads
CONTEXT_DIR = DATA_DIR / "context"

# --- Analysis period -------------------------------------------------------------------
# Calendar years are used so that BTS flight data lines up exactly with the FAA
# calendar-year (CY) enplanement statistics. Comparisons always use the same period.
CURRENT_YEAR = 2025
PRIOR_YEAR = 2024
BASELINE_YEAR = 2019  # pre-pandemic reference for passenger recovery

# --- Long-haul definition --------------------------------------------------------------
# EUROCONTROL classifies flights > 4,000 km as long-haul. 4,000 km = 2,485 statute miles.
LONG_HAUL_THRESHOLD_MILES = float(os.getenv("LONG_HAUL_THRESHOLD_MILES", "2485"))

# --- Peer universe for scoring -----------------------------------------------------------
# Only FAA hub airports (Large / Medium / Small) with meaningful reporting-carrier activity
# are ranked; non-hubs have too little BTS on-time coverage to compare fairly.
SCORING_HUB_TYPES = ("L", "M", "S")
MIN_ANNUAL_DEPARTURES = 1_000

# --- Default scoring weights (prototype assumptions) ------------------------------------
# Data confidence is deliberately NOT part of the score (it is reported next to it) so that
# airports are not rewarded merely for having more complete data.
DEFAULT_WEIGHTS: dict[str, float] = {
    "demand": 0.40,
    "capacity_pressure": 0.35,
    "operations": 0.25,
}

# Sub-metric weights inside each component (sum to 1 within a component).
DEMAND_SUBWEIGHTS = {
    "enplanements": 0.40,        # scale of passenger demand (log-percentile)
    "enplanement_growth": 0.35,  # CY current vs prior
    "recovery_vs_2019": 0.25,    # CY current vs 2019
}
CONGESTION_SUBWEIGHTS = {
    "dep_delay_rate": 0.25,      # share of operated departures >= 15 min late
    "avg_taxi_out": 0.25,        # surface / airfield congestion proxy
    "nas_delay_per_flight": 0.15,  # delay minutes attributed to the National Airspace System
    "arr_delay_rate": 0.15,
    "peak_hour_movements": 0.20,  # schedule intensity in the busiest hour
}
OPERATIONS_SUBWEIGHTS = {
    "departures": 0.50,          # scale of flight activity
    "departure_growth": 0.50,    # CY current vs prior
}

# --- LLM (optional) --------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
LLM_ENABLED = bool(ANTHROPIC_API_KEY) and os.getenv("DISABLE_LLM", "").lower() not in ("1", "true", "yes")

HTTP_TIMEOUT_SECONDS = 120
HTTP_RETRIES = 3
