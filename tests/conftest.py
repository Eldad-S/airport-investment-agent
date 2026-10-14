"""Shared test fixtures.

Unit tests use tiny, hand-built fixtures (clearly synthetic, defined below) so expected values
can be verified by hand. Integration tests run against the committed real-data snapshot in
data/processed and are skipped if it is missing. No test makes a network call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR  # noqa: E402
from src.data_sources.repository import Dataset, REQUIRED_FILES, load_dataset  # noqa: E402

SNAPSHOT_AVAILABLE = all((PROCESSED_DIR / f).exists() for f in REQUIRED_FILES)
requires_snapshot = pytest.mark.skipif(not SNAPSHOT_AVAILABLE, reason="processed data snapshot not built")


def _airport_month_row(airport, year, month, sched, cancelled, del15, taxi_sum, nas=0.0, peak_sum=310, days=31):
    operated = sched - cancelled
    return dict(airport=airport, year=year, month=month, scheduled_departures=sched,
                cancelled_departures=cancelled, operated_departures=operated, dep_del15_count=del15,
                dep_del15_obs=operated, dep_delay_minutes_sum=del15 * 40.0, taxi_out_sum=taxi_sum,
                taxi_out_obs=operated, scheduled_arrivals=sched, arr_del15_count=del15,
                arr_del15_obs=operated, nas_delay_minutes_sum=nas, daily_peak_movements_sum=peak_sum,
                days_observed=days)


@pytest.fixture
def tiny_dataset() -> Dataset:
    """Synthetic 4-airport dataset.

    AAA: origin under test. BBB is ~2,650 mi east of AAA (long-haul), CCC ~35 mi away
    (short-haul), DDD has no coordinates (distance not computable).
    """
    airports = pd.DataFrame([
        dict(iata="AAA", faa_lid="AAA", name="Alpha Intl", city="Alpha", state="CA", lat=40.0, lon=-100.0, airport_type="large_airport"),
        dict(iata="BBB", faa_lid="BBB", name="Bravo Intl", city="Bravo", state="MA", lat=40.0, lon=-50.0, airport_type="large_airport"),
        dict(iata="CCC", faa_lid="CCC", name="Charlie Rgnl", city="Charlie", state="CA", lat=40.5, lon=-100.0, airport_type="medium_airport"),
        dict(iata="DDD", faa_lid="DDD", name="Delta Field", city="Delta", state="TX", lat=np.nan, lon=np.nan, airport_type="small_airport"),
    ])
    enpl = pd.DataFrame([
        dict(locid="AAA", iata="AAA", hub="L", enpl_cy2025=1_100_000, enpl_cy2024=1_000_000, enpl_cy2019=1_200_000),
        dict(locid="BBB", iata="BBB", hub="M", enpl_cy2025=500_000, enpl_cy2024=500_000, enpl_cy2019=400_000),
        dict(locid="CCC", iata="CCC", hub="S", enpl_cy2025=90_000, enpl_cy2024=100_000, enpl_cy2019=np.nan),
    ])
    rows = []
    for m in range(1, 13):
        # AAA: 200 scheduled/month in 2025 (10 cancelled, 38 delayed), 180 in 2024
        rows.append(_airport_month_row("AAA", 2025, m, 200, 10, 38, 190 * 20.0, nas=500.0))
        rows.append(_airport_month_row("AAA", 2024, m, 180, 0, 18, 180 * 15.0))
        rows.append(_airport_month_row("BBB", 2025, m, 100, 0, 10, 100 * 12.0))
        rows.append(_airport_month_row("BBB", 2024, m, 100, 0, 10, 100 * 12.0))
    for m in range(1, 7):  # CCC only has half a year of 2025 data -> growth must be blank
        rows.append(_airport_month_row("CCC", 2025, m, 100, 0, 5, 100 * 10.0))
    routes = pd.DataFrame([
        dict(year=2025, month=1, origin="AAA", dest="BBB", scheduled=10, cancelled=2, diverted=1, bts_distance_miles=2650.0),
        dict(year=2025, month=1, origin="AAA", dest="CCC", scheduled=30, cancelled=0, diverted=0, bts_distance_miles=35.0),
        dict(year=2025, month=1, origin="AAA", dest="DDD", scheduled=5, cancelled=0, diverted=0, bts_distance_miles=900.0),
        dict(year=2024, month=1, origin="AAA", dest="BBB", scheduled=99, cancelled=0, diverted=0, bts_distance_miles=2650.0),
    ])
    return Dataset(airports=airports, enplanements=enpl, airport_month=pd.DataFrame(rows), route_month=routes,
                   manifest={"built_at_utc": "test"})


def make_peer_table(n: int = 6, seed: int = 0) -> pd.DataFrame:
    """Synthetic airport table shaped like build_airport_table() output, for scoring tests."""
    rng = np.random.default_rng(seed)
    codes = [f"A{i:02d}" for i in range(n)]
    t = pd.DataFrame({
        "name": [f"Airport {c}" for c in codes], "city": codes, "state": ["CA"] * n,
        "lat": 35.0, "lon": -100.0, "airport_type": "large_airport", "hub": ["L"] * n,
        "enplanements": rng.uniform(1e6, 3e7, n), "enplanements_prior": rng.uniform(1e6, 3e7, n),
        "enplanements_2019": rng.uniform(1e6, 3e7, n),
        "operated_departures": rng.uniform(5e3, 2e5, n), "operated_departures_prior": rng.uniform(5e3, 2e5, n),
        "dep_delay_rate": rng.uniform(.1, .3, n), "avg_taxi_out_min": rng.uniform(10, 25, n),
        "arr_delay_rate": rng.uniform(.1, .3, n), "nas_delay_per_arrival": rng.uniform(0, 6, n),
        "peak_hour_movements": rng.uniform(5, 100, n), "cancel_rate": .01,
        "months_covered": 12, "months_covered_prior": 12,
    }, index=pd.Index(codes, name="iata"))
    t["enplanement_growth"] = t["enplanements"] / t["enplanements_prior"] - 1
    t["recovery_vs_2019"] = t["enplanements"] / t["enplanements_2019"]
    t["departure_growth"] = t["operated_departures"] / t["operated_departures_prior"] - 1
    return t


@pytest.fixture
def peer_table() -> pd.DataFrame:
    return make_peer_table()


@pytest.fixture(scope="session")
def real_agent():
    if not SNAPSHOT_AVAILABLE:
        pytest.skip("processed data snapshot not built")
    from src.agent.orchestrator import AnalystAgent
    from src.agent.tools import AnalyticsEngine
    return AnalystAgent(AnalyticsEngine(load_dataset()), llm_parser=None)
