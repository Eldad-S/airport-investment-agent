import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import (build_airport_table, haversine_miles, long_haul_share, ops_metrics,
                                   safe_div)
from src.data_sources.bts_ontime import clean_flights, normalize_code


@pytest.mark.parametrize("raw,expected", [
    ("LAX", "LAX"), (" lax ", "LAX"), ("KLAX", "LAX"), ("sfo", "SFO"), ("1G4", "1G4"),
    ("LA", None), ("LAXX1", None), ("", None), (None, None), (float("nan"), None), ("L@X", None),
])
def test_normalize_code(raw, expected):
    assert normalize_code(raw) == expected


def test_clean_flights_removes_duplicates_and_invalid_codes():
    base = dict(FlightDate="2025-01-01", Reporting_Airline="AA", Flight_Number_Reporting_Airline=1,
                Origin="LAX", Dest="JFK", CRSDepTime=800, Cancelled=0.0, Diverted=0.0)
    df = pd.DataFrame([base, base, {**base, "Flight_Number_Reporting_Airline": 2},
                       {**base, "Origin": "??", "Flight_Number_Reporting_Airline": 3},
                       {**base, "Dest": " jfk", "Flight_Number_Reporting_Airline": 4, "Cancelled": np.nan}])
    out, stats = clean_flights(df)
    assert stats["rows_raw"] == 5
    assert stats["rows_invalid_airport_code"] == 1
    assert stats["rows_duplicate_removed"] == 1
    assert stats["rows_clean"] == 3
    assert set(out["Dest"]) == {"JFK"}
    assert out["Cancelled"].isna().sum() == 0  # missing cancel flag treated as not cancelled


def test_safe_div_never_returns_inf():
    r = safe_div(pd.Series([1, 2, 3]), pd.Series([0, 2, np.nan]))
    assert math.isnan(r[0]) and r[1] == 1 and math.isnan(r[2])


def test_ops_metrics_denominators(tiny_dataset):
    m = ops_metrics(tiny_dataset.airport_month, 2025).loc["AAA"]
    assert m["scheduled_departures"] == 2400
    assert m["operated_departures"] == 2280
    assert m["cancel_rate"] == pytest.approx(120 / 2400)
    # delay rate uses OPERATED flights as denominator, not scheduled
    assert m["dep_delay_rate"] == pytest.approx(38 * 12 / 2280)
    assert m["avg_taxi_out_min"] == pytest.approx(190 * 20 * 12 / 2280)
    assert m["peak_hour_movements"] == pytest.approx(10.0)
    assert m["months_covered"] == 12


def test_consistent_periods_in_airport_table(tiny_dataset):
    t = build_airport_table(tiny_dataset, current_year=2025, prior_year=2024)
    assert t.loc["AAA", "operated_departures"] == 2280
    assert t.loc["AAA", "operated_departures_prior"] == 2160
    assert t.loc["AAA", "departure_growth"] == pytest.approx(2280 / 2160 - 1)
    assert t.loc["AAA", "enplanement_growth"] == pytest.approx(0.10)
    assert t.loc["BBB", "recovery_vs_2019"] == pytest.approx(1.25)


def test_partial_year_growth_is_blank_not_guessed(tiny_dataset):
    t = build_airport_table(tiny_dataset)
    assert t.loc["CCC", "months_covered"] == 6
    assert math.isnan(t.loc["CCC", "departure_growth"])
    assert math.isnan(t.loc["CCC", "recovery_vs_2019"])  # 2019 value missing -> NaN, not 0


def test_haversine_known_distance():
    # LAX -> JFK great-circle is ~2,475 statute miles (BTS reports 2,475)
    d = haversine_miles(33.9425, -118.4081, 40.6398, -73.7789)
    assert d == pytest.approx(2475, abs=10)


def test_long_haul_share_numerator_and_denominator(tiny_dataset):
    r = long_haul_share(tiny_dataset, "AAA", threshold_miles=2485, year=2025)
    assert r.scheduled_departures == 45
    assert r.cancelled_excluded == 2                 # cancelled flights never flew
    assert r.missing_coordinates_excluded == 5       # DDD has no coordinates
    assert r.eligible_departures == 38               # (10-2) to BBB + 30 to CCC
    assert r.long_haul_departures == 8               # diverted flight still counts (it departed)
    assert r.share_pct == pytest.approx(8 / 38 * 100)
    assert r.top_long_haul_routes["dest"].tolist() == ["BBB"]


def test_long_haul_threshold_is_configurable(tiny_dataset):
    low = long_haul_share(tiny_dataset, "AAA", threshold_miles=10, year=2025)
    high = long_haul_share(tiny_dataset, "AAA", threshold_miles=5000, year=2025)
    assert low.share_pct == pytest.approx(100.0)
    assert high.share_pct == pytest.approx(0.0)


def test_long_haul_handles_airport_without_data(tiny_dataset):
    r = long_haul_share(tiny_dataset, "ZZZ", threshold_miles=2485, year=2025)
    assert r.share_pct is None and r.eligible_departures == 0
