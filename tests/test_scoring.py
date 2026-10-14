import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_peer_table

from src.analytics.scoring import (DEMAND_SUBWEIGHTS, normalize_weights, percentile_score, score_airports,
                                   score_band, sensitivity_analysis)


def test_percentile_score_scale_and_missing():
    s = percentile_score(pd.Series([10, 20, 30, np.nan]))
    assert s.tolist()[:3] == [0.0, 50.0, 100.0]
    assert math.isnan(s.iloc[3])  # missing stays missing - never imputed as 0


def test_percentile_score_ties_and_single_value():
    assert percentile_score(pd.Series([5, 5, 5])).tolist() == [50.0, 50.0, 50.0]
    assert percentile_score(pd.Series([7.0])).tolist() == [50.0]


def test_normalize_weights():
    w = normalize_weights({"demand": 2, "capacity_pressure": 1, "operations": 1})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["demand"] == pytest.approx(0.5)
    assert normalize_weights({"demand": -5, "capacity_pressure": 1, "operations": 0})["demand"] == 0
    with pytest.raises(ValueError):
        normalize_weights({"demand": 0, "capacity_pressure": 0, "operations": 0})


def test_scores_bounded_and_components_returned(peer_table):
    s = score_airports(peer_table)
    assert s["opportunity_score"].between(0, 100).all()
    for c in ("score_demand", "score_capacity_pressure", "score_operations", "confidence", "explanation"):
        assert c in s.columns
    assert s["rank"].tolist() == list(range(1, len(s) + 1))


def test_ranking_is_deterministic(peer_table):
    a = score_airports(peer_table)
    b = score_airports(peer_table.sample(frac=1, random_state=3))  # input order must not matter
    pd.testing.assert_series_equal(a["rank"], b.loc[a.index, "rank"])
    pd.testing.assert_series_equal(a["opportunity_score"], b.loc[a.index, "opportunity_score"])


def test_tie_break_is_alphabetical():
    t = pd.concat([make_peer_table(1)] * 2)
    t.index = pd.Index(["ZZZ", "AAA"], name="iata")
    s = score_airports(t)
    assert s.index.tolist() == ["AAA", "ZZZ"]


def test_weights_change_ranking_in_expected_direction(peer_table):
    t = peer_table.copy()
    # Make A00 the most congested on every proxy and the smallest on demand.
    for col in ("dep_delay_rate", "avg_taxi_out_min", "arr_delay_rate", "nas_delay_per_arrival", "peak_hour_movements"):
        t.loc["A00", col] = t[col].max() * 2
    for col in ("enplanements", "enplanement_growth", "recovery_vs_2019"):
        t.loc["A00", col] = t[col].min() / 2 if t[col].min() > 0 else t[col].min() - 1
    cap_heavy = score_airports(t, {"demand": 0.05, "capacity_pressure": 0.9, "operations": 0.05})
    dem_heavy = score_airports(t, {"demand": 0.9, "capacity_pressure": 0.05, "operations": 0.05})
    assert cap_heavy.loc["A00", "rank"] < dem_heavy.loc["A00", "rank"]


def test_missing_input_is_reweighted_not_penalized(peer_table):
    t = peer_table.copy()
    t.loc["A01", "recovery_vs_2019"] = np.nan
    s = score_airports(t)
    row = s.loc["A01"]
    present = {k: w for k, w in DEMAND_SUBWEIGHTS.items() if k != "recovery_vs_2019"}
    expected = sum(row[f"pct_{k}"] * w for k, w in present.items()) / sum(present.values())
    assert row["score_demand"] == pytest.approx(expected)
    assert "recovery vs 2019" in row["missing_inputs"]
    assert row["status"] == "scored"


def test_missing_critical_component_gives_insufficient_evidence(peer_table):
    t = peer_table.copy()
    for col in ("dep_delay_rate", "avg_taxi_out_min", "arr_delay_rate", "nas_delay_per_arrival", "peak_hour_movements"):
        t.loc["A02", col] = np.nan
    s = score_airports(t)
    assert s.loc["A02", "status"] == "insufficient_evidence"
    assert math.isnan(s.loc["A02", "opportunity_score"])
    assert math.isnan(s.loc["A02", "rank"])
    assert s.loc["A02", "explanation"].startswith("Insufficient evidence")


def test_confidence_not_part_of_score(peer_table):
    t = peer_table.copy()
    t.loc["A03", "months_covered_prior"] = 6  # lowers confidence only
    s1, s2 = score_airports(peer_table), score_airports(t)
    assert s2.loc["A03", "confidence"] != "High"
    assert s1.loc["A03", "opportunity_score"] == pytest.approx(s2.loc["A03", "opportunity_score"])


def test_partial_coverage_flag_lowers_confidence(peer_table):
    t = peer_table.copy()
    t.loc["A04", "enplanements"] = t["enplanements"].max() * 50  # far more pax than covered flights explain
    s = score_airports(t)
    assert s.loc["A04", "confidence"] in ("Medium", "Low")
    assert any("BTS-covered" in r for r in s.loc["A04", "confidence_reasons"])


def test_peer_universe_excludes_non_hubs_and_tiny_airports(peer_table):
    t = peer_table.copy()
    t.loc["A00", "hub"] = "N"
    t.loc["A01", "operated_departures"] = 50
    s = score_airports(t)
    assert "A00" not in s.index and "A01" not in s.index


def test_sensitivity_analysis(peer_table):
    r = sensitivity_analysis(peer_table, top_n=3)
    assert "Base" in r.scenarios and "Equal weights" in r.scenarios
    assert len(r.scenarios) == 8  # base + 3 components x (+/-) + equal
    assert (r.summary["best_rank"] <= r.summary["base_rank"]).all()
    assert (r.summary["worst_rank"] >= r.summary["base_rank"]).all()
    base = score_airports(peer_table)["rank"]
    assert (r.ranks["Base"] == base.loc[r.ranks.index]).all()
    assert set(r.scenario_impact.columns) >= {"scenario", "avg_rank_shift_base_top", "top_n_changes"}


def test_score_band():
    assert score_band(80) == "strong" and score_band(55) == "moderate" and score_band(10) == "weak"
    assert score_band(float("nan")) == "not scored"
