"""Deterministic, explainable investment-opportunity scoring (0-100).

Method
  1. Peer universe: FAA hub airports (L/M/S) with >= MIN_ANNUAL_DEPARTURES operated
     departures in the current year.
  2. Each sub-metric is normalized to 0-100 by percentile rank within the peer universe
     (0 = lowest peer, 100 = highest; ties share the average rank). Percentiles are
     scale-free and robust to outliers such as ATL's volume.
  3. Sub-metrics combine into three components with fixed sub-weights (config.py).
     A missing sub-metric is dropped and the remaining sub-weights re-normalized - the
     airport is neither rewarded nor penalized for the gap - and it is reported.
  4. Components combine with user-adjustable weights into the opportunity score.
  5. Evidence confidence (High / Medium / Low) is computed separately and is NOT part of
     the score, so airports are not rewarded merely for data completeness.
  6. If an entire critical component (demand or capacity pressure) is missing, the airport
     gets status "insufficient_evidence" and no score.

The opportunity score is a relative screening signal. It is NOT an ROI, NPV, IRR or
profitability forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (CONGESTION_SUBWEIGHTS, DEFAULT_WEIGHTS, DEMAND_SUBWEIGHTS,
                        MIN_ANNUAL_DEPARTURES, OPERATIONS_SUBWEIGHTS, SCORING_HUB_TYPES)

COMPONENTS: dict[str, dict[str, float]] = {
    "demand": DEMAND_SUBWEIGHTS,
    "capacity_pressure": CONGESTION_SUBWEIGHTS,
    "operations": OPERATIONS_SUBWEIGHTS,
}
COMPONENT_LABELS = {
    "demand": "Passenger demand & growth",
    "capacity_pressure": "Capacity pressure (congestion proxy)",
    "operations": "Flight activity & growth",
}
CRITICAL_COMPONENTS = ("demand", "capacity_pressure")

# Raw metric column used for each sub-metric.
SUBMETRIC_SOURCE = {
    "enplanements": "enplanements",
    "enplanement_growth": "enplanement_growth",
    "recovery_vs_2019": "recovery_vs_2019",
    "dep_delay_rate": "dep_delay_rate",
    "avg_taxi_out": "avg_taxi_out_min",
    "nas_delay_per_flight": "nas_delay_per_arrival",
    "arr_delay_rate": "arr_delay_rate",
    "peak_hour_movements": "peak_hour_movements",
    "departures": "operated_departures",
    "departure_growth": "departure_growth",
}
SUBMETRIC_LABELS = {
    "enplanements": "passenger volume",
    "enplanement_growth": "passenger growth",
    "recovery_vs_2019": "recovery vs 2019",
    "dep_delay_rate": "departure delay rate",
    "avg_taxi_out": "taxi-out time",
    "nas_delay_per_flight": "airspace/airport (NAS) delay",
    "arr_delay_rate": "arrival delay rate",
    "peak_hour_movements": "peak-hour schedule intensity",
    "departures": "flight volume",
    "departure_growth": "flight growth",
}


def score_band(score: float) -> str:
    """Plain-language band for a 0-100 opportunity score (relative to peers)."""
    if score is None or pd.isna(score):
        return "not scored"
    return "strong" if score >= 65 else "moderate" if score >= 50 else "weak"


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in COMPONENTS}
    total = sum(w.values())
    if total <= 0:
        raise ValueError("At least one component weight must be positive.")
    return {k: v / total for k, v in w.items()}


def percentile_score(s: pd.Series) -> pd.Series:
    """Map values to 0-100 by percentile rank among non-missing peers. NaN stays NaN."""
    valid = s.dropna()
    if len(valid) <= 1:
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    ranks = valid.rank(method="average")
    out = (ranks - 1) / (len(valid) - 1) * 100
    return out.reindex(s.index)


def peer_universe(table: pd.DataFrame) -> pd.DataFrame:
    mask = table["hub"].isin(SCORING_HUB_TYPES) & (table["operated_departures"].fillna(0) >= MIN_ANNUAL_DEPARTURES)
    return table[mask].copy()


COVERAGE_RATIO_FLAG = 1.5  # pax per covered departure vs peer median that signals partial coverage


def _confidence(row: pd.Series, n_missing: int, n_total: int, coverage_ratio: float | None = None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    pts = 1.0 - n_missing / n_total * 0.6
    if n_missing:
        reasons.append(f"{n_missing} of {n_total} inputs missing")
    if coverage_ratio is not None and not pd.isna(coverage_ratio) and coverage_ratio > COVERAGE_RATIO_FLAG:
        pts -= 0.3
        reasons.append(f"passengers per BTS-covered departure is {coverage_ratio:.1f}x the peer median - a large share "
                       "of traffic is international or on non-reporting carriers, which the flight metrics do not see")
    def num(key: str) -> float:
        v = row.get(key)
        return 0.0 if v is None or pd.isna(v) else float(v)

    if num("months_covered") < 12 or num("months_covered_prior") < 12:
        pts -= 0.2
        reasons.append("BTS flight data does not cover every month of both years")
    if num("operated_departures") < 3650:
        pts -= 0.2
        reasons.append("small sample (< ~10 reporting-carrier departures/day)")
    level = "High" if pts >= 0.85 else "Medium" if pts >= 0.6 else "Low"
    return level, reasons


def score_airports(table: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Score the peer universe. Returns one row per airport sorted by rank, with every
    sub-metric percentile, component score, overall score, confidence and status."""
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    peers = peer_universe(table)
    if peers.empty:
        return peers

    for sub, col in SUBMETRIC_SOURCE.items():
        peers[f"pct_{sub}"] = percentile_score(peers[col]) if col in peers else np.nan

    n_total = len(SUBMETRIC_SOURCE)
    missing_lists, statuses, confidences, conf_reasons = [], [], [], []
    for comp, subw in COMPONENTS.items():
        vals = peers[[f"pct_{s}" for s in subw]].to_numpy(dtype=float)
        wts = np.array(list(subw.values()))
        present = ~np.isnan(vals)
        wsum = (present * wts).sum(axis=1)
        comp_score = np.where(wsum > 0, np.nansum(vals * wts, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
        peers[f"score_{comp}"] = comp_score

    # Coverage check: FAA enplanements include every carrier and international passengers; BTS
    # flights cover domestic reporting carriers only. A high ratio means partial flight coverage.
    ppd = peers["enplanements"] / peers["operated_departures"].where(peers["operated_departures"] > 0)
    peers["coverage_ratio"] = ppd / ppd.median()

    for iata, row in peers.iterrows():
        missing = [SUBMETRIC_LABELS[s] for s in SUBMETRIC_SOURCE if pd.isna(row[f"pct_{s}"])]
        missing_lists.append(missing)
        crit_missing = [c for c in CRITICAL_COMPONENTS if pd.isna(row[f"score_{c}"])]
        statuses.append("insufficient_evidence" if crit_missing else "scored")
        lvl, why = _confidence(row, len(missing), n_total, row["coverage_ratio"])
        confidences.append(lvl)
        conf_reasons.append(why)
    peers["missing_inputs"] = missing_lists
    peers["status"] = statuses
    peers["confidence"] = confidences
    peers["confidence_reasons"] = conf_reasons

    # Overall score: weighted mean of available components (operations may be missing).
    comp_vals = peers[[f"score_{c}" for c in COMPONENTS]].to_numpy(dtype=float)
    cw = np.array([w[c] for c in COMPONENTS])
    present = ~np.isnan(comp_vals)
    wsum = (present * cw).sum(axis=1)
    overall = np.nansum(comp_vals * cw, axis=1) / np.where(wsum > 0, wsum, 1)
    peers["opportunity_score"] = np.where(peers["status"] == "scored", overall, np.nan)

    # Deterministic ordering: score desc, then IATA code asc as tie-breaker.
    peers = peers.reset_index().sort_values(["opportunity_score", "iata"], ascending=[False, True],
                                            na_position="last")
    peers["rank"] = np.where(peers["opportunity_score"].notna(),
                             np.arange(1, len(peers) + 1), np.nan)
    peers["explanation"] = [explain_row(r, len(peers)) for _, r in peers.iterrows()]
    return peers.set_index("iata")


def explain_row(row: pd.Series, n: int) -> str:
    if row["status"] != "scored":
        return "Insufficient evidence: " + ", ".join(row["missing_inputs"]) + " missing."
    strengths = [(SUBMETRIC_LABELS[s], row[f"pct_{s}"]) for s in SUBMETRIC_SOURCE
                 if pd.notna(row[f"pct_{s}"]) and row[f"pct_{s}"] >= 75]
    weak = [(SUBMETRIC_LABELS[s], row[f"pct_{s}"]) for s in SUBMETRIC_SOURCE
            if pd.notna(row[f"pct_{s}"]) and row[f"pct_{s}"] <= 25]
    strengths.sort(key=lambda x: -x[1])
    weak.sort(key=lambda x: x[1])
    parts = []
    if strengths:
        parts.append("high " + ", ".join(f"{lbl} (P{p:.0f})" for lbl, p in strengths[:3]))
    if weak:
        parts.append("low " + ", ".join(f"{lbl} (P{p:.0f})" for lbl, p in weak[:2]))
    text = "; ".join(parts) or "close to the peer median on most inputs"
    return text[0].upper() + text[1:] + "."


def component_contributions(scored: pd.DataFrame, iata: str, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Weighted points each component contributes to the overall score."""
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    row = scored.loc[iata]
    present = {c: row[f"score_{c}"] for c in COMPONENTS if pd.notna(row[f"score_{c}"])}
    wsum = sum(w[c] for c in present)
    return {c: present[c] * w[c] / wsum for c in present} if wsum else {}


# ---------------------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------------------
@dataclass
class SensitivityResult:
    scenarios: dict[str, dict[str, float]]
    ranks: pd.DataFrame            # airports x scenarios (rank)
    summary: pd.DataFrame          # per airport: base rank, best, worst, top-N frequency
    scenario_impact: pd.DataFrame  # per scenario: avg |rank shift| in base top-N, top-N changes
    top_n: int


def weight_scenarios(base: dict[str, float], delta: float = 0.10) -> dict[str, dict[str, float]]:
    base = normalize_weights(base)
    sc = {"Base": base}
    for c in COMPONENTS:
        for sign, tag in ((+1, "+"), (-1, "-")):
            w = dict(base)
            w[c] = max(0.0, w[c] + sign * delta)
            sc[f"{COMPONENT_LABELS[c].split(' (')[0]} {tag}{int(delta * 100)}pp"] = normalize_weights(w)
    sc["Equal weights"] = normalize_weights({c: 1 for c in COMPONENTS})
    return sc


def sensitivity_analysis(table: pd.DataFrame, base_weights: dict[str, float] | None = None,
                         top_n: int = 10, delta: float = 0.10,
                         extra: dict[str, dict[str, float]] | None = None) -> SensitivityResult:
    scenarios = weight_scenarios(base_weights or DEFAULT_WEIGHTS, delta)
    if extra:
        scenarios.update({k: normalize_weights(v) for k, v in extra.items()})
    ranks = pd.DataFrame({name: score_airports(table, w)["rank"] for name, w in scenarios.items()})
    ranks = ranks.dropna(subset=["Base"]).sort_values("Base")
    summary = pd.DataFrame({
        "base_rank": ranks["Base"],
        "best_rank": ranks.min(axis=1),
        "worst_rank": ranks.max(axis=1),
        f"in_top_{top_n}_share": (ranks <= top_n).mean(axis=1),
    })
    base_top = ranks.index[ranks["Base"] <= top_n]
    impact = []
    for name in scenarios:
        if name == "Base":
            continue
        top = set(ranks.index[ranks[name] <= top_n])
        impact.append({
            "scenario": name,
            "avg_rank_shift_base_top": float((ranks.loc[base_top, name] - ranks.loc[base_top, "Base"]).abs().mean()),
            "top_n_changes": len(set(base_top) - top),
        })
    impact_df = pd.DataFrame(impact).sort_values("avg_rank_shift_base_top", ascending=False).reset_index(drop=True)
    return SensitivityResult(scenarios, ranks, summary, impact_df, top_n)
