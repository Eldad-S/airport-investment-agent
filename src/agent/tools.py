"""Agent tools: explicit, deterministic, unit-testable functions over the analytics layer.

The conversational layer (rule-based router or LLM) only *chooses* these tools and their
arguments. Every number shown to the user is produced here.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

import pandas as pd

from src.analytics import demand as demand_mod
from src.analytics.metrics import build_airport_table, long_haul_share, monthly_trend
from src.analytics.scoring import (COMPONENTS, CONGESTION_SUBWEIGHTS, SUBMETRIC_LABELS,
                                   SUBMETRIC_SOURCE, component_contributions, normalize_weights,
                                   score_airports, sensitivity_analysis)
from src.config import DEFAULT_WEIGHTS, LONG_HAUL_THRESHOLD_MILES
from src.regions import REGIONS, STATE_NAMES


@dataclass
class AirportMatch:
    iata: str
    name: str
    city: str
    state: str
    in_peer_set: bool


class AnalyticsEngine:
    """Holds the dataset, the airport table and the current scoring for a weight config."""

    def __init__(self, ds, weights: dict[str, float] | None = None):
        self.ds = ds
        self.table = build_airport_table(ds)
        self.set_weights(weights or DEFAULT_WEIGHTS)

    def set_weights(self, weights: dict[str, float]) -> None:
        self.weights = normalize_weights(weights)
        self.scored = score_airports(self.table, self.weights)

    # ---- lookup -------------------------------------------------------------------------
    def known_codes(self) -> set[str]:
        return set(self.table.index)

    def search_airports(self, query: str, limit: int = 8) -> list[AirportMatch]:
        q = query.strip().lower()
        t = self.table
        if not q:
            return []
        code_hit = t.index.str.lower() == q
        text = (t["name"].fillna("") + " " + t["city"].fillna("")).str.lower()
        hits = t[code_hit | text.str.contains(q, regex=False)]
        if hits.empty:  # fuzzy fallback for typos ("anchorge")
            names = dict(zip(t["city"].fillna("").str.lower(), t.index))
            close = difflib.get_close_matches(q, list(names), n=limit, cutoff=0.8)
            hits = t.loc[[names[c] for c in close]]
        hits = hits.assign(_e=hits["enplanements"].fillna(0)).sort_values("_e", ascending=False).head(limit)
        return [AirportMatch(i, r["name"], r["city"], r["state"], i in self.scored.index) for i, r in hits.iterrows()]

    def airport_label(self, iata: str) -> str:
        if iata not in self.table.index:
            return iata
        r = self.table.loc[iata]
        return f"{iata} ({r['name']}, {r['city']}, {r['state']})"

    # ---- tools --------------------------------------------------------------------------
    def get_airport_metrics(self, iata: str) -> dict:
        row = self.table.loc[iata].to_dict()
        if iata in self.scored.index:
            s = self.scored.loc[iata]
            row.update({k: s[k] for k in s.index if k.startswith(("score_", "pct_"))})
            row.update({"opportunity_score": s["opportunity_score"], "rank": s["rank"],
                        "confidence": s["confidence"], "status": s["status"],
                        "explanation": s["explanation"], "missing_inputs": s["missing_inputs"],
                        "confidence_reasons": s["confidence_reasons"]})
        return row

    def compare_airports(self, codes: list[str]) -> pd.DataFrame:
        rows = {}
        for c in codes:
            m = self.get_airport_metrics(c)
            rows[c] = {
                "Airport": m.get("name"),
                "FAA hub size": {"L": "Large", "M": "Medium", "S": "Small", "N": "Non-hub"}.get(m.get("hub"), "n/a"),
                "Enplanements CY2025": m.get("enplanements"),
                "Enplanement growth": m.get("enplanement_growth"),
                "Recovery vs 2019": m.get("recovery_vs_2019"),
                "Operated departures": m.get("operated_departures"),
                "Departure growth": m.get("departure_growth"),
                "Departure delay rate": m.get("dep_delay_rate"),
                "Arrival delay rate": m.get("arr_delay_rate"),
                "Avg taxi-out (min)": m.get("avg_taxi_out_min"),
                "NAS delay (min/arrival)": m.get("nas_delay_per_arrival"),
                "Peak-hour movements": m.get("peak_hour_movements"),
                "Cancellation rate": m.get("cancel_rate"),
                "Capacity-pressure score": m.get("score_capacity_pressure"),
                "Opportunity score": m.get("opportunity_score"),
                "National rank": int(m["rank"]) if pd.notna(m.get("rank")) else None,
                "Confidence": m.get("confidence", "Not scored"),
            }
        return pd.DataFrame(rows)

    def assess_congestion(self, iata: str) -> dict:
        m = self.get_airport_metrics(iata)
        comps = []
        for sub, w in CONGESTION_SUBWEIGHTS.items():
            comps.append({"Indicator": SUBMETRIC_LABELS[sub], "Weight": w,
                          "Value": m.get(SUBMETRIC_SOURCE[sub]), "Peer percentile": m.get(f"pct_{sub}")})
        return {"iata": iata, "score": m.get("score_capacity_pressure"),
                "components": pd.DataFrame(comps), "metrics": m}

    def long_haul(self, iata: str, threshold_miles: float | None = None):
        return long_haul_share(self.ds, iata, threshold_miles or LONG_HAUL_THRESHOLD_MILES)

    def assess_demand(self, iata: str):
        return demand_mod.assess_demand(self.ds, self.table, self.scored, iata)

    def region_states(self, region: str | None) -> tuple[str, ...] | None:
        if not region:
            return None
        if region in REGIONS:
            return REGIONS[region]
        if region in STATE_NAMES:
            return (region,)
        return None

    def rank_airports(self, region: str | None = None, top_n: int = 10,
                      weights: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (ranked scored airports, unscored airports in the area with reasons)."""
        scored = self.scored if weights is None else score_airports(self.table, weights)
        states = self.region_states(region)
        ranked = scored[scored["status"] == "scored"]
        unscored_scored = scored[scored["status"] != "scored"]
        if states:
            ranked = ranked[ranked["state"].isin(states)]
            unscored_scored = unscored_scored[unscored_scored["state"].isin(states)]
            outside = self.table[self.table["state"].isin(states) & ~self.table.index.isin(scored.index)
                                 & self.table["enplanements"].fillna(0).ge(10_000)]
        else:
            outside = self.table.iloc[0:0]
        unscored = pd.concat([
            unscored_scored.assign(reason=unscored_scored["explanation"])[["name", "state", "enplanements", "reason"]],
            outside.assign(reason=outside["hub"].map(
                lambda h: "Non-hub airport - outside the scored peer set" if h == "N"
                else "Too few reporting-carrier flights to compare"))[["name", "state", "enplanements", "reason"]],
        ])
        return ranked.head(top_n), unscored.sort_values("enplanements", ascending=False)

    def why_rank(self, a: str, b: str) -> dict:
        ca = component_contributions(self.scored, a, self.weights)
        cb = component_contributions(self.scored, b, self.weights)
        diffs = {c: ca.get(c, 0) - cb.get(c, 0) for c in COMPONENTS}
        subs = []
        for sub in SUBMETRIC_SOURCE:
            pa, pb = self.scored.loc[a, f"pct_{sub}"], self.scored.loc[b, f"pct_{sub}"]
            if pd.notna(pa) and pd.notna(pb):
                subs.append((SUBMETRIC_LABELS[sub], pa - pb, pa, pb))
        subs.sort(key=lambda x: -abs(x[1]))
        return {"contrib_a": ca, "contrib_b": cb, "diffs": diffs, "sub_diffs": subs[:4]}

    def sensitivity(self, top_n: int = 10, extra: dict[str, dict[str, float]] | None = None):
        return sensitivity_analysis(self.table, self.weights, top_n=top_n, extra=extra)

    def trend(self, iata: str) -> pd.DataFrame:
        return monthly_trend(self.ds, iata)

    @staticmethod
    def data_needed() -> dict[str, list[tuple[str, str]]]:
        return {"To quantify unmet demand": demand_mod.REQUIRED_DATA_UNMET_DEMAND,
                "To underwrite an investment": demand_mod.REQUIRED_DATA_INVESTMENT}
