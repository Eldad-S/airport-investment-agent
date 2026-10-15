"""Conversational orchestrator: question -> ParsedQuery -> deterministic tools -> answer.

Every substantive answer contains: a bottom line, numerical evidence, the observation
period, sources, caveats and a confidence / evidence-quality statement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.agent.intents import ConversationContext, ParsedQuery
from src.agent.router import Router
from src.agent.tools import AnalyticsEngine
from src.analytics.demand import load_context_notes
from src.analytics.metrics import fmt_num, fmt_pct, fmt_signed_pct
from src.analytics.scoring import (COMPONENT_LABELS, COMPONENTS, SUBMETRIC_LABELS, SUBMETRIC_SOURCE,
                                   component_contributions, normalize_weights, score_airports, score_band)
from src.config import CURRENT_YEAR, LONG_HAUL_THRESHOLD_MILES, PRIOR_YEAR
from src.sources import cite, metrics_registry, sources_table
from src.ui import charts

log = logging.getLogger(__name__)

PERIOD_OPS = f"Jan-Dec {CURRENT_YEAR} (BTS monthly data), growth vs Jan-Dec {PRIOR_YEAR}"
PERIOD_PAX = f"CY{CURRENT_YEAR} (FAA), growth vs CY{PRIOR_YEAR}, recovery vs CY2019"
SCORE_CAVEAT = ("The opportunity score is a relative screening signal (percentiles vs. U.S. hub airports), "
                "not an ROI, NPV, IRR or profitability forecast.")
CONGESTION_CAVEAT = ("Congestion is measured through operational proxies (delays, taxi-out, NAS delay, peak-hour "
                     "schedule intensity). It does not measure terminal crowding or capacity utilization directly; "
                     "delays also reflect weather and airline operations.")
COVERAGE_CAVEAT = ("Flight metrics cover domestic flights by BTS reporting carriers only (excludes most regional-only, "
                   "charter, cargo and foreign airlines).")

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "compare congestion LAX SNA",
    "what % of flights out of Anchorage are long haul?",
    "unmet demand at SFO - what's going on?",
    "top 10 airports per the model",
    "what data would we need to validate an investment?",
]


@dataclass
class AgentResponse:
    text: str
    parsed: ParsedQuery | None = None
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    charts: list = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


def _footer(period: str, sources: str, caveats: list[str], confidence: str) -> str:
    cav = "\n".join(f"- {c}" for c in caveats)
    return (f"\n\n---\n**Period:** {period}  \n**Confidence:** {confidence}  \n**Sources:** {sources}\n\n"
            f"**Caveats**\n{cav}")


def format_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable comparison table; missing values are shown explicitly as MISSING."""
    pct_rows = {"Enplanement growth", "Departure growth"}
    rate_rows = {"Departure delay rate", "Arrival delay rate", "Cancellation rate", "Recovery vs 2019"}
    out = df.copy().astype(object)
    for idx in out.index:
        for col in out.columns:
            v = df.loc[idx, col]
            missing = v is None or (isinstance(v, float) and pd.isna(v))
            if missing:
                out.loc[idx, col] = "MISSING"
            elif idx in pct_rows:
                out.loc[idx, col] = fmt_signed_pct(v)
            elif idx in rate_rows:
                out.loc[idx, col] = fmt_pct(v)
            elif isinstance(v, float):
                out.loc[idx, col] = fmt_num(v, 1) if v < 1000 else fmt_num(v)
            else:
                out.loc[idx, col] = str(v)
    # All cells are display strings so the table serializes cleanly to Arrow (no mixed types).
    return out.astype(str)


def _conf_summary(levels: list[str]) -> str:
    if not levels:
        return "n/a"
    counts = pd.Series(levels).value_counts()
    return ", ".join(f"{n} {lvl}" for lvl, n in counts.items()) + " (evidence confidence, separate from the score)"


class AnalystAgent:
    def __init__(self, engine: AnalyticsEngine, llm_parser=None):
        self.engine = engine
        self.llm = llm_parser
        cities = {}
        peers = engine.scored.sort_values("enplanements", ascending=True)
        for iata, r in peers.iterrows():  # largest airport per city wins (assigned last)
            city = str(r.get("city") or "").lower()
            if len(city) >= 5 and city not in ("mobile", "eagle", "washington", "jackson", "columbus", "national"):
                cities[city] = iata
        self.router = Router(engine.known_codes(), cities)

    # ---- entry point ----------------------------------------------------------------------
    def parse(self, text: str, ctx: ConversationContext) -> ParsedQuery:
        if self.llm is not None:
            q = self.llm.parse(text, ctx, self.engine.known_codes())
            if q is not None:
                return q
        return self.router.parse(text, ctx)

    def run(self, q: ParsedQuery, ctx: ConversationContext | None = None) -> AgentResponse:
        """Execute an already-parsed query with the deterministic tools."""
        ctx = ctx or ConversationContext()
        handler = getattr(self, f"_h_{q.intent}")
        try:
            resp = handler(q, ctx)
        except KeyError as e:  # airport outside the dataset slipped through
            log.exception("lookup failed")
            resp = AgentResponse(f"I couldn't find data for {e}. Try a U.S. airport code such as LAX or BOS.")
        resp.parsed = q
        return resp

    def ask(self, text: str, ctx: ConversationContext | None = None) -> tuple[AgentResponse, ConversationContext]:
        ctx = ctx or ConversationContext()
        if not text or not text.strip():
            return AgentResponse(self._help_text()), ctx
        q = self.parse(text, ctx)
        resp = self.run(q, ctx)
        prefix = ""
        if q.unknown_codes:
            prefix += (f"_Note: I have no data for {', '.join(q.unknown_codes)} - the dataset covers U.S. airports "
                       f"in the BTS / FAA statistics._\n\n")
        if q.used_context and q.airports:
            typed = [a for a in q.airports if a.lower() in text.lower()]
            if len(typed) == len(q.airports):
                prefix += f"_Continuing the previous analysis ({q.intent.replace('_', ' ')}) for {', '.join(q.airports)}._\n\n"
            else:
                prefix += f"_Using airports from the conversation: {', '.join(q.airports)}._\n\n"
        resp.text = prefix + resp.text
        new_ctx = ctx
        if q.intent not in ("off_topic", "help", "clarify", "unsupported_aviation", "sources", "methodology", "data_needed"):
            applied_region = q.region if self.engine.region_states(q.region) else None
            new_ctx = ConversationContext(
                last_intent=q.intent,
                last_airports=q.airports or ctx.last_airports,
                last_region=applied_region if q.intent == "rank" else ctx.last_region,
                last_focus=q.focus or ctx.last_focus,
            )
        return resp, new_ctx

    # ---- handlers -------------------------------------------------------------------------
    def _h_rank(self, q: ParsedQuery, ctx) -> AgentResponse:
        e = self.engine
        top_n = q.top_n or 10
        region = q.region if e.region_states(q.region) else None
        requested = q.unmapped_region or (q.region if q.region and region is None else None)
        ranked, unscored = e.rank_airports(region, top_n=top_n)
        area = region or "the U.S."
        calls = [f"rank_airports(region={region!r}, top_n={top_n})"]
        n_peers = int((e.scored["status"] == "scored").sum())
        note = (f"_Note: I couldn't map \"{requested}\" to a region I know (U.S. regions such as New England, "
                f"Midwest, West Coast, or a state), so this is the **national** ranking._\n\n" if requested else "")
        if ranked.empty:
            txt = f"**Bottom line:** none of the airports in {area} are in the scored peer set of {n_peers} hub airports."
            if not unscored.empty:
                txt += "\n\nAirports in this area that could not be scored:\n" + "\n".join(
                    f"- **{i}** {r['name']}: {r['reason']}" for i, r in unscored.iterrows())
            return AgentResponse(txt, tool_calls=calls)

        lead = ranked.head(3)
        leaders = ", ".join(f"**{i}** ({r['city']}, score {r['opportunity_score']:.0f} = {score_band(r['opportunity_score'])}, "
                            f"#{int(r['rank'])} nationally)" for i, r in lead.iterrows())
        strong = [i for i, r in ranked.iterrows() if score_band(r["opportunity_score"]) == "strong"]
        verdict = (f"{len(strong)} of {len(ranked)} listed airports reach the 'strong' band (score >= 65): {', '.join(strong)}."
                   if strong else "None of the listed airports reach the 'strong' band (score >= 65).")
        txt = (f"**Bottom line:** the top screening candidates in {area} are {leaders}. {verdict} "
               f"Scores are relative to {n_peers} U.S. hub airports.\n\n**Why they rank where they do**\n")
        for i, r in ranked.iterrows():
            txt += f"- **#{int(r['rank'])} {i}** - {r['name']}: {r['explanation']}\n"
        if not unscored.empty:
            txt += "\n**Not scored** (shown for completeness):\n" + "\n".join(
                f"- {i} ({r['name']}, {fmt_num(r['enplanements'])} pax): {r['reason']}" for i, r in unscored.head(8).iterrows())
        if q.region == "New England" or (q.region and len(ranked) <= 6):
            txt += ("\n\n_Terminal expansion specifically: the model ranks demand and operational pressure. Physical "
                    "feasibility (land, gates, regulation) and existing capital programs are not modeled - see "
                    "'what data would we need'._")
        table = pd.DataFrame({
            "National rank": ranked["rank"].astype(int),
            "Airport": ranked["name"], "State": ranked["state"],
            "Score": ranked["opportunity_score"].round(1),
            "Confidence": ranked["confidence"],
            "Demand": ranked["score_demand"].round(0),
            "Capacity pressure": ranked["score_capacity_pressure"].round(0),
            "Flight activity": ranked["score_operations"].round(0),
            "Enplanements CY25": ranked["enplanements"].map(fmt_num),
            "Pax growth": ranked["enplanement_growth"].map(fmt_signed_pct),
        })
        txt += _footer(f"{PERIOD_PAX}; {PERIOD_OPS}", cite("faa_enplanements", "bts_ontime", "ourairports"),
                       [SCORE_CAVEAT, CONGESTION_CAVEAT, COVERAGE_CAVEAT,
                        "Weights are prototype assumptions (adjust in the sidebar or ask 'what if capacity weight is 50%')."],
                       _conf_summary(list(ranked["confidence"])))
        return AgentResponse(note + txt, tables=[("Ranking", table)],
                             charts=[charts.score_breakdown_bar(ranked, e.weights, f"Score breakdown - {area}")],
                             tool_calls=calls)

    def _h_compare(self, q: ParsedQuery, ctx) -> AgentResponse:
        e = self.engine
        codes = q.airports[:5]
        df = e.compare_airports(codes)
        scored = e.scored
        in_peer = [c for c in codes if c in scored.index]
        calls = [f"compare_airports({codes})"] + [f"assess_congestion({c!r})" for c in in_peer]
        focus = q.focus or "general"

        lines = []
        if len(in_peer) >= 2:
            a, b = in_peer[0], in_peer[1]
            sa, sb = scored.loc[a], scored.loc[b]
            if focus == "congestion":
                hi, lo = (a, b) if sa["score_capacity_pressure"] >= sb["score_capacity_pressure"] else (b, a)
                lines.append(f"**Bottom line:** {hi} shows more congestion pressure than {lo} "
                             f"(capacity-pressure score {scored.loc[hi, 'score_capacity_pressure']:.0f} vs "
                             f"{scored.loc[lo, 'score_capacity_pressure']:.0f} out of 100, relative to peer hubs).")
            else:
                hi, lo = (a, b) if sa["opportunity_score"] >= sb["opportunity_score"] else (b, a)
                lines.append(f"**Bottom line:** {hi} scores higher as a screening opportunity than {lo} "
                             f"({scored.loc[hi, 'opportunity_score']:.0f} vs {scored.loc[lo, 'opportunity_score']:.0f}).")
            diffs = []
            for sub in SUBMETRIC_SOURCE:
                pa, pb = sa[f"pct_{sub}"], sb[f"pct_{sub}"]
                if pd.notna(pa) and pd.notna(pb) and abs(pa - pb) >= 20:
                    diffs.append((abs(pa - pb), f"{SUBMETRIC_LABELS[sub]}: {a} P{pa:.0f} vs {b} P{pb:.0f}"))
            if focus == "congestion":
                diffs = [d for d in diffs if any(k in d[1] for k in ("delay", "taxi", "NAS", "peak"))] or diffs
            if diffs:
                lines.append("\n**Material differences** (>= 20 percentile points):\n" +
                             "\n".join(f"- {d}" for _, d in sorted(diffs, reverse=True)[:6]))
            ta, tb = e.table.loc[a], e.table.loc[b]
            if focus == "congestion":
                lines.append(
                    f"\n**Raw readings ({a} vs {b}):** departure delay rate {fmt_pct(ta['dep_delay_rate'])} vs "
                    f"{fmt_pct(tb['dep_delay_rate'])}; arrival delay rate {fmt_pct(ta['arr_delay_rate'])} vs "
                    f"{fmt_pct(tb['arr_delay_rate'])}; taxi-out {ta['avg_taxi_out_min']:.1f} vs {tb['avg_taxi_out_min']:.1f} min; "
                    f"NAS delay {ta['nas_delay_per_arrival']:.1f} vs {tb['nas_delay_per_arrival']:.1f} min/arrival; "
                    f"busiest-hour movements {ta['peak_hour_movements']:.0f} vs {tb['peak_hour_movements']:.0f}.")
            lines.append(f"\n**Scale:** {a} handled {fmt_num(ta['enplanements'])} passengers vs {fmt_num(tb['enplanements'])} "
                         f"at {b}; {fmt_num(ta['operated_departures'])} vs {fmt_num(tb['operated_departures'])} reporting-carrier departures.")
        else:
            lines.append("**Bottom line:** at least one airport is outside the scored peer set, so only raw metrics are compared.")
        notes = load_context_notes()
        hyp = [(c, n) for c in codes for n in notes.get(c, [])]
        if hyp:
            lines.append("\n**Context that may explain differences** (qualitative hypotheses, not used in scores):\n" +
                         "\n".join(f"- {c}: {n['note']} _[{n['source']}]({n['url']})_" for c, n in hyp))
        missing = {c: e.scored.loc[c, "missing_inputs"] if c in e.scored.index else ["not scored"] for c in codes}
        miss_txt = "; ".join(f"{c}: {', '.join(m)}" for c, m in missing.items() if m)
        caveats = [CONGESTION_CAVEAT, COVERAGE_CAVEAT,
                   "All airports are measured over the identical period and definitions."]
        if miss_txt:
            caveats.append(f"Missing inputs - {miss_txt}.")
        conf = ", ".join(f"{c}: {e.scored.loc[c, 'confidence'] if c in e.scored.index else 'Not scored'}" for c in codes)
        txt = "\n".join(lines) + _footer(f"{PERIOD_OPS}; {PERIOD_PAX}", cite("bts_ontime", "faa_enplanements"), caveats, conf)
        figs = [charts.congestion_percentiles(scored, in_peer) if focus == "congestion" else charts.component_comparison(scored, in_peer),
                charts.monthly_delay({c: e.trend(c) for c in codes})]
        return AgentResponse(txt, tables=[("Side-by-side (same period)", format_comparison(df))], charts=figs, tool_calls=calls)

    def _h_congestion(self, q: ParsedQuery, ctx) -> AgentResponse:
        if len(q.airports) >= 2:
            return self._h_compare(q.model_copy(update={"focus": "congestion"}), ctx)
        c = q.airports[0]
        e = self.engine
        if c not in e.scored.index:
            return self._h_profile(q, ctx)
        res = e.assess_congestion(c)
        comp = res["components"].copy()
        score = res["score"]
        level = "high" if score >= 70 else "moderate" if score >= 40 else "low"
        top = comp.sort_values("Peer percentile", ascending=False).iloc[0]
        txt = (f"**Bottom line:** {e.airport_label(c)} has a **{level}** capacity-pressure (congestion proxy) score of "
               f"**{score:.0f}/100** relative to peer hub airports. Strongest signal: {top['Indicator']} "
               f"(P{top['Peer percentile']:.0f}).\n\n**Evidence behind the score**")
        fmt = {"departure delay rate": fmt_pct, "arrival delay rate": fmt_pct,
               "taxi-out time": lambda v: f"{v:.1f} min", "airspace/airport (NAS) delay": lambda v: f"{v:.1f} min / arrival",
               "peak-hour schedule intensity": lambda v: f"{v:.0f} movements in busiest hour"}
        comp["Value"] = [fmt[i](v) if pd.notna(v) else "MISSING" for i, v in zip(comp["Indicator"], comp["Value"])]
        comp["Weight"] = comp["Weight"].map(lambda w: f"{w:.0%}")
        comp["Peer percentile"] = comp["Peer percentile"].map(lambda p: "n/a" if pd.isna(p) else f"P{p:.0f}")
        txt += "\n" + "\n".join(f"- {r['Indicator']}: {r['Value']} ({r['Peer percentile']}, weight {r['Weight']})" for _, r in comp.iterrows())
        txt += _footer(PERIOD_OPS, cite("bts_ontime"), [CONGESTION_CAVEAT, COVERAGE_CAVEAT],
                       f"{e.scored.loc[c, 'confidence']} - " + ("; ".join(e.scored.loc[c, "confidence_reasons"]) or "all inputs present, full-year coverage"))
        return AgentResponse(txt, tables=[("Congestion evidence", comp)],
                             charts=[charts.congestion_percentiles(e.scored, [c]), charts.monthly_delay({c: e.trend(c)})],
                             tool_calls=[f"assess_congestion({c!r})"])

    def _h_long_haul(self, q: ParsedQuery, ctx) -> AgentResponse:
        c = q.airports[0]
        e = self.engine
        thr = q.threshold_miles or LONG_HAUL_THRESHOLD_MILES
        r = e.long_haul(c, thr)
        calls = [f"long_haul_share({c!r}, threshold_miles={thr:.0f})"]
        if r.share_pct is None:
            return AgentResponse(f"**Bottom line:** I can't compute a long-haul share for {c}: no eligible departures "
                                 f"with known destinations in CY{CURRENT_YEAR}.", tool_calls=calls)
        txt = (f"**Bottom line:** **{r.share_pct:.1f}%** of eligible departures from {e.airport_label(c)} were long-haul "
               f"(>= {thr:,.0f} statute miles) in CY{r.year}.\n\n"
               f"**Calculation:** {r.long_haul_departures:,} long-haul departures / {r.eligible_departures:,} eligible departures\n"
               f"- Scheduled departures: {r.scheduled_departures:,}\n"
               f"- Excluded - cancelled (never flew): {r.cancelled_excluded:,}\n"
               f"- Excluded - destination without coordinates: {r.missing_coordinates_excluded:,}\n"
               f"- Diverted flights are kept (they departed for the scheduled destination); duplicate flight records "
               f"were removed at ingestion; codeshares are not double counted because BTS records each flight once "
               f"under the operating carrier.\n"
               f"- Distance: great-circle from OurAirports coordinates (max difference vs BTS distance field: "
               f"{r.max_distance_discrepancy_miles:.0f} mi).\n"
               f"- Data coverage: {r.months_covered}/12 months.\n\n**Threshold sensitivity:** " +
               ", ".join(f">= {t:,.0f} mi: {s:.1f}%" for t, s in r.sensitivity.items() if s is not None))
        caveats = [f"Long-haul threshold = {thr:,.0f} mi (EUROCONTROL's 4,000 km long-haul boundary by default; configurable).",
                   "Only domestic flights by BTS reporting carriers are included: international flights and intra-Alaska "
                   "regional carriers are not in this dataset, so the share is of covered departures, not all departures."]
        if c == "ANC":
            caveats.append("ANC's large all-cargo operation is not reflected (passenger reporting carriers only).")
        txt += _footer(f"Jan-Dec {r.year}", cite("bts_ontime", "ourairports"), caveats,
                       "High for the calculation itself; Medium as a description of all ANC traffic because of coverage limits"
                       if c == "ANC" else "High for covered flights; coverage limits apply")
        top = r.top_long_haul_routes.copy()
        top["distance_miles"] = top["distance_miles"].round(0)
        return AgentResponse(txt, tables=[("Top long-haul destinations", top)], tool_calls=calls)

    def _h_demand(self, q: ParsedQuery, ctx) -> AgentResponse:
        c = q.airports[0]
        e = self.engine
        d = e.assess_demand(c)
        txt = (f"**Bottom line:** demand pressure at {e.airport_label(c)} is **{d.pressure_level}** on proxy evidence "
               f"({d.pressure_rationale}). **Unmet demand cannot be quantified numerically with the available data** - "
               f"no estimate is given.\n\n**1. Observed traffic**\n" +
               "\n".join(f"- {k}: {v}" for k, v in d.observed.items()) +
               "\n\n**2. Growth & seasonality**\n" + "\n".join(f"- {k}: {v}" for k, v in d.growth.items()) +
               "\n\n**3. Capacity-constraint evidence (proxies)**\n" + "\n".join(f"- {x}" for x in d.constraint_evidence))
        if d.hypotheses:
            txt += ("\n\n**4. What might explain it** (hypotheses from public context - verify; not used in scores)\n" +
                    "\n".join(f"- {h['note']} _[{h['source']}]({h['url']})_" for h in d.hypotheses))
        else:
            txt += "\n\n**4. What might explain it:** no curated context for this airport; the proxies above are the evidence."
        txt += (f"\n\n**5. Actual unmet demand:** {d.unmet_demand_statement}\n\n**Data needed to quantify it**\n" +
                "\n".join(f"- **{n}** - {why}" for n, why in d.required_data))
        conf = (e.scored.loc[c, "confidence"] if c in e.scored.index else "Low") + \
            " for the proxy readings; no quantitative confidence possible for unmet demand itself"
        txt += _footer(f"{PERIOD_PAX}; {PERIOD_OPS}", cite("faa_enplanements", "bts_ontime", "context_notes"),
                       ["Passenger growth is not unmet demand; delays are not proof of terminal congestion.",
                        CONGESTION_CAVEAT, COVERAGE_CAVEAT], conf)
        return AgentResponse(txt, charts=[charts.monthly_departures({c: e.trend(c)})],
                             tool_calls=[f"assess_demand({c!r})", f"assess_congestion({c!r})"])

    def _h_profile(self, q: ParsedQuery, ctx) -> AgentResponse:
        if len(q.airports) >= 2:
            return self._h_compare(q, ctx)
        c = q.airports[0]
        e = self.engine
        m = e.get_airport_metrics(c)
        hub = {"L": "Large hub", "M": "Medium hub", "S": "Small hub", "N": "Non-hub"}.get(m.get("hub"), "n/a")
        if "opportunity_score" in m and pd.notna(m.get("opportunity_score")):
            head = (f"**Bottom line:** {e.airport_label(c)} ({hub}) scores **{m['opportunity_score']:.0f}/100** "
                    f"(#{int(m['rank'])} of {int((e.scored['status'] == 'scored').sum())}). {m['explanation']}")
            conf = f"{m['confidence']}" + (f" - {'; '.join(m['confidence_reasons'])}" if m['confidence_reasons'] else "")
        else:
            head = (f"**Bottom line:** {e.airport_label(c)} ({hub}) is outside the scored peer set (FAA hub airports with "
                    f">= 1,000 reporting-carrier departures), so it has no opportunity score; raw metrics below.")
            conf = "Low - limited reporting-carrier coverage"
        txt = head + "\n\n" + "\n".join([
            f"- Passengers (enplanements) CY{CURRENT_YEAR}: {fmt_num(m.get('enplanements'))} ({fmt_signed_pct(m.get('enplanement_growth'))} YoY; {fmt_pct(m.get('recovery_vs_2019'), 0)} of 2019)",
            f"- Operated departures: {fmt_num(m.get('operated_departures'))} ({fmt_signed_pct(m.get('departure_growth'))} YoY)",
            f"- Departure delay rate: {fmt_pct(m.get('dep_delay_rate'))}; arrival delay rate: {fmt_pct(m.get('arr_delay_rate'))}",
            f"- Avg taxi-out: {fmt_num(m.get('avg_taxi_out_min'), 1)} min; NAS delay: {fmt_num(m.get('nas_delay_per_arrival'), 1)} min/arrival",
            f"- Peak-hour scheduled movements: {fmt_num(m.get('peak_hour_movements'), 0)}",
        ])
        if "score_demand" in m:
            txt += "\n\n**Component scores:** " + ", ".join(
                f"{COMPONENT_LABELS[k]} {m[f'score_{k}']:.0f}" for k in COMPONENTS if pd.notna(m.get(f"score_{k}")))
        txt += _footer(f"{PERIOD_PAX}; {PERIOD_OPS}", cite("faa_enplanements", "bts_ontime"),
                       [SCORE_CAVEAT, COVERAGE_CAVEAT], conf)
        return AgentResponse(txt, charts=[charts.monthly_departures({c: e.trend(c)})],
                             tool_calls=[f"get_airport_metrics({c!r})"])

    def _scenario_weights(self, q: ParsedQuery) -> dict[str, float]:
        base = dict(self.engine.weights)
        changes = q.weight_changes()
        if q.weight_mode == "absolute":
            fixed = {k: max(0.0, min(1.0, v)) for k, v in changes.items()}
            rest = [k for k in base if k not in fixed]
            remaining = max(0.0, 1 - sum(fixed.values()))
            rest_total = sum(base[k] for k in rest) or 1
            w = {**{k: base[k] / rest_total * remaining for k in rest}, **fixed}
        else:
            w = {k: max(0.0, base[k] + changes.get(k, 0)) for k in base}
        return normalize_weights(w)

    def _h_sensitivity(self, q: ParsedQuery, ctx) -> AgentResponse:
        e = self.engine
        base_w = e.weights
        sens = e.sensitivity()
        impact = sens.scenario_impact
        wfmt = lambda w: ", ".join(f"{COMPONENT_LABELS[k].split(' (')[0]} {v:.0%}" for k, v in w.items())
        if q.weight_changes():
            new_w = self._scenario_weights(q)
            new = score_airports(e.table, new_w)
            base = e.scored
            top_base = base[base["status"] == "scored"].head(10)
            focus = list(dict.fromkeys(list(top_base.index) + [a for a in q.airports + ctx.last_airports if a in base.index]))
            tbl = pd.DataFrame({
                "Airport": [base.loc[i, "name"] for i in focus],
                "Base rank": [int(base.loc[i, "rank"]) for i in focus],
                "New rank": [int(new.loc[i, "rank"]) for i in focus],
                "Base score": [round(base.loc[i, "opportunity_score"], 1) for i in focus],
                "New score": [round(new.loc[i, "opportunity_score"], 1) for i in focus],
            }, index=focus)
            tbl["Change"] = tbl["Base rank"] - tbl["New rank"]
            new_top = set(new[new["rank"] <= 10].index)
            dropped = set(top_base.index) - new_top
            entered = new_top - set(top_base.index)
            txt = (f"**Bottom line:** with weights **{wfmt(new_w)}** (base: {wfmt(base_w)}), "
                   f"{10 - len(dropped)} of the base top-10 stay in the top 10.")
            if entered:
                txt += f" Entering the top 10: {', '.join(sorted(entered))}."
            if dropped:
                txt += f" Dropping out: {', '.join(sorted(dropped))}."
            moved = tbl.reindex(tbl["Change"].abs().sort_values(ascending=False).index).head(3)
            txt += "\n\n**Biggest moves:** " + "; ".join(f"{i} {r['Base rank']} -> {r['New rank']}" for i, r in moved.iterrows())
            txt += "\n\n_This is a what-if: the base weights (sidebar) are unchanged._"
            tables = [("What-if ranking", tbl)]
            calls = [f"score_airports(weights={ {k: round(v, 2) for k, v in new_w.items()} })"]
        else:
            most = impact.iloc[0]
            stable = sens.summary[(sens.summary["base_rank"] <= 10) & (sens.summary[f"in_top_{sens.top_n}_share"] == 1)]
            txt = (f"**Bottom line:** the ranking is most sensitive to **{most['scenario']}** (average shift of "
                   f"{most['avg_rank_shift_base_top']:.1f} ranks for the base top-10, {int(most['top_n_changes'])} top-10 changes). "
                   f"{len(stable)} of the base top-10 airports stay in the top 10 under every scenario "
                   f"({', '.join(stable.index)}), so those conclusions are robust to reasonable weight changes.\n\n"
                   f"Each component weight was moved by +/-10 percentage points (re-normalized), plus an equal-weights case. "
                   f"Base weights: {wfmt(base_w)}.")
            tables = [("Scenario impact", impact.round(2)),
                      ("Rank range (top 15)", sens.summary.head(15).round(2))]
            calls = ["sensitivity_analysis(delta=0.10)"]
        txt += _footer(PERIOD_OPS, cite("faa_enplanements", "bts_ontime"),
                       ["Weights are prototype assumptions, not validated investment parameters.", SCORE_CAVEAT],
                       "Deterministic - re-running gives identical results")
        return AgentResponse(txt, tables=tables, charts=[charts.rank_range(sens.summary)], tool_calls=calls)

    def _h_why_rank(self, q: ParsedQuery, ctx) -> AgentResponse:
        e = self.engine
        codes = [c for c in q.airports if c in e.scored.index and e.scored.loc[c, "status"] == "scored"]
        if not codes:
            return self._h_clarify(q, ctx)
        if len(codes) == 1:
            c = codes[0]
            r = e.scored.loc[c]
            contrib = {COMPONENT_LABELS[k]: v for k, v in component_contributions(e.scored, c, e.weights).items()}
            txt = (f"**Bottom line:** {c} ranks #{int(r['rank'])} with {r['opportunity_score']:.0f}/100. {r['explanation']}\n\n"
                   "**Points contributed by each component:**\n" + "\n".join(f"- {k}: {v:.1f}" for k, v in contrib.items()))
        else:
            a, b = sorted(codes[:2], key=lambda x: e.scored.loc[x, "rank"])
            w = e.why_rank(a, b)
            ra, rb = e.scored.loc[a], e.scored.loc[b]
            main = max(w["diffs"].items(), key=lambda kv: kv[1])
            txt = (f"**Bottom line:** {a} (#{int(ra['rank'])}, {ra['opportunity_score']:.1f}) ranks above {b} "
                   f"(#{int(rb['rank'])}, {rb['opportunity_score']:.1f}) mainly because of **{COMPONENT_LABELS[main[0]]}** "
                   f"(+{main[1]:.1f} weighted points).\n\n**Weighted point difference by component ({a} minus {b}):**\n" +
                   "\n".join(f"- {COMPONENT_LABELS[k]}: {v:+.1f}" for k, v in w["diffs"].items()) +
                   "\n\n**Largest input differences (peer percentiles):**\n" +
                   "\n".join(f"- {lbl}: {a} P{pa:.0f} vs {b} P{pb:.0f}" for lbl, _, pa, pb in w["sub_diffs"]))
        txt += _footer(f"{PERIOD_PAX}; {PERIOD_OPS}", cite("faa_enplanements", "bts_ontime"), [SCORE_CAVEAT],
                       "Deterministic decomposition of the score")
        return AgentResponse(txt, charts=[charts.component_comparison(e.scored, codes[:2])],
                             tool_calls=[f"explain_rank({codes[:2]})"])

    def _h_sources(self, q: ParsedQuery, ctx) -> AgentResponse:
        reg = metrics_registry()
        focus = q.focus or ctx.last_focus
        if focus == "congestion":
            reg = reg[reg["Metric"].str.contains("delay|taxi|Peak|Capacity", case=False)]
        elif focus == "long_haul":
            reg = reg[reg["Metric"].str.contains("Long|departures", case=False)]
        elif focus == "demand":
            reg = reg[reg["Metric"].str.contains("Enplanement|Recovery|demand|departures", case=False)]
        txt = ("**Bottom line:** every number comes from public U.S. government data processed by deterministic code; "
               "nothing is generated by the language model.\n\n"
               "- **BTS On-Time Performance** - flights, cancellations, delays, taxi times, routes (Jan 2024-Dec 2025)\n"
               "- **FAA Passenger Boarding Data** - annual enplanements (CY2019, CY2024, CY2025)\n"
               "- **OurAirports** - names, states, coordinates (for distances)\n"
               "- **Curated context notes** - qualitative hypotheses only, each with a citation\n\n"
               f"Snapshot built: {self.engine.ds.built_at}. All data is **historical, not real-time**.")
        return AgentResponse(txt, tables=[("Sources", sources_table(self.engine.ds.manifest)), ("Metric registry", reg)],
                             tool_calls=["get_sources()"])

    def _h_data_needed(self, q: ParsedQuery, ctx) -> AgentResponse:
        needed = self.engine.data_needed()
        txt = ("**Bottom line:** the current model is a screening tool. To validate an actual investment you would need "
               "evidence on (a) real unmet demand, (b) physical / regulatory capacity, and (c) project economics:\n")
        for section, items in needed.items():
            txt += f"\n**{section}**\n" + "\n".join(f"- **{n}** - {why}" for n, why in items) + "\n"
        txt += ("\n**Also improves this prototype:** BTS T-100 international segments (domestic vs. international mix), "
                "intra-Alaska / regional carrier data, FAA OPSNET operations counts (all aircraft incl. cargo / GA).")
        airports = q.airports or ctx.last_airports
        notes = load_context_notes()
        specific = [(c, n) for c in airports for n in notes.get(c, [])]
        if specific:
            txt += "\n\n**Specific to the airports discussed:**\n" + "\n".join(f"- {c}: {n['note']}" for c, n in specific)
        return AgentResponse(txt, tool_calls=["data_needed()"])

    def _h_methodology(self, q: ParsedQuery, ctx) -> AgentResponse:
        w = self.engine.weights
        txt = ("**How the opportunity score works** (deterministic, reproducible):\n"
               "1. Peer set: FAA hub airports (large / medium / small) with >= 1,000 reporting-carrier departures.\n"
               "2. Each input is converted to a **percentile vs. peers** (0 = lowest, 100 = highest).\n"
               "3. Inputs roll up into three components:\n"
               "   - **Passenger demand & growth** - enplanements, YoY growth, recovery vs 2019\n"
               "   - **Capacity pressure** - departure / arrival delay rates, taxi-out time, NAS delay, peak-hour intensity\n"
               "   - **Flight activity & growth** - operated departures and YoY growth\n"
               f"4. Weighted total (current weights: {', '.join(f'{COMPONENT_LABELS[k]} {v:.0%}' for k, v in w.items())}).\n"
               "5. **Evidence confidence** (High / Medium / Low) is reported separately and does NOT change the score.\n"
               "6. Missing inputs are dropped and the remaining weights re-normalized (no reward or penalty); "
               "missing demand or capacity evidence -> 'insufficient evidence', no score.\n\n" + SCORE_CAVEAT)
        return AgentResponse(txt, tool_calls=["get_methodology()"])

    def _help_text(self) -> str:
        return ("I'm an airport-investment screening analyst for **U.S. airports**. I can rank opportunities, compare "
                "airports, show congestion and demand evidence, compute long-haul shares, run what-if weight scenarios, "
                "and tell you where every number comes from. Informal questions are fine. Try:\n" +
                "\n".join(f"- _{x}_" for x in EXAMPLES))

    def _h_help(self, q: ParsedQuery, ctx) -> AgentResponse:
        return AgentResponse(self._help_text())

    def _h_off_topic(self, q: ParsedQuery, ctx) -> AgentResponse:
        return AgentResponse("That's outside what I can help with - I only analyze **U.S. airport investment "
                             f"opportunities** from public aviation data, so I'd rather not guess.\n\n" + self._help_text())

    def _h_unsupported_aviation(self, q: ParsedQuery, ctx) -> AgentResponse:
        subj = q.off_topic_subject or "that"
        txt = (f"I don't have data on **{subj}** - this tool uses historical BTS / FAA statistics (calendar years "
               f"2024-2025), not live feeds, fares or forecasts, so I won't guess.")
        if q.airports:
            txt += f"\n\nWhat I *can* tell you about {', '.join(q.airports)}: its passenger volume, growth, congestion " \
                   f"evidence and investment score - e.g. ask _\"how congested is {q.airports[0]}\"_."
        return AgentResponse(txt)

    def _h_clarify(self, q: ParsedQuery, ctx) -> AgentResponse:
        return AgentResponse("Which airport do you mean? Give me an IATA code or a name - e.g. _\"how congested is BOS\"_, "
                             "_\"long haul share ANC\"_ or _\"compare LAX and SNA\"_.")
