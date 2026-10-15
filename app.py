"""Airport Investment Intelligence Agent - Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.agent.intents import ConversationContext, ParsedQuery
from src.agent.llm import make_llm_parser
from src.agent.orchestrator import EXAMPLES, AnalystAgent, format_comparison
from src.agent.tools import AnalyticsEngine
from src.analytics.metrics import fmt_num, fmt_pct, fmt_signed_pct
from src.analytics.scoring import (COMPONENT_LABELS, CONGESTION_SUBWEIGHTS, DEMAND_SUBWEIGHTS,
                                   OPERATIONS_SUBWEIGHTS, SUBMETRIC_LABELS, normalize_weights, score_band)
from src.config import (CURRENT_YEAR, DEFAULT_WEIGHTS, LLM_ENABLED, LLM_MODEL, LONG_HAUL_THRESHOLD_MILES,
                        PRIOR_YEAR)
from src.data_sources.http import DataSourceError
from src.data_sources.repository import load_dataset
from src.regions import REGIONS
from src.sources import metrics_registry, sources_table
from src.ui import charts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="wide")

_app_bg = "#0e1117" if (st.context.theme.type or "light") == "dark" else "#ffffff"
st.markdown(f"<style>:root {{--app-bg: {_app_bg};}}</style>", unsafe_allow_html=True)
st.markdown("""
<style>
  .explain-card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 14px 16px; height: 100%;}
  .explain-card h4 {margin: 0 0 6px 0; font-size: 1rem;}
  .explain-card p {margin: 0; font-size: .9rem; opacity: .85;}
  .small-note {font-size: .82rem; opacity: .75;}
  /* Frozen tab bar: stays visible below Streamlit's header while the page scrolls. */
  .stTabs [role="tablist"] {
    position: sticky; top: 3.75rem; z-index: 990;
    background-color: var(--app-bg); padding-top: .25rem;
  }
  /* Frozen section headers just below the tab bar: any st.container(key="sticky_...").
     The rule targets Streamlit's same-height wrapper around the keyed container, so the
     header sticks within its parent block (a whole tab, or one Methodology section). */
  div:has(> [class*="st-key-sticky_"]) {
    position: sticky; top: calc(3.75rem + 2.75rem); z-index: 980;
    background-color: var(--app-bg); padding-bottom: .5rem;
    border-bottom: 1px solid rgba(128,128,128,.2);
  }
  .st-key-sticky_overview h1 {padding-top: .25rem; padding-bottom: .25rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------
# Data & engine (cached)
# ---------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data snapshot...")
def get_dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Scoring airports...")
def get_engine(weights_key: tuple[tuple[str, float], ...]) -> AnalyticsEngine:
    return AnalyticsEngine(get_dataset(), dict(weights_key))


@st.cache_resource
def get_llm():
    return make_llm_parser()


try:
    ds = get_dataset()
except DataSourceError as e:
    st.error(f"**Data snapshot not available.** {e}")
    st.stop()

# ---------------------------------------------------------------------------------------
# Sidebar: weights, data status, AI mode
# ---------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ Airport Investment\n### Intelligence Agent")
    st.caption("Decision-support prototype for screening U.S. airport modernization opportunities.")

    st.markdown("#### Scoring weights")
    if st.button("Reset to defaults", width="stretch"):
        for k, v in DEFAULT_WEIGHTS.items():
            st.session_state[f"w_{k}"] = int(v * 100)
    raw_w = {}
    for k, v in DEFAULT_WEIGHTS.items():
        st.session_state.setdefault(f"w_{k}", int(v * 100))
        raw_w[k] = st.slider(COMPONENT_LABELS[k], 0, 100, step=5, key=f"w_{k}")
    if sum(raw_w.values()) == 0:
        st.warning("All weights are zero - using defaults.")
        raw_w = {k: v * 100 for k, v in DEFAULT_WEIGHTS.items()}
    weights = normalize_weights(raw_w)
    st.caption("Normalized: " + " · ".join(f"{COMPONENT_LABELS[k].split(' (')[0].split(' &')[0]} {v:.0%}"
                                            for k, v in weights.items()))
    st.caption("Weights are prototype assumptions, not industry standards.")

    st.divider()
    st.markdown("#### Data snapshot")
    bts = ds.manifest.get("sources", {}).get("bts_ontime", {})
    st.markdown(f"- **Period:** CY{CURRENT_YEAR} vs CY{PRIOR_YEAR}\n"
                f"- **Flights analyzed:** {bts.get('rows_clean_total', 0) / 1e6:.1f}M\n"
                f"- **Built:** {ds.built_at[:10]}\n"
                f"- **Type:** historical, not real-time")

    st.divider()
    st.markdown("#### AI mode")
    llm = get_llm()
    if llm:
        st.success(f"Claude ({LLM_MODEL}) interprets questions. All numbers are computed deterministically.")
    else:
        st.info("Rule-based question parser (no API key set). Fully functional; add ANTHROPIC_API_KEY to .env "
                "for richer language understanding.")

engine = get_engine(tuple(sorted(weights.items())))
scored = engine.scored
ranked_all = scored[scored["status"] == "scored"]
agent = AnalystAgent(engine, llm)

all_codes = sorted(engine.table.index)
label_of = {c: f"{c} - {engine.table.loc[c, 'name']} ({engine.table.loc[c, 'city']}, {engine.table.loc[c, 'state']})"
            for c in all_codes}

tab_overview, tab_chat, tab_opps, tab_compare, tab_detail, tab_method = st.tabs(
    ["🏠 Overview", "🤖 AI Analyst", "📊 Opportunities", "⚖️ Compare", "🔎 Airport detail", "📚 Methodology & sources"])

# ---------------------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------------------
with tab_overview:
    with st.container(key="sticky_overview"):
        st.title("Where would airport capacity investment pay off?")
    st.markdown("This tool screens **U.S. hub airports** for terminal / capacity investment by combining passenger "
                "demand, operational congestion and flight activity from **public BTS and FAA data**. It points "
                "analysts to where due diligence is worth doing - it does not replace it.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Airports scored", f"{len(ranked_all)}", help="FAA large/medium/small hubs with >= 1,000 reporting-carrier departures")
    k2.metric("Flights analyzed", f"{bts.get('rows_clean_total', 0) / 1e6:.1f}M", help="BTS on-time records, Jan 2024 - Dec 2025")
    k3.metric("Observation period", f"CY{CURRENT_YEAR}", help=f"Growth measured vs CY{PRIOR_YEAR}; passenger recovery vs 2019")
    k4.metric("Top opportunity (current weights)",
              f"{ranked_all.index[0]} · {ranked_all.iloc[0]['opportunity_score']:.0f}", help=ranked_all.iloc[0]["name"])

    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='explain-card'><h4>🎯 Opportunity score (0-100)</h4><p>Where an airport sits versus its "
                "peers on demand, congestion pressure and flight activity. 65+ = strong, 50-65 = moderate, "
                "&lt;50 = weak. Relative, not absolute.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='explain-card'><h4>🧪 Evidence confidence</h4><p>How much the data can be trusted for "
                "that airport (coverage, sample size, missing inputs). Reported separately - it never "
                "raises or lowers the score.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='explain-card'><h4>⚠️ What it is not</h4><p>Not an ROI, NPV or profitability forecast. "
                "Returns need capex, revenues, financing and regulatory analysis - see "
                "'what data would we need' in the AI Analyst.</p></div>", unsafe_allow_html=True)
    st.write("")
    m1, m2 = st.columns([3, 2])
    with m1:
        st.plotly_chart(charts.opportunity_map(scored), key="ov_map")
    with m2:
        st.plotly_chart(charts.score_breakdown_bar(ranked_all.head(10), weights, "Top 10 - what drives the score"), key="ov_top10")

    st.markdown("#### Airport search")
    q = st.text_input("Find an airport by code, name or city", placeholder="e.g. Boston, SNA, Anchorage")
    if q:
        hits = engine.search_airports(q)
        if not hits:
            st.warning("No matching U.S. airport in the dataset.")
        for h in hits:
            s = scored.loc[h.iata] if h.iata in scored.index else None
            extra = (f"score **{s['opportunity_score']:.0f}** ({score_band(s['opportunity_score'])}), rank #{int(s['rank'])}, "
                     f"confidence {s['confidence']}" if s is not None else "not in scored peer set")
            st.markdown(f"- **{h.iata}** {h.name} - {h.city}, {h.state}: {extra}")

# ---------------------------------------------------------------------------------------
# AI Analyst (chat)
# ---------------------------------------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.ctx = ConversationContext()

    with st.container(key="sticky_chat"):  # sticky (see CSS): examples stay reachable while scrolling
        st.subheader("Ask the AI analyst")
        st.caption("Informal questions are fine (e.g. *compare congestion LAX SNA*). The assistant picks deterministic "
                   "analysis tools; it never invents numbers. Follow-ups use the conversation context.")
        cols = st.columns(3)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 3].button(ex, key=f"ex_{i}", width="stretch"):
                st.session_state.pending = ex
        if st.session_state.messages and st.button("🧹 Clear conversation"):
            st.session_state.messages = []
            st.session_state.ctx = ConversationContext()
            st.rerun()

    def render_answer(msg: dict, idx: int):
        st.markdown(msg["text"])
        for t_i, (title, df) in enumerate(msg.get("tables", [])):
            st.markdown(f"**{title}**")
            st.dataframe(df, key=f"chat_tbl_{idx}_{t_i}")
        for c_i, fig in enumerate(msg.get("charts", [])):
            st.plotly_chart(fig, key=f"chat_fig_{idx}_{c_i}")
        if msg.get("parsed"):
            with st.expander("🔧 How this was answered"):
                p = msg["parsed"]
                st.markdown(f"- **Interpreted as:** `{p.intent}` (parser: {p.parser})\n"
                            f"- **Airports:** {', '.join(p.airports) or '-'} · **Region:** {p.region or '-'}"
                            f"{' · used conversation context' if p.used_context else ''}\n"
                            f"- **Tools called:** " + (", ".join(f"`{t}`" for t in msg.get("tools", [])) or "none") +
                            f"\n- **Weights in effect:** " + ", ".join(f"{k} {v:.0%}" for k, v in weights.items()))

    chat_box = st.container()  # history renders here, above the input box
    with chat_box:
        for i, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"]):
                if m["role"] == "user":
                    st.markdown(m["text"])
                else:
                    render_answer(m, i)

    prompt = st.chat_input("Ask about U.S. airports - e.g. 'how jammed is SFO?'")
    prompt = prompt or st.session_state.pop("pending", None)
    if prompt:
        st.session_state.messages.append({"role": "user", "text": prompt})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    resp, st.session_state.ctx = agent.ask(prompt, st.session_state.ctx)
                msg = {"role": "assistant", "text": resp.text, "tables": resp.tables, "charts": resp.charts,
                       "tools": resp.tool_calls, "parsed": resp.parsed}
                render_answer(msg, len(st.session_state.messages))
        st.session_state.messages.append(msg)

# ---------------------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------------------
with tab_opps:
    with st.container(key="sticky_opps"):
        st.subheader("Investment opportunity ranking")
        f1, f2, f3 = st.columns(3)
        region = f1.selectbox("Region", ["All U.S."] + list(REGIONS))
        hubs = f2.multiselect("FAA hub size", ["L", "M", "S"], default=["L", "M", "S"],
                              format_func=lambda h: {"L": "Large", "M": "Medium", "S": "Small"}[h])
        conf = f3.multiselect("Evidence confidence", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    view = ranked_all[ranked_all["hub"].isin(hubs) & ranked_all["confidence"].isin(conf)]
    if region != "All U.S.":
        view = view[view["state"].isin(REGIONS[region])]
    table = pd.DataFrame({
        "Rank": view["rank"].astype(int), "Airport": view["name"], "State": view["state"],
        "Score": view["opportunity_score"].round(1), "Band": view["opportunity_score"].map(score_band),
        "Confidence": view["confidence"],
        "Demand": view["score_demand"].round(0), "Capacity pressure": view["score_capacity_pressure"].round(0),
        "Flight activity": view["score_operations"].round(0),
        "Passengers CY25": view["enplanements"], "Pax growth %": (view["enplanement_growth"] * 100).round(1),
        "Delay rate %": (view["dep_delay_rate"] * 100).round(1), "Taxi-out (min)": view["avg_taxi_out_min"].round(1),
        "Why": view["explanation"],
    })
    st.dataframe(table, height=520, column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Passengers CY25": st.column_config.NumberColumn(format="localized"),
        "Why": st.column_config.TextColumn(width="large"),
    })
    st.caption(f"{len(view)} airports shown. Component scores are 0-100 percentiles vs all {len(ranked_all)} peers. "
               "Capacity pressure = congestion proxies (delay, taxi-out, NAS delay, peak-hour intensity), not utilization.")

    with st.expander("🧪 Sensitivity analysis - does the ranking survive different weights?", expanded=False):
        sens = engine.sensitivity()
        s1, s2 = st.columns([3, 2])
        s1.plotly_chart(charts.rank_range(sens.summary), key="sens_range")
        s2.markdown("**Impact of each +/-10pp weight change** (on the base top-10)")
        s2.dataframe(sens.scenario_impact.rename(columns={"avg_rank_shift_base_top": "Avg rank shift",
                                                          "top_n_changes": "Top-10 changes"}).round(2), hide_index=True)
        stable = sens.summary[(sens.summary["base_rank"] <= 10) & (sens.summary["in_top_10_share"] == 1)]
        s2.success(f"{len(stable)} of the top 10 stay in the top 10 under every scenario: {', '.join(stable.index)}")

# ---------------------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------------------
with tab_compare:
    with st.container(key="sticky_compare"):
        st.subheader("Side-by-side comparison (identical period and definitions)")
        sel = st.multiselect("Airports", all_codes, default=["LAX", "SNA"], format_func=lambda c: label_of[c],
                             max_selections=5)
    if len(sel) < 2:
        st.info("Select at least two airports.")
    else:
        st.dataframe(format_comparison(engine.compare_airports(sel)), height=640)
        st.caption(f"Flights: Jan-Dec {CURRENT_YEAR} vs Jan-Dec {PRIOR_YEAR} (BTS). Passengers: CY{CURRENT_YEAR} (FAA). "
                   "MISSING = input not available for that airport.")
        in_peer = [c for c in sel if c in scored.index]
        a, b = st.columns(2)
        a.plotly_chart(charts.component_comparison(scored, in_peer), key="cmp_comp")
        b.plotly_chart(charts.congestion_percentiles(scored, in_peer), key="cmp_cong")
        trends = {c: engine.trend(c) for c in sel}
        a2, b2 = st.columns(2)
        a2.plotly_chart(charts.monthly_departures(trends), key="cmp_dep")
        b2.plotly_chart(charts.monthly_delay(trends), key="cmp_delay")
        with st.expander("📝 Analyst summary of material differences", expanded=True):
            r = agent.run(ParsedQuery(intent="compare", airports=sel, focus="general"))
            st.markdown(r.text)

# ---------------------------------------------------------------------------------------
# Airport detail
# ---------------------------------------------------------------------------------------
with tab_detail:
    with st.container(key="sticky_detail"):
        code = st.selectbox("Airport", all_codes, index=all_codes.index("SFO") if "SFO" in all_codes else 0,
                            format_func=lambda c: label_of[c])
        m = engine.get_airport_metrics(code)
        hub = {"L": "Large hub", "M": "Medium hub", "S": "Small hub", "N": "Non-hub"}.get(m.get("hub"), "n/a")
        st.subheader(f"{code} · {m['name']}")
    st.caption(f"{m['city']}, {m['state']} · FAA {hub} · lat {m['lat']:.3f}, lon {m['lon']:.3f}")
    d1, d2, d3, d4, d5 = st.columns(5)
    if pd.notna(m.get("opportunity_score")):
        d1.metric("Opportunity score", f"{m['opportunity_score']:.0f}", score_band(m["opportunity_score"]), delta_color="off")
        d2.metric("National rank", f"#{int(m['rank'])}", f"of {len(ranked_all)}", delta_color="off")
        d3.metric("Evidence confidence", m["confidence"])
    else:
        d1.metric("Opportunity score", "not scored")
        d2.metric("National rank", "-")
        d3.metric("Evidence confidence", "-")
    d4.metric(f"Passengers CY{CURRENT_YEAR}", fmt_num(m.get("enplanements")), fmt_signed_pct(m.get("enplanement_growth")))
    d5.metric("Operated departures", fmt_num(m.get("operated_departures")), fmt_signed_pct(m.get("departure_growth")))

    if pd.notna(m.get("opportunity_score")):
        st.markdown(f"**Why:** {m['explanation']}")
        if m["confidence_reasons"]:
            st.caption("Confidence notes: " + "; ".join(m["confidence_reasons"]))
    trends = {code: engine.trend(code)}
    t1, t2 = st.columns(2)
    t1.plotly_chart(charts.monthly_departures(trends), key="det_dep")
    t2.plotly_chart(charts.monthly_delay(trends), key="det_delay")

    if code in scored.index:
        st.markdown("#### Score breakdown & congestion evidence")
        b1, b2 = st.columns(2)
        b1.plotly_chart(charts.score_breakdown_bar(scored.loc[[code]], weights, "Weighted points by component"), key="det_breakdown")
        cg = engine.assess_congestion(code)["components"].copy()
        cg["Peer percentile"] = cg["Peer percentile"].round(0)
        cg["Weight"] = cg["Weight"].map(lambda w: f"{w:.0%}")
        cg["Value"] = cg["Value"].round(3)
        b2.markdown(f"**Capacity-pressure score: {m['score_capacity_pressure']:.0f}/100**")
        b2.dataframe(cg, hide_index=True)
        b2.caption("Delay rates are fractions (0.20 = 20%). Proxies only - not terminal utilization.")

        st.markdown("#### Demand evidence")
        dem = engine.assess_demand(code)
        st.markdown(f"**Proxy-based demand pressure: {dem.pressure_level}** - {dem.pressure_rationale}.")
        e1, e2 = st.columns(2)
        e1.markdown("\n".join(f"- {k}: **{v}**" for k, v in {**dem.observed, **dem.growth}.items()))
        e2.markdown("\n".join(f"- {x}" for x in dem.constraint_evidence))
        if dem.hypotheses:
            st.markdown("**Possible explanations (hypotheses from public context - verify):**\n" +
                        "\n".join(f"- {h['note']} _[{h['source']}]({h['url']})_" for h in dem.hypotheses))
        st.info(dem.unmet_demand_statement)

    st.markdown("#### Long-haul share of departures")
    thr = st.slider("Long-haul threshold (statute miles)", 500, 4000, int(LONG_HAUL_THRESHOLD_MILES), 50,
                    help="Default 2,485 mi = EUROCONTROL's 4,000 km long-haul boundary")
    lh = engine.long_haul(code, thr)
    if lh.share_pct is None:
        st.warning("No eligible departures in the BTS data for this airport.")
    else:
        l1, l2, l3 = st.columns(3)
        l1.metric("Long-haul share", f"{lh.share_pct:.1f}%")
        l2.metric("Long-haul / eligible departures", f"{lh.long_haul_departures:,} / {lh.eligible_departures:,}")
        l3.metric("Excluded (cancelled / no coords)", f"{lh.cancelled_excluded:,} / {lh.missing_coordinates_excluded:,}")
    st.caption("Domestic reporting-carrier flights only. " + metrics_registry().set_index("Metric").loc["Long-haul share", "Definition"])

# ---------------------------------------------------------------------------------------
# Methodology & sources
# ---------------------------------------------------------------------------------------
def method_section(key: str, title: str):
    """A section whose heading stays pinned only while the section itself is on screen."""
    section = st.container(key=f"msec_{key}")
    with section.container(key=f"sticky_m_{key}"):
        st.subheader(title)
    return section


with tab_method:
    with method_section("sources", "Data sources (all public, historical)"):
        st.dataframe(sources_table(ds.manifest), hide_index=True, column_config={"URL": st.column_config.LinkColumn()})
    with method_section("registry", "Metric registry - observed vs derived"):
        st.dataframe(metrics_registry(), hide_index=True)
    with method_section("scoring", "Scoring model"):
        st.markdown(
            "1. **Peer set:** FAA large / medium / small hubs with >= 1,000 reporting-carrier departures in "
            f"CY{CURRENT_YEAR}.\n2. **Normalization:** every input -> percentile among peers (0-100).\n"
            "3. **Components** (sub-weights below) -> **weighted total** using the sidebar weights.\n"
            "4. **Missing inputs** are dropped and the remaining weights re-normalized; missing demand or capacity "
            "evidence -> *insufficient evidence* (no score).\n5. **Confidence** is separate from the score.\n"
            "6. **Sensitivity:** each weight +/-10pp and an equal-weights case (Opportunities tab).")
        sub = pd.DataFrame([(COMPONENT_LABELS[c], SUBMETRIC_LABELS[s], f"{w:.0%}") for c, d in
                            (("demand", DEMAND_SUBWEIGHTS), ("capacity_pressure", CONGESTION_SUBWEIGHTS),
                             ("operations", OPERATIONS_SUBWEIGHTS)) for s, w in d.items()],
                           columns=["Component", "Input", "Weight within component"])
        st.dataframe(sub, hide_index=True)
    with method_section("limits", "Known limitations"):
        st.markdown(
            "- Flight data covers **domestic flights by BTS reporting carriers** only (no international, most regional "
            "or cargo flights). Airports with large international traffic get *Medium* confidence.\n"
            "- **Congestion is proxied** by delays, taxi-out, NAS delay and peak-hour intensity. Terminal crowding, gate "
            "utilization and runway capacity are not measured.\n"
            "- **Unmet demand is never estimated numerically** - it needs load factors, fares, slot/gate requests and "
            "declared capacity.\n- **Feasibility** (land, regulation, existing capital programs) and **financial returns** "
            "are not modeled.\n- Data is **historical** (calendar years) and refreshed by `python scripts/build_dataset.py`.")
    st.markdown("<p class='small-note'>Full write-up: docs/architecture.md and docs/data_sources.md in the repository.</p>",
                unsafe_allow_html=True)
