# ✈️ Airport Investment Intelligence Agent

A decision-support prototype that helps an investment team **screen U.S. airports for terminal / capacity
investment**. It combines real public aviation data (BTS + FAA), a **deterministic, explainable scoring
model**, and a **chat analyst** that understands informal questions such as *"compare congestion LAX SNA"*.

> **What it is:** a screening tool that tells analysts *where due diligence is worth doing*.
> **What it is not:** an ROI / NPV / profitability forecast or an investment recommendation.

---

## Quick start (about 2 minutes)

Requires Python 3.11 - 3.13 (tested on 3.12).

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
```
```bash
pip install -r requirements.txt
```
```bash
streamlit run app.py
```

Or one command: `./run.ps1` (Windows) / `./run.sh` (macOS/Linux) - creates the venv on first run and starts the app.

The app opens at http://localhost:8501. **No API key and no downloads are needed**: the processed real-data
snapshot (~1 MB) is committed in `data/processed/`.

**Optional - Claude for language understanding:** copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
Claude then interprets questions; every number is still computed by deterministic Python. Check it with
`python scripts/check_llm.py "how jammed is sfo"`. If the key is missing or the API fails, the built-in
rule-based parser takes over automatically.

**Run tests:** `pytest -q` (88 tests, no network access needed).

**Refresh the data from source:** `python scripts/build_dataset.py` (downloads ~750 MB of BTS files once,
cached in `data/raw/`, ~10 - 15 min).

---

## 5-minute demo script

| # | Where | What to do / ask | What it shows |
|---|---|---|---|
| 1 | 🏠 Overview | Point at the map, top-10 chart and the three "how to read" cards | Score vs. confidence vs. "not ROI" |
| 2 | 🤖 AI Analyst | *Which airports in New England are strong candidates for terminal expansion?* | Regional ranking, reasons, non-scored airports listed honestly |
| 3 | 🤖 AI Analyst | *compare congestion LAX SNA* → then *what about SAN* | Slang, same-period comparison, follow-up context |
| 4 | 🤖 AI Analyst | *% long haul out of anchorage* | Numerator / denominator / exclusions / threshold sensitivity |
| 5 | 🤖 AI Analyst | *unmet demand at SFO - what's going on?* | Evidence + hypotheses, **refuses to invent** an unmet-demand number |
| 6 | 🤖 AI Analyst | *what if capacity weight is 50%* → *which assumptions matter most* | What-if and sensitivity analysis |
| 7 | 🤖 AI Analyst | *tell me a joke* / *weather at LAX?* | Graceful handling of off-topic and no-data questions |
| 8 | Sidebar | Move the weight sliders | Live re-ranking across all tabs |
| 9 | 📊 / ⚖️ / 🔎 tabs | Ranking table, side-by-side compare, airport drill-down | Full transparency of every input |

Open "🔧 How this was answered" under any chat answer to show the parsed intent and the exact tools called.

---

## What it answers

| Assignment question | Status |
|---|---|
| New England terminal-expansion candidates | ✅ Ranked with reasons; BOS is the only "strong" candidate (#9 nationally); non-hub airports listed as not scored |
| Congestion LAX vs SNA | ✅ Capacity-pressure index + five raw proxies over the identical period, plus context (SNA's settlement agreement) |
| % long-haul departures at ANC | ✅ 18.5 % (3,446 / 18,650 eligible departures, CY2025, >= 2,485 mi) with coverage caveats |
| Unmet demand at SFO | ⚠️ By design no number - proxy evidence (Elevated pressure), hypotheses and the datasets needed to quantify it |
| Most promising airports per the model | ✅ ORD, DEN, MCO, BNA, MIA … with component breakdown; robust to +/-10pp weight changes |
| Additional data needed | ✅ Grouped: unmet demand, underwriting, and prototype improvements |

Follow-ups supported: *why did BOS rank above BDL*, *what if I increase capacity weight*, *which data sources
support this*, *compare both airports over the same period*, *show the evidence behind the congestion score*,
*which assumptions matter most*, *what about SAN*.

---

## How it works (one paragraph)

`scripts/build_dataset.py` downloads **14.1 M real flight records** (BTS On-Time Performance, Jan 2024 - Dec
2025), **FAA calendar-year passenger boardings** (CY2019/24/25) and **OurAirports** coordinates, cleans and
aggregates them into a small snapshot with a provenance manifest. The analytics layer computes metrics,
a congestion-proxy index and a 0-100 **opportunity score** (percentiles vs. 141 U.S. hub airports, weighted:
demand 40 %, capacity pressure 35 %, flight activity 25 %). The chat agent maps a question to one of ~12
deterministic tools; Claude (optional) only helps choose the tool. Every answer states the bottom line,
evidence, period, sources, caveats and confidence.

```
Public data ──► build_dataset.py ──► data/processed (snapshot + manifest)
                                          │
                       analytics: metrics · congestion · demand · scoring · sensitivity
                                          │
                 agent tools (deterministic) ◄── router (rules) / Claude (optional, intent only)
                                          │
                              Streamlit dashboard + chat
```

Details: [docs/architecture.md](docs/architecture.md) · [docs/data_sources.md](docs/data_sources.md)

## Project layout

```
app.py                       Streamlit UI (6 tabs + chat)
src/config.py                periods, thresholds, weights (all assumptions in one place)
src/data_sources/            BTS / FAA / OurAirports ingestion, cached HTTP, snapshot loader
src/analytics/               metrics.py, scoring.py (incl. congestion index + sensitivity), demand.py
src/agent/                   router.py (rules), llm.py (optional Claude), tools.py, orchestrator.py
src/sources.py               source + metric registry (observed vs derived, period, limitations)
data/processed/              committed real-data snapshot + manifest.json
data/context/                curated, cited qualitative notes (hypotheses only, never scored)
scripts/                     build_dataset.py, download_bts.py, check_llm.py
tests/                       88 unit / integration / evaluation tests
```

## Key limitations (be candid in the demo)

- Flight data = **domestic flights of BTS reporting carriers** only; international gateways (JFK, MIA, LAX…)
  are flagged *Medium* confidence because a large share of their passengers is not in the flight data.
- **Congestion is proxied** (delays, taxi-out, NAS delay, peak-hour schedule) - not terminal utilization.
- **Unmet demand is never estimated numerically**; feasibility (land, regulation, existing capex) and
  financial returns are out of scope.
- Weights are **prototype assumptions**; the sensitivity analysis shows how much they matter.
