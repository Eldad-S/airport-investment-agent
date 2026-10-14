"""Source registry: where every metric comes from, whether it is observed or derived, its
period and its limitations. Retrieval timestamps are read from the snapshot manifest."""
from __future__ import annotations

import pandas as pd

from src.config import BASELINE_YEAR, CURRENT_YEAR, LONG_HAUL_THRESHOLD_MILES, PRIOR_YEAR

SOURCES = {
    "bts_ontime": {
        "name": "BTS Reporting Carrier On-Time Performance",
        "provider": "U.S. DOT Bureau of Transportation Statistics (TranStats)",
        "url": "https://www.transtats.bts.gov/",
        "access": "Public monthly zip files, no authentication",
        "license": "U.S. government public data",
        "period": f"Jan {PRIOR_YEAR} - Dec {CURRENT_YEAR} (monthly)",
        "nature": "Historical (not real-time)",
        "limitations": "Domestic flights only; reporting carriers only (>= 0.5% of domestic scheduled "
                       "passenger revenue) - excludes most regional-only, charter, cargo and foreign "
                       "carriers. Delay is an operational proxy, not a capacity measurement.",
    },
    "faa_enplanements": {
        "name": "FAA Passenger Boarding (Enplanement) Data",
        "provider": "Federal Aviation Administration (ACAIS)",
        "url": "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger",
        "access": "Public calendar-year spreadsheets, no authentication",
        "license": "U.S. government public data",
        "period": f"CY{BASELINE_YEAR}, CY{PRIOR_YEAR}, CY{CURRENT_YEAR}",
        "nature": "Historical annual totals",
        "limitations": "Annual totals only (no monthly / hourly); domestic + international combined; "
                       "FAA identifiers crosswalked to IATA codes.",
    },
    "ourairports": {
        "name": "OurAirports airport reference data",
        "provider": "OurAirports (community-maintained)",
        "url": "https://ourairports.com/data/",
        "access": "Public CSV, no authentication",
        "license": "Public domain",
        "period": "Snapshot at retrieval time",
        "nature": "Reference data",
        "limitations": "Community-maintained; coordinates used for great-circle distances.",
    },
    "context_notes": {
        "name": "Curated airport context notes",
        "provider": "Analyst-curated from public airport / FAA publications (cited per note)",
        "url": "data/context/airport_notes.json",
        "access": "Bundled file",
        "license": "n/a",
        "period": "n/a",
        "nature": "Qualitative hypotheses - NOT used in any score",
        "limitations": "Must be verified; used only to suggest explanations.",
    },
}

METRICS = [
    # metric, sources, observed/derived, period, definition, limitation
    ("Passenger enplanements", "faa_enplanements", "Observed", f"CY{CURRENT_YEAR}",
     "Revenue passengers boarding at the airport", "Annual only"),
    ("Enplanement growth", "faa_enplanements", "Derived", f"CY{CURRENT_YEAR} vs CY{PRIOR_YEAR}",
     "current / prior - 1", "Passenger growth is not unmet demand"),
    ("Recovery vs 2019", "faa_enplanements", "Derived", f"CY{CURRENT_YEAR} vs CY{BASELINE_YEAR}",
     "current / 2019", ""),
    ("Operated departures", "bts_ontime", "Observed", f"CY{CURRENT_YEAR}",
     "Scheduled minus cancelled departures", "Reporting carriers, domestic only"),
    ("Departure growth", "bts_ontime", "Derived", f"CY{CURRENT_YEAR} vs CY{PRIOR_YEAR}",
     "Operated departures current / prior - 1; blank if either year is incomplete", ""),
    ("Departure delay rate", "bts_ontime", "Observed (rate derived)", f"CY{CURRENT_YEAR}",
     "Share of operated departures >= 15 min late", "Delays can reflect airline ops / weather, not only capacity"),
    ("Arrival delay rate", "bts_ontime", "Observed (rate derived)", f"CY{CURRENT_YEAR}",
     "Share of operated arrivals >= 15 min late", ""),
    ("Avg taxi-out time", "bts_ontime", "Observed (mean derived)", f"CY{CURRENT_YEAR}",
     "Mean gate-out to wheels-off minutes", "Proxy for surface/airfield congestion; layout-dependent"),
    ("NAS delay per arrival", "bts_ontime", "Derived", f"CY{CURRENT_YEAR}",
     "Minutes of delay attributed to the National Airspace System per operated arrival",
     "NAS includes non-extreme weather as well as volume and ATC"),
    ("Peak-hour movements", "bts_ontime", "Derived", f"CY{CURRENT_YEAR}",
     "Average over days of the busiest scheduled hour (deps + arrs)", "Reporting carriers only; schedule-based"),
    ("Long-haul share", "bts_ontime + ourairports", "Derived", f"CY{CURRENT_YEAR}",
     f"Departures with great-circle distance >= {LONG_HAUL_THRESHOLD_MILES:,.0f} mi / eligible departures",
     "Domestic reporting-carrier flights only"),
    ("Capacity-pressure score", "bts_ontime", "Derived (index)", f"CY{CURRENT_YEAR}",
     "Weighted percentile index of the five congestion proxies", "Relative to peers; not utilization"),
    ("Opportunity score", "all", "Derived (model)", f"CY{CURRENT_YEAR}",
     "Weighted percentile model, 0-100", "Screening signal - not ROI / NPV / IRR"),
    ("Unmet demand", "-", "Not available", "-",
     "Requires load factor, fares, slot / gate requests or capacity data", "Never estimated numerically"),
]


def metrics_registry() -> pd.DataFrame:
    return pd.DataFrame(METRICS, columns=["Metric", "Source", "Type", "Period", "Definition", "Limitations"])


def sources_table(manifest: dict | None = None) -> pd.DataFrame:
    manifest = manifest or {}
    rows = []
    for sid, s in SOURCES.items():
        retrieved = manifest.get("sources", {}).get(sid, {}).get("retrieved_at_utc", "n/a")
        rows.append({"Source": s["name"], "Provider": s["provider"], "Period": s["period"],
                     "Nature": s["nature"], "Retrieved (UTC)": retrieved, "Access": s["access"],
                     "Limitations": s["limitations"], "URL": s["url"]})
    return pd.DataFrame(rows)


def cite(*source_ids: str) -> str:
    return "; ".join(f"[{SOURCES[s]['name']}]({SOURCES[s]['url']})" if SOURCES[s]['url'].startswith("http")
                     else SOURCES[s]["name"] for s in source_ids)
