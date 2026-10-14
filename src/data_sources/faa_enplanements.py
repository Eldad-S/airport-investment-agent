"""FAA Passenger Boarding (Enplanement) Data for U.S. airports - calendar-year spreadsheets.

Source: https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger
Public, no authentication. Derived from the FAA Air Carrier Activity Information System
(ACAIS); covers commercial-service airports (>= 2,500 annual enplanements), all carriers,
domestic AND international passengers combined. The FAA identifier (Locid) is an FAA LID,
which usually - but not always - equals the IATA code; we crosswalk via OurAirports.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import BASELINE_YEAR, CURRENT_YEAR, RAW_DIR
from src.data_sources.http import DataSourceError, download

log = logging.getLogger(__name__)

SOURCE_ID = "faa_enplanements"
_BASE = "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger"
_PREV = "https://www.faa.gov/sites/faa.gov/files/airports/planning_capacity/passenger_allcargo_stats/passenger"
URLS = {
    # The CY2025 file contains CY25 and CY24 columns; the CY2019 file contains CY19 (baseline).
    CURRENT_YEAR: f"{_BASE}/arp-cy{CURRENT_YEAR}-commercial-service-enplanements.xlsx",
    BASELINE_YEAR: f"{_PREV}/cy{str(BASELINE_YEAR)[2:]}-commercial-service-enplanements.xlsx",
}


def fetch(year: int, force: bool = False) -> Path:
    return download(URLS[year], RAW_DIR / SOURCE_ID / f"cy{year}_enplanements.xlsx", force=force)


def parse_enplanement_sheet(path: Path) -> pd.DataFrame:
    """Parse an FAA CY enplanement workbook into tidy columns. Robust to header-name drift."""
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Locid", "Hub", "S/L", "ST"}
    if not required.issubset(df.columns):
        raise DataSourceError(f"{path.name}: unexpected columns {list(df.columns)}")
    enpl_cols = [c for c in df.columns if c.upper().startswith("CY") and "ENPLANEMENT" in c.upper()]
    if not enpl_cols:
        raise DataSourceError(f"{path.name}: no 'CY xx Enplanements' columns found")
    rename = {c: f"enpl_cy20{c.split()[1]}" for c in enpl_cols}  # 'CY 25 Enplanements' -> enpl_cy2025
    out = df.rename(columns=rename)[["Locid", "Airport Name", "City", "ST", "S/L", "Hub", *rename.values()]]
    out = out.rename(columns={"Locid": "locid", "Airport Name": "faa_name", "City": "faa_city",
                              "ST": "faa_state", "S/L": "service_level", "Hub": "hub"})
    out["locid"] = out["locid"].astype(str).str.strip().str.upper()
    out["hub"] = out["hub"].fillna("N").astype(str).str.strip().replace({"None": "N", "nan": "N"})
    out = out[out["locid"].str.fullmatch(r"[A-Z0-9]{3,4}")]
    for c in rename.values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.drop_duplicates("locid")


def build_enplanements(current_path: Path, baseline_path: Path) -> pd.DataFrame:
    cur = parse_enplanement_sheet(current_path)
    base = parse_enplanement_sheet(baseline_path)
    base_col = f"enpl_cy{BASELINE_YEAR}"
    return cur.merge(base[["locid", base_col]], on="locid", how="left")
