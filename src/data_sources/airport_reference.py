"""OurAirports reference data: names, municipalities, states and coordinates.

Source: https://ourairports.com/data/ (public domain). Mirrored as CSV on GitHub Pages.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.data_sources.http import DataSourceError, download

SOURCE_ID = "ourairports"
URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
US_COUNTRIES = {"US", "PR", "VI", "GU", "AS", "MP"}
KEEP_TYPES = {"large_airport", "medium_airport", "small_airport", "seaplane_base"}


def fetch(force: bool = False) -> Path:
    return download(URL, RAW_DIR / SOURCE_ID / "airports.csv", force=force)


def build_reference(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    needed = {"iata_code", "local_code", "iso_country", "iso_region", "type", "name",
              "municipality", "latitude_deg", "longitude_deg"}
    if not needed.issubset(raw.columns):
        raise DataSourceError(f"OurAirports schema changed; missing {needed - set(raw.columns)}")
    df = raw[raw["iso_country"].isin(US_COUNTRIES) & raw["type"].isin(KEEP_TYPES)
             & raw["iata_code"].str.fullmatch(r"[A-Z0-9]{3}")].copy()
    df["state"] = df.apply(lambda r: r["iso_region"].split("-")[-1] if r["iso_country"] == "US"
                           else r["iso_country"], axis=1)
    out = pd.DataFrame({
        "iata": df["iata_code"].str.upper(),
        "faa_lid": df["local_code"].str.upper().replace("", pd.NA),
        "name": df["name"],
        "city": df["municipality"],
        "state": df["state"],
        "lat": pd.to_numeric(df["latitude_deg"], errors="coerce"),
        "lon": pd.to_numeric(df["longitude_deg"], errors="coerce"),
        "airport_type": df["type"],
    })
    # Prefer the most significant facility if an IATA code is duplicated.
    rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2, "seaplane_base": 3}
    out = out.assign(_r=out["airport_type"].map(rank)).sort_values(["iata", "_r"]).drop_duplicates("iata")
    return out.drop(columns="_r").reset_index(drop=True)
