"""Loads the processed, committed data snapshot that the app and analytics run on.

The app never calls external APIs at runtime: it reads the snapshot produced by
scripts/build_dataset.py. This keeps the demo fast and reproducible, and means an API outage
cannot silently change the numbers. The snapshot carries its own provenance (manifest.json).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data_sources.http import DataSourceError

REQUIRED_FILES = ("airports_reference.csv", "faa_enplanements.csv",
                  "ontime_airport_month.parquet", "ontime_route_month.parquet", "manifest.json")


@dataclass
class Dataset:
    airports: pd.DataFrame        # reference: iata, name, city, state, lat, lon, airport_type
    enplanements: pd.DataFrame    # FAA CY enplanements keyed by iata
    airport_month: pd.DataFrame   # BTS airport-month aggregates
    route_month: pd.DataFrame     # BTS route-month aggregates
    manifest: dict = field(default_factory=dict)

    @property
    def built_at(self) -> str:
        return self.manifest.get("built_at_utc", "unknown")


def load_dataset(path: Path = PROCESSED_DIR) -> Dataset:
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise DataSourceError(
            f"Processed data snapshot is incomplete (missing: {', '.join(missing)}). "
            "Run `python scripts/build_dataset.py` to download and build it.")
    airports = pd.read_csv(path / "airports_reference.csv", keep_default_na=False,
                           na_values={"faa_lid": [""], "lat": [""], "lon": [""]})
    enpl = pd.read_csv(path / "faa_enplanements.csv")
    enpl = enpl.dropna(subset=["iata"]).drop_duplicates("iata")
    return Dataset(
        airports=airports,
        enplanements=enpl,
        airport_month=pd.read_parquet(path / "ontime_airport_month.parquet"),
        route_month=pd.read_parquet(path / "ontime_route_month.parquet"),
        manifest=json.loads((path / "manifest.json").read_text()),
    )
