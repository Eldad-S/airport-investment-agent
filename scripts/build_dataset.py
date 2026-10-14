"""End-to-end data acquisition: download public sources and build the processed snapshot.

    python scripts/build_dataset.py            # uses cached raw files where present
    python scripts/build_dataset.py --force    # re-download everything

Outputs (committed, small) in data/processed/:
    airports_reference.csv     OurAirports names / states / coordinates
    faa_enplanements.csv       FAA CY passenger boardings (current, prior, 2019 baseline)
    ontime_airport_month.parquet   BTS airport-month operations & delay aggregates
    ontime_route_month.parquet     BTS route-month flight counts & distances
    manifest.json              provenance: URLs, retrieval times, row counts, QA stats

If a source fails the script stops with an error - it never fabricates replacement data.
The app keeps working from the previously committed snapshot.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from scripts.download_bts import analysis_months, download_all  # noqa: E402
from src.config import BASELINE_YEAR, CURRENT_YEAR, PROCESSED_DIR  # noqa: E402
from src.data_sources import airport_reference, bts_ontime, faa_enplanements  # noqa: E402

log = logging.getLogger("build_dataset")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _process(ym: tuple[int, int]):
    return bts_ontime.process_month(*ym)


def crosswalk_faa_to_iata(enpl: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """FAA LIDs usually equal IATA codes; when they don't, map through OurAirports local_code."""
    iata_set = set(ref["iata"])
    lid_map = ref.dropna(subset=["faa_lid"]).set_index("faa_lid")["iata"].to_dict()

    def to_iata(lid: str) -> str | None:
        if lid in lid_map and lid_map[lid] != lid and lid not in iata_set:
            return lid_map[lid]
        if lid in iata_set:
            return lid
        return lid_map.get(lid)

    enpl = enpl.copy()
    enpl["iata"] = enpl["locid"].map(to_iata)
    stats = {
        "faa_rows": int(len(enpl)),
        "mapped_identical": int((enpl["iata"] == enpl["locid"]).sum()),
        "mapped_via_crosswalk": int(((enpl["iata"] != enpl["locid"]) & enpl["iata"].notna()).sum()),
        "unmapped": int(enpl["iata"].isna().sum()),
    }
    return enpl, stats


def main(force: bool = False, workers: int = 4) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"built_at_utc": _now(), "sources": {}}

    # 1. Airport reference
    ref_path = airport_reference.fetch(force=force)
    ref = airport_reference.build_reference(ref_path)
    manifest["sources"]["ourairports"] = {"url": airport_reference.URL, "retrieved_at_utc": _now(),
                                          "rows": int(len(ref))}

    # 2. FAA enplanements
    cur_p = faa_enplanements.fetch(CURRENT_YEAR, force=force)
    base_p = faa_enplanements.fetch(BASELINE_YEAR, force=force)
    enpl = faa_enplanements.build_enplanements(cur_p, base_p)
    enpl, xw = crosswalk_faa_to_iata(enpl, ref)
    manifest["sources"]["faa_enplanements"] = {
        "urls": list(faa_enplanements.URLS.values()), "retrieved_at_utc": _now(),
        "rows": int(len(enpl)), "identifier_crosswalk": xw,
    }

    # 3. BTS on-time performance
    errors = download_all(workers=workers)
    if errors:
        raise SystemExit("BTS download failed; snapshot NOT rebuilt:\n" + "\n".join(errors))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_process, analysis_months()))
    airport_month = pd.concat([r[0] for r in results], ignore_index=True)
    route_month = pd.concat([r[1] for r in results], ignore_index=True)
    month_stats = [r[2] for r in results]
    manifest["sources"]["bts_ontime"] = {
        "url_template": bts_ontime.URL_TEMPLATE, "retrieved_at_utc": _now(),
        "months": month_stats,
        "rows_raw_total": sum(s["rows_raw"] for s in month_stats),
        "rows_clean_total": sum(s["rows_clean"] for s in month_stats),
        "duplicates_removed_total": sum(s["rows_duplicate_removed"] for s in month_stats),
    }

    # Keep only reference rows that are actually used, to keep the snapshot small.
    used = set(airport_month["airport"]) | set(route_month["dest"]) | set(enpl["iata"].dropna())
    ref = ref[ref["iata"].isin(used)]

    ref.to_csv(PROCESSED_DIR / "airports_reference.csv", index=False)
    enpl.to_csv(PROCESSED_DIR / "faa_enplanements.csv", index=False)
    airport_month.to_parquet(PROCESSED_DIR / "ontime_airport_month.parquet", index=False)
    route_month.to_parquet(PROCESSED_DIR / "ontime_route_month.parquet", index=False)
    (PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("snapshot written to %s", PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download all sources")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(force=args.force, workers=args.workers)
