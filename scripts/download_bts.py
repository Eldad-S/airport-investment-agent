"""Download BTS on-time monthly files for the analysis years into data/raw (cached).

Usage:  python scripts/download_bts.py [--workers 4]
Downloads are cached; re-running only fetches missing months.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CURRENT_YEAR, PRIOR_YEAR  # noqa: E402
from src.data_sources.bts_ontime import fetch_month  # noqa: E402
from src.data_sources.http import DataSourceError  # noqa: E402


def analysis_months() -> list[tuple[int, int]]:
    return [(y, m) for y in (PRIOR_YEAR, CURRENT_YEAR) for m in range(1, 13)]


def download_all(workers: int = 4) -> list[str]:
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_month, y, m): (y, m) for y, m in analysis_months()}
        for fut in as_completed(futs):
            y, m = futs[fut]
            try:
                p = fut.result()
                logging.info("ok %d-%02d -> %s", y, m, p.name)
            except DataSourceError as e:
                logging.error("FAILED %d-%02d: %s", y, m, e)
                errors.append(f"{y}-{m:02d}: {e}")
    return errors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    errs = download_all(ap.parse_args().workers)
    if errs:
        print("Some months failed:\n" + "\n".join(errs))
        sys.exit(1)
