"""Small HTTP helper: cached downloads with timeouts, retries and informative errors."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from src.config import HTTP_RETRIES, HTTP_TIMEOUT_SECONDS

log = logging.getLogger(__name__)

USER_AGENT = "airport-investment-agent/0.1 (research prototype)"


class DataSourceError(RuntimeError):
    """Raised when an external data source cannot be retrieved. Never silently replaced."""


def download(url: str, dest: Path, *, force: bool = False, retries: int = HTTP_RETRIES) -> Path:
    """Download `url` to `dest` unless it is already cached. Writes atomically via a .part file."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log.info("cache hit: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s (attempt %d/%d)", url, attempt, retries)
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS,
                              headers={"User-Agent": USER_AGENT}) as r:
                if r.status_code == 404:
                    raise DataSourceError(f"Not found (404): {url}")
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except DataSourceError:
            raise
        except (requests.RequestException, OSError) as e:  # network / disk problems: retry
            last_err = e
            log.warning("download failed (%s): %s", url, e)
            time.sleep(2 * attempt)
    raise DataSourceError(f"Failed to download {url} after {retries} attempts: {last_err}")
