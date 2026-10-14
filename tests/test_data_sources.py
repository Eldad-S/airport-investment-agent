"""Data-source failure handling and parsing. All network calls are mocked."""
import pandas as pd
import pytest
import requests

from src.data_sources import http
from src.data_sources.faa_enplanements import parse_enplanement_sheet
from src.data_sources.http import DataSourceError, download
from src.data_sources.repository import load_dataset


class _Resp:
    def __init__(self, status, chunks=(b"data",)):
        self.status_code = status
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_download_retries_then_raises(monkeypatch, tmp_path):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(http.requests, "get", boom)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    with pytest.raises(DataSourceError, match="after 3 attempts"):
        download("https://example.invalid/x.zip", tmp_path / "x.zip")
    assert calls["n"] == 3
    assert not (tmp_path / "x.zip").exists()  # no partial / fabricated file left behind


def test_download_404_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: _Resp(404))
    with pytest.raises(DataSourceError, match="404"):
        download("https://example.invalid/missing.zip", tmp_path / "m.zip")


def test_download_uses_cache(monkeypatch, tmp_path):
    p = tmp_path / "cached.zip"
    p.write_bytes(b"already here")
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: pytest.fail("should not hit network"))
    assert download("https://example.invalid/c.zip", p) == p


def test_download_success_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: _Resp(200, (b"ab", b"cd")))
    p = download("https://example.invalid/ok.zip", tmp_path / "ok.zip")
    assert p.read_bytes() == b"abcd"


def test_missing_snapshot_gives_actionable_error(tmp_path):
    with pytest.raises(DataSourceError, match="build_dataset"):
        load_dataset(tmp_path)


def test_faa_sheet_parsing(tmp_path):
    p = tmp_path / "faa.xlsx"
    pd.DataFrame({"Rank": [1, 2, 3], "RO": ["WP"] * 3, "ST": ["CA", "CA", "AK"], "Locid": ["LAX", " sna", "ANC"],
                  "City": ["Los Angeles", "Santa Ana", "Anchorage"], "Airport Name": ["a", "b", "c"],
                  "S/L": ["P"] * 3, "Hub": ["L", "M", None], "CY 25 Enplanements": [100, 50, "n/a"],
                  "CY 24 Enplanements": [90, 50, 10], "% Change": [.1, 0, None]}).to_excel(p, index=False)
    df = parse_enplanement_sheet(p).set_index("locid")
    assert list(df.index) == ["LAX", "SNA", "ANC"]
    assert df.loc["LAX", "enpl_cy2025"] == 100
    assert pd.isna(df.loc["ANC", "enpl_cy2025"])  # invalid value -> NaN, not 0
    assert df.loc["ANC", "hub"] == "N"


def test_faa_sheet_schema_change_is_reported(tmp_path):
    p = tmp_path / "bad.xlsx"
    pd.DataFrame({"foo": [1]}).to_excel(p, index=False)
    with pytest.raises(DataSourceError, match="unexpected columns"):
        parse_enplanement_sheet(p)
