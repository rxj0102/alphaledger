"""Loaders for free data sources, converted to point-in-time rows.

Each source has a pure parser (tested offline against fixture text) and a thin
fetch function that does the network call. Availability rules:

* Daily prices and factor returns: available at the close of the same day.
* FRED/ALFRED macro series: available at each vintage's ``realtime_start``,
  so revisions never leak backwards.

Known limitation: yfinance only lists tickers that trade today, so any
universe built from it has survivorship bias. Results that use it should say so.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

KEN_FRENCH_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


# ----------------------------------------------------------------- Ken French
def parse_ken_french_daily(text: str, source: str = "ken_french") -> pd.DataFrame:
    """Parse the Fama/French daily factors CSV (values in percent) to PIT rows."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not (parts[0].isdigit() and len(parts[0]) == 8):
            continue
        rows.append(parts)
    header = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 1 and parts[0] == "" and "Mkt-RF" in parts:
            header = parts[1:]
            break
    if header is None:
        raise ValueError("could not find Ken French header row")
    df = pd.DataFrame(rows).iloc[:, : len(header) + 1]
    df.columns = ["date"] + header
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    long = df.melt(id_vars="date", var_name="field", value_name="value")
    long["value"] = long["value"].astype(float) / 100.0
    long["entity"] = "FF"
    long["period"] = long["date"]
    long["available_at"] = long["date"]
    long["source"] = source
    return long[["entity", "field", "period", "available_at", "value", "source"]]


def fetch_ken_french_daily() -> pd.DataFrame:  # pragma: no cover - network
    import urllib.request

    with urllib.request.urlopen(KEN_FRENCH_DAILY_URL, timeout=60) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("latin-1")
    return parse_ken_french_daily(text)


# ----------------------------------------------------------------- FRED/ALFRED
def parse_fred_vintages(payload: str | dict, series_id: str) -> pd.DataFrame:
    """Parse a FRED observations JSON response requested with the full real-time
    period (realtime_start=1776-07-04, realtime_end=9999-12-31) into PIT rows.

    Each observation row is one vintage of one period; ``realtime_start`` is
    when that value first became known.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    obs = pd.DataFrame(data.get("observations", []))
    if obs.empty:
        return pd.DataFrame(
            columns=["entity", "field", "period", "available_at", "value", "source"]
        )
    obs = obs[obs["value"] != "."].copy()
    out = pd.DataFrame(
        {
            "entity": "US",
            "field": series_id,
            "period": pd.to_datetime(obs["date"]),
            "available_at": pd.to_datetime(obs["realtime_start"]),
            "value": obs["value"].astype(float),
            "source": "fred",
        }
    )
    return out.reset_index(drop=True)


def fetch_fred_vintages(series_id: str, api_key: str) -> pd.DataFrame:  # pragma: no cover
    """Fetch all vintages of a FRED series. Requires a free FRED API key."""
    import urllib.parse
    import urllib.request

    q = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "api_key": api_key,
            "file_type": "json",
        }
    )
    with urllib.request.urlopen(f"{FRED_OBS_URL}?{q}", timeout=60) as r:
        return parse_fred_vintages(r.read().decode(), series_id)


# ----------------------------------------------------------------- prices
def prices_to_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a dates x tickers close-price panel."""
    close = close.sort_index()
    rets = close.pct_change(fill_method=None)
    return rets.replace([np.inf, -np.inf], np.nan)


def prices_to_pit(close: pd.DataFrame, field: str = "close", source: str = "yfinance") -> pd.DataFrame:
    """Convert a dates x tickers panel to PIT rows available at the same close."""
    long = close.rename_axis("period").reset_index().melt(
        id_vars="period", var_name="entity", value_name="value"
    ).dropna(subset=["value"])
    long["field"] = field
    long["available_at"] = long["period"]
    long["source"] = source
    return long[["entity", "field", "period", "available_at", "value", "source"]]


def fetch_yfinance_close(tickers, start, end=None) -> pd.DataFrame:  # pragma: no cover
    """Adjusted daily closes from yfinance (survivorship-biased universe)."""
    import yfinance as yf

    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if not isinstance(df.columns, pd.MultiIndex):
        close.columns = list(tickers)
    return close.dropna(how="all")
