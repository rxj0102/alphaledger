"""Loaders for free data sources, converted to point-in-time rows.

Each source has a pure parser (tested offline against fixture text) and a thin
fetch function that does the network call. Availability rules:

* Daily prices: available at the close of the same day.
* Ken French factor returns: the library is updated monthly with a lag, so a
  day's factor return is treated as available at the end of the following
  month. Checked once: the file downloaded on 26 Nov 2026 was built from the
  202609 CRSP database and dated 26 Oct 2026, so September data appeared
  before the end of October. Use this dating when factors feed a signal.
* Ken French industry portfolios as an asset universe: their daily returns
  are realized each day from prices known at the close, so they can be used
  as same-day returns for backtests. Portfolios are rebuilt every June from
  the stocks alive then, so the universe has no survivorship bias.
* FRED and ALFRED: either the official API with a free key (exact release
  dates) or keyless monthly ALFRED snapshots (each value enters at the first
  snapshot that contains it, at most a month late, never early).
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
    long["available_at"] = long["date"] + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)
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


# ----------------------------------------------------------------- industries
KEN_FRENCH_INDUSTRY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "{n}_Industry_Portfolios_daily_CSV.zip"
)


def parse_ken_french_industries(text: str, weighting: str = "value") -> pd.DataFrame:
    """Daily industry portfolio returns (dates x industries, decimal returns).

    ``weighting`` picks the value-weighted (default) or equal-weighted block.
    Missing values (-99.99 or -999) become NaN.
    """
    marker = {"value": "Average Value Weighted Returns", "equal": "Average Equal Weighted Returns"}[weighting]
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if marker in ln)
    except StopIteration:
        raise ValueError(f"block {marker!r} not found") from None
    header = [h.strip() for h in lines[start + 1].split(",")][1:]
    rows = []
    for ln in lines[start + 2:]:
        parts = [p.strip() for p in ln.split(",")]
        if not (parts[0].isdigit() and len(parts[0]) == 8):
            break  # end of this block
        rows.append(parts[: len(header) + 1])
    df = pd.DataFrame(rows, columns=["date"] + header)
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m%d")
    df.index.name = None
    df = df.astype(float)
    df = df.mask(df <= -99.99)
    return df / 100.0


def fetch_ken_french_industries(n: int = 49, weighting: str = "value") -> pd.DataFrame:  # pragma: no cover
    import urllib.request

    with urllib.request.urlopen(KEN_FRENCH_INDUSTRY_URL.format(n=n), timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_ken_french_industries(text, weighting)


# ----------------------------------------------------------------- FRED/ALFRED
ALFRED_CSV_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series}&vintage_date={date}"


def parse_alfred_snapshot(text: str, series: str, vintage_date) -> pd.DataFrame:
    """One keyless ALFRED snapshot: every period's value as known on ``vintage_date``."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("observation_date"):
        raise ValueError("unexpected ALFRED response")
    rows = []
    for ln in lines[1:]:
        d, v = ln.split(",")[:2]
        if v.strip() in ("", "."):
            continue
        rows.append((d, float(v)))
    out = pd.DataFrame(rows, columns=["period", "value"])
    out["period"] = pd.to_datetime(out["period"])
    out["available_at"] = pd.Timestamp(vintage_date)
    out["entity"] = "US"
    out["field"] = series
    out["source"] = "alfred_snapshot"
    return out[["entity", "field", "period", "available_at", "value", "source"]]


def snapshots_to_pit(snapshots: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge snapshots, keeping a row only when a period first appears or its value changes.

    A value therefore enters at the first snapshot that contains it, and a
    revision at the first snapshot that shows it.
    """
    if not snapshots:
        return pd.DataFrame(columns=["entity", "field", "period", "available_at", "value", "source"])
    allr = pd.concat(snapshots, ignore_index=True).sort_values(["entity", "field", "period", "available_at"])
    prev = allr.groupby(["entity", "field", "period"])["value"].shift(1)
    keep = prev.isna() | (allr["value"] != prev)
    return allr[keep].reset_index(drop=True)


def fetch_alfred_snapshots(series: str, start: str, end: str | None = None,
                           freq: str = "MS") -> pd.DataFrame:  # pragma: no cover - network
    """Point-in-time rows for a FRED series without an API key.

    Requests one ALFRED vintage per ``freq`` step (month starts by default).
    Values become available at the snapshot date, which is on or after the
    true release date, so the result can be late by up to one step but never
    early. Use ``fetch_fred_vintages`` with an API key for exact dates.
    """
    import urllib.request

    dates = pd.date_range(start, end or pd.Timestamp.today().normalize(), freq=freq)
    snaps = []
    for d in dates:
        url = ALFRED_CSV_URL.format(series=series, date=d.strftime("%Y-%m-%d"))
        with urllib.request.urlopen(url, timeout=60) as r:
            snaps.append(parse_alfred_snapshot(r.read().decode(), series, d))
    return snapshots_to_pit(snaps)


# ----------------------------------------------------------------- FRED API
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
