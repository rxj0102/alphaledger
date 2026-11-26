import os

import pytest
import json

import pandas as pd

from alphaledger.data import PITStore
from alphaledger.data.sources import (
    parse_fred_vintages,
    parse_ken_french_daily,
    prices_to_pit,
    prices_to_returns,
)

KF_FIXTURE = """This file was created by CMPT_ME_BEME_RETS_DAILY using the 202409 CRSP database.
The Tbill return is the simple daily rate that, over the number of trading days
in the month, compounds to 1-month TBill rate from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RF
19260701,    0.10,   -0.25,   -0.27,    0.01
19260702,    0.45,   -0.33,   -0.06,    0.01
19260706,    0.17,    0.30,   -0.39,    0.01

Copyright 2024 Eugene F. Fama and Kenneth R. French
"""

FRED_FIXTURE = {
    "observations": [
        {"realtime_start": "2024-04-25", "realtime_end": "2024-05-29", "date": "2024-01-01", "value": "28269.2"},
        {"realtime_start": "2024-05-30", "realtime_end": "9999-12-31", "date": "2024-01-01", "value": "28255.9"},
        {"realtime_start": "2024-07-25", "realtime_end": "9999-12-31", "date": "2024-04-01", "value": "."},
    ]
}


def test_ken_french_parse():
    df = parse_ken_french_daily(KF_FIXTURE)
    assert set(df["field"]) == {"Mkt-RF", "SMB", "HML", "RF"}
    assert len(df) == 12
    row = df[(df["field"] == "Mkt-RF") & (df["period"] == "1926-07-02")]
    assert abs(row["value"].iloc[0] - 0.0045) < 1e-12
    # Published with a lag: a July 1926 return is only treated as known at end of August.
    assert (df["available_at"] == pd.Timestamp("1926-08-31")).all()


def test_fred_vintages_into_pit_store():
    rows = parse_fred_vintages(json.dumps(FRED_FIXTURE), "GDP")
    assert len(rows) == 2  # missing "." value dropped
    s = PITStore()
    s.add(rows)
    assert s.latest("2024-05-01", "GDP")["US"] == 28269.2
    assert s.latest("2024-06-01", "GDP")["US"] == 28255.9


def test_prices_helpers():
    close = pd.DataFrame(
        {"AAA": [10.0, 11.0, None, 12.1], "BBB": [20.0, 19.0, 19.0, 20.9]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    r = prices_to_returns(close)
    assert abs(r["AAA"].iloc[1] - 0.1) < 1e-12
    assert pd.isna(r["AAA"].iloc[2]) and pd.isna(r["AAA"].iloc[3])
    pit = prices_to_pit(close)
    assert len(pit) == 7
    assert (pit["available_at"] == pit["period"]).all()


IND_FIXTURE = """This file was created using the 202609 CRSP database.
Missing data are indicated by -99.99 or -999.

  Average Value Weighted Returns -- Daily
,Agric,Food ,Soda
19260701,   0.56,  -0.07, -99.99
19260702,   0.29,   0.06,   1.00

  Average Equal Weighted Returns -- Daily
,Agric,Food ,Soda
19260701,   9.00,   9.00,   9.00
"""


def test_industry_parse_value_block_and_missing():
    from alphaledger.data.sources import parse_ken_french_industries

    df = parse_ken_french_industries(IND_FIXTURE)
    assert list(df.columns) == ["Agric", "Food", "Soda"]
    assert len(df) == 2  # stops at the end of the value-weighted block
    assert df.loc["1926-07-01", "Agric"] == pytest.approx(0.0056)
    assert pd.isna(df.loc["1926-07-01", "Soda"])
    assert parse_ken_french_industries(IND_FIXTURE, "equal").iloc[0, 0] == pytest.approx(0.09)


def test_alfred_snapshots_to_pit():
    from alphaledger.data.sources import parse_alfred_snapshot, snapshots_to_pit

    a = parse_alfred_snapshot("observation_date,INDPRO_20200101\n2019-10-01,100.0\n2019-11-01,101.0\n",
                              "INDPRO", "2020-01-01")
    b = parse_alfred_snapshot("observation_date,INDPRO_20200201\n2019-10-01,100.0\n2019-11-01,100.5\n"
                              "2019-12-01,102.0\n", "INDPRO", "2020-02-01")
    pit = snapshots_to_pit([a, b])
    assert len(pit) == 4  # Oct once, Nov first print and revision, Dec first print
    s = PITStore()
    s.add(pit)
    assert s.latest("2020-01-15", "INDPRO")["US"] == 101.0  # Nov as first printed
    hist = s.as_of("2020-02-15", "INDPRO").set_index("period")["value"]
    assert hist[pd.Timestamp("2019-11-01")] == 100.5  # revision seen from February
    assert hist[pd.Timestamp("2019-12-01")] == 102.0


# ---------------------------------------------------------------- live network checks
# Off by default so CI does not depend on outside services:
#   ALPHALEDGER_NETWORK_TESTS=1 pytest tests/test_sources.py

network = pytest.mark.skipif(not os.environ.get("ALPHALEDGER_NETWORK_TESTS"),
                             reason="set ALPHALEDGER_NETWORK_TESTS=1 to run live data checks")


@network
def test_live_ken_french_factors():
    from alphaledger.data.sources import fetch_ken_french_daily

    df = fetch_ken_french_daily()
    assert {"Mkt-RF", "SMB", "HML", "RF"} <= set(df["field"])
    assert df["period"].min() == pd.Timestamp("1926-07-01")
    assert (df["available_at"] > df["period"]).all()


@network
def test_live_ken_french_industries():
    from alphaledger.data.sources import fetch_ken_french_industries

    df = fetch_ken_french_industries(49)
    assert df.shape[1] == 49 and len(df) > 25000
    assert df.min().min() > -1.0  # decimal returns; no -99.99 or -999 codes left
    assert df.loc["1990":].notna().all().all()  # complete since 1990


@network
def test_live_yfinance():
    from alphaledger.data.sources import fetch_yfinance_close

    close = fetch_yfinance_close(["AAPL", "MSFT"], "2024-01-02", "2024-02-01")
    assert list(close.columns) == ["AAPL", "MSFT"] and len(close) > 15
    assert prices_to_returns(close).abs().max().max() < 0.2


@network
def test_live_alfred_snapshots():
    from alphaledger.data.sources import fetch_alfred_snapshots

    pit = fetch_alfred_snapshots("INDPRO", "2020-01-01", "2020-04-01")
    s = PITStore()
    s.add(pit)
    jan = s.as_of("2020-01-15", "INDPRO")
    assert jan["period"].max() < pd.Timestamp("2020-01-01")  # December not yet released in early January
    assert (pit["available_at"] >= pit["period"]).all()
