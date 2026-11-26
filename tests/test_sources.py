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
    assert (df["available_at"] == df["period"]).all()


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
