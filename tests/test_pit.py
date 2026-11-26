import pandas as pd
import pytest

from alphaledger.data import LookAheadError, PITStore


def gdp_store():
    # Q1 GDP first printed at 100 on Apr 28, revised to 102 on May 30.
    # Q2 first printed on Jul 28.
    s = PITStore()
    s.add(
        pd.DataFrame(
            {
                "entity": ["US"] * 3,
                "field": ["GDP"] * 3,
                "period": ["2024-03-31", "2024-03-31", "2024-06-30"],
                "available_at": ["2024-04-28", "2024-05-30", "2024-07-28"],
                "value": [100.0, 102.0, 105.0],
            }
        )
    )
    return s


def test_nothing_visible_before_release():
    s = gdp_store()
    assert s.as_of("2024-04-27", "GDP").empty


def test_first_print_then_revision():
    s = gdp_store()
    assert s.latest("2024-05-01", "GDP")["US"] == 100.0
    assert s.latest("2024-06-01", "GDP")["US"] == 102.0


def test_revision_does_not_leak_backwards():
    s = gdp_store()
    q1_seen_in_may = s.as_of("2024-05-15", "GDP")
    assert q1_seen_in_may["value"].tolist() == [100.0]


def test_latest_picks_most_recent_period():
    s = gdp_store()
    assert s.latest("2024-08-01", "GDP")["US"] == 105.0
    hist = s.as_of("2024-08-01", "GDP")
    assert hist["value"].tolist() == [102.0, 105.0]


def test_panel_respects_availability_and_lag():
    s = gdp_store()
    dates = pd.to_datetime(["2024-04-27", "2024-04-28", "2024-05-30", "2024-05-31", "2024-07-29"])
    p = s.panel("GDP", dates)
    assert pd.isna(p.loc["2024-04-27", "US"])
    assert p.loc["2024-04-28", "US"] == 100.0
    assert p.loc["2024-05-30", "US"] == 102.0
    assert p.loc["2024-07-29", "US"] == 105.0
    lagged = s.panel("GDP", dates, lag_days=1)
    assert pd.isna(lagged.loc["2024-04-28", "US"])
    assert lagged.loc["2024-05-30", "US"] == 100.0
    assert lagged.loc["2024-05-31", "US"] == 102.0


def test_assert_no_lookahead():
    s = gdp_store()
    used = s.as_of("2024-08-01", "GDP")
    s.assert_no_lookahead("2024-08-01", used)
    with pytest.raises(LookAheadError):
        s.assert_no_lookahead("2024-06-01", used)


def test_available_before_period_rejected_unless_allowed():
    bad = pd.DataFrame(
        {"entity": ["X"], "field": ["f"], "period": ["2024-06-30"],
         "available_at": ["2024-01-01"], "value": [1.0]}
    )
    with pytest.raises(ValueError):
        PITStore().add(bad)
    s = PITStore(allow_early_availability=True)
    s.add(bad)
    assert len(s) == 1


def test_duplicate_vintage_keeps_last():
    s = gdp_store()
    s.add(pd.DataFrame({"entity": ["US"], "field": ["GDP"], "period": ["2024-03-31"],
                        "available_at": ["2024-04-28"], "value": [99.0]}))
    assert len(s) == 3
    assert s.latest("2024-05-01", "GDP")["US"] == 99.0
