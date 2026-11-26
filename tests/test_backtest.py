import numpy as np
import pandas as pd
import pytest

from alphaledger.backtest import run_backtest
from alphaledger.backtest.metrics import max_drawdown, sharpe
from alphaledger.data import LookAheadError
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.synthetic import make_market


def toy():
    idx = pd.bdate_range("2024-01-01", periods=4)
    r = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.03], "B": [0.0, -0.01, 0.02, 0.01]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]}, index=idx)
    return w, r


def test_lag_one_alignment_by_hand():
    w, r = toy()
    res = run_backtest(w, r, lag=1)
    # Day 2 holds day-1 weights (A), day 3 holds A, day 4 holds B.
    assert res.returns.tolist() == pytest.approx([0.02, -0.01, 0.01])


def test_zero_lag_refused_by_default():
    w, r = toy()
    with pytest.raises(LookAheadError):
        run_backtest(w, r, lag=0)


def test_costs_by_hand():
    w, r = toy()
    res = run_backtest(w, r, lag=1, cost_bps=10)
    # Turnover: 1 (enter A), 0, 2 (A to B).
    assert res.turnover.tolist() == pytest.approx([1.0, 0.0, 2.0])
    assert (res.gross_returns - res.returns).tolist() == pytest.approx([0.001, 0.0, 0.002])


def test_same_day_signal_is_fake_alpha_only_without_lag():
    # Signal = today's return. Known at today's close, so with lag=1 on pure
    # noise it should earn nothing; with lag=0 it earns its own return.
    r = make_market(n_assets=100, drift_vol=0.0, seed=7)
    w = quantile_long_short(r)
    leaked = run_backtest(w, r, lag=0, allow_zero_lag=True).stats["sharpe"]
    honest = run_backtest(w, r, lag=1).stats["sharpe"]
    assert leaked > 20
    assert abs(honest) < 1.5


def test_monthly_rebalance_holds_positions():
    r = make_market(n_assets=30, n_days=300, seed=1)
    w = rebalance(quantile_long_short(momentum(r, lookback=60, skip=5)))
    res = run_backtest(w, r)
    changes = (res.positions.diff().abs().sum(axis=1) > 0).sum()
    months = res.positions.index.to_period("M").nunique()
    assert changes <= months


def test_momentum_recovers_planted_edge():
    sharpes = []
    for seed in range(4):
        r = make_market(n_assets=200, drift_vol=0.0008, seed=seed)
        w = rebalance(quantile_long_short(momentum(r)))
        sharpes.append(run_backtest(w, r).stats["sharpe"])
    assert min(sharpes) > 1.5


def test_momentum_finds_nothing_in_noise_and_sign_flip_loses():
    noise = []
    for seed in range(6):
        r = make_market(n_assets=200, drift_vol=0.0, seed=seed)
        noise.append(run_backtest(rebalance(quantile_long_short(momentum(r))), r).stats["sharpe"])
    assert abs(np.mean(noise)) < 0.6
    r = make_market(n_assets=200, drift_vol=0.0008, seed=0)
    flipped = run_backtest(rebalance(quantile_long_short(-momentum(r))), r).stats["sharpe"]
    assert flipped < -1.5


def test_momentum_uses_no_future_returns():
    r = make_market(n_assets=10, n_days=400, seed=3)
    base = momentum(r, lookback=60, skip=5)
    t = r.index[300]
    shocked = r.copy()
    shocked.loc[shocked.index >= t] *= -5  # change today and every later day
    after = momentum(shocked, lookback=60, skip=5)
    # Signal at t uses returns through t-5 only, so it must be unchanged.
    pd.testing.assert_series_equal(base.loc[t], after.loc[t])


def test_metrics_by_hand():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert np.isnan(sharpe(pd.Series([0.01, 0.01, 0.01])))
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected)


def test_delisting_return_applied_once_and_position_closed():
    idx = pd.bdate_range("2024-01-01", periods=6)
    r = pd.DataFrame({"A": [0.01, 0.01, 0.01, None, None, None], "B": [0.0] * 6}, index=idx)
    w = pd.DataFrame({"A": [1.0] * 6, "B": [0.0] * 6}, index=idx)
    res = run_backtest(w, r, lag=1)
    # Held A from day 2; A stops trading after day 3, so day 4 takes -30%, then nothing.
    assert res.returns.tolist() == pytest.approx([0.01, 0.01, -0.30, 0.0, 0.0])
    assert res.delistings == 1
    zero = run_backtest(w, r, lag=1, delist_return=None)
    assert zero.returns.tolist() == pytest.approx([0.01, 0.01, 0.0, 0.0, 0.0])


def test_interior_gap_is_not_a_delisting():
    idx = pd.bdate_range("2024-01-01", periods=5)
    r = pd.DataFrame({"A": [0.01, None, 0.02, 0.01, 0.01]}, index=idx)
    w = pd.DataFrame({"A": [1.0] * 5}, index=idx)
    res = run_backtest(w, r, lag=1)
    assert res.delistings == 0
    assert res.returns.tolist() == pytest.approx([0.0, 0.02, 0.01, 0.01])
