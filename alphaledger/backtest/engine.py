"""Vectorized daily backtester with an enforced execution lag.

Convention: ``weights.loc[t]`` is the portfolio decided using information up
to the close of day ``t``. With ``lag=1`` it is held over day ``t+1`` and earns
``returns.loc[t+1]``. A lag of zero means trading on the same close the signal
was computed from, which is look-ahead for almost every daily signal, so it
is refused unless explicitly allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.pit import LookAheadError
from .metrics import summarize


@dataclass
class BacktestResult:
    returns: pd.Series  # net daily portfolio returns
    gross_returns: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame  # weights actually held each day
    lag: int
    cost_bps: float

    @property
    def stats(self) -> dict:
        s = summarize(self.returns, self.turnover)
        s.update({"lag": self.lag, "cost_bps": self.cost_bps})
        return s


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    lag: int = 1,
    cost_bps: float = 0.0,
    allow_zero_lag: bool = False,
) -> BacktestResult:
    if not isinstance(lag, int) or lag < 0:
        raise ValueError("lag must be a non-negative integer")
    if lag == 0 and not allow_zero_lag:
        raise LookAheadError(
            "lag=0 trades on the same close used to build the signal; "
            "pass allow_zero_lag=True only if the signal is truly known before the open"
        )
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")

    cols = weights.columns.intersection(returns.columns)
    if len(cols) == 0:
        raise ValueError("weights and returns share no columns")
    idx = returns.index.sort_values()
    # Rows missing from ``weights`` (e.g. monthly rebalancing) hold the last
    # decided portfolio; NaN cells inside a decided row mean no position.
    w = (
        weights.sort_index()
        .reindex(columns=cols)
        .fillna(0.0)
        .reindex(idx, method="ffill")
        .fillna(0.0)
    )
    r = returns.reindex(index=idx, columns=cols)

    held = w.shift(lag).fillna(0.0)
    gross = (held * r.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 1e4

    # Drop the warm-up period before the first live position.
    live = held.abs().sum(axis=1) > 0
    if live.any():
        start = live.idxmax()
        net, gross, turnover, held = net.loc[start:], gross.loc[start:], turnover.loc[start:], held.loc[start:]

    return BacktestResult(
        returns=net.rename("net"),
        gross_returns=gross.rename("gross"),
        turnover=turnover.rename("turnover"),
        positions=held,
        lag=lag,
        cost_bps=float(cost_bps),
    )
