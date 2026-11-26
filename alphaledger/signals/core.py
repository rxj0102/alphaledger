"""Signal construction and portfolio weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Cross-sectional momentum: compounded return from t-lookback to t-skip.

    The value at row t uses returns through day t-skip only, so it is known at
    the close of day t. The classic 12-1 month version is lookback=252, skip=21.
    """
    if skip >= lookback:
        raise ValueError("skip must be smaller than lookback")
    log_r = np.log1p(returns)
    window = log_r.rolling(lookback - skip, min_periods=lookback - skip).sum()
    return np.expm1(window.shift(skip))


def quantile_long_short(signal: pd.DataFrame, q: float = 0.2, min_assets: int = 5) -> pd.DataFrame:
    """Dollar-neutral weights: long the top q, short the bottom q, equal weight.

    Each leg sums to 0.5 in absolute value, so gross exposure is 1.0.
    Rows with fewer than ``min_assets`` valid signals get zero weight.
    """
    if not 0 < q <= 0.5:
        raise ValueError("q must be in (0, 0.5]")
    ranks = signal.rank(axis=1, pct=True)
    n_valid = signal.notna().sum(axis=1)
    long = (ranks > 1 - q).astype(float)
    short = (ranks <= q).astype(float)
    long = long.div(long.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    short = short.div(short.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    w = (long - short).fillna(0.0)
    w.loc[n_valid < min_assets] = 0.0
    return w


def rebalance(weights: pd.DataFrame, period: str = "M") -> pd.DataFrame:
    """Keep only the last decided row of each period ("M" month, "W" week, "Q" quarter).

    The backtester holds each kept row until the next one.
    """
    return weights.groupby(weights.index.to_period(period)).tail(1)
