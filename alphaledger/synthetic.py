"""Synthetic markets with known properties, for testing the pipeline.

``make_market`` gives each asset a slowly drifting expected return (a highly
persistent AR(1)). Past returns then carry information about future returns,
so a correct momentum backtest must find a positive edge, and a broken one
(wrong lag, wrong sign, leaked data) shows up as a deviation from that.
With ``persistence=0`` or ``drift_vol=0`` returns are pure noise and any edge
found is spurious.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_market(
    n_assets: int = 50,
    n_days: int = 252 * 8,
    daily_vol: float = 0.02,
    drift_vol: float = 0.0004,
    persistence: float = 0.998,
    start: str = "2010-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    mu = np.zeros((n_days, n_assets))
    innov_sd = drift_vol * np.sqrt(max(1 - persistence**2, 0.0))
    mu[0] = rng.normal(0, drift_vol, n_assets)
    for t in range(1, n_days):
        mu[t] = persistence * mu[t - 1] + rng.normal(0, innov_sd, n_assets)
    eps = rng.normal(0, daily_vol, (n_days, n_assets))
    rets = mu + eps
    cols = [f"A{i:03d}" for i in range(n_assets)]
    return pd.DataFrame(rets, index=dates, columns=cols)
