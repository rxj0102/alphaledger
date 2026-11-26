"""Performance metrics for daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods))


def ann_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = pd.Series(r).dropna()
    return float(r.mean() * periods) if len(r) else float("nan")


def ann_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = pd.Series(r).dropna()
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough loss of the compounded equity curve (negative number)."""
    r = pd.Series(r).fillna(0.0)
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity.to_numpy()]))[1:]
    return float((equity.to_numpy() / peak - 1.0).min())


def t_stat(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else float("nan")


def summarize(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = pd.Series(r).dropna()
    out = {
        "n_obs": int(len(r)),
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "t_stat": t_stat(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
    }
    if turnover is not None:
        out["avg_daily_turnover"] = float(pd.Series(turnover).mean())
    return out
