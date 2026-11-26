"""Vectorized daily backtester with an enforced execution lag.

Convention: ``weights.loc[t]`` is the portfolio decided using information up
to the close of day ``t``, and ``returns.loc[t]`` is the close-to-close return
ending on day ``t``.

* ``lag=1``: the portfolio earns ``returns.loc[t+1]``, which means it was
  bought at the close of day ``t``, the same close the signal was computed
  from. That is only achievable if the signal can be computed and traded in
  the closing auction. It is optimistic, especially for short-horizon signals.
* ``lag=2``: bought at the close of day ``t+1``. This is the conservative
  next-day setting, and the research gateway's default.
* ``lag=0`` earns the very return the signal was built from; it is refused
  unless explicitly allowed.

Missing returns: a NaN inside an asset's history (a halt) earns zero. A NaN
after an asset's last valid return means it stopped trading. By default a
position held at that point takes a one-off delisting return (-30%, in line
with the average performance-related delisting return reported by Shumway,
1997) and is closed; treating those returns as zero would hide the losses.
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
    delistings: int = 0  # held positions that hit a delisting

    @property
    def stats(self) -> dict:
        s = summarize(self.returns, self.turnover)
        s.update({"lag": self.lag, "cost_bps": self.cost_bps, "delistings": self.delistings})
        return s


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    lag: int = 1,
    cost_bps: float = 0.0,
    allow_zero_lag: bool = False,
    delist_return: float | None = -0.30,
) -> BacktestResult:
    """``delist_return=None`` treats missing returns after delisting as zero."""
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

    # Delistings: NaN after an asset's last valid return.
    last_valid = r.apply(lambda c: c.last_valid_index())
    pos = pd.Series(range(len(idx)), index=idx)
    after = pd.DataFrame(
        {c: (pos > pos[last_valid[c]]) if last_valid[c] is not None else pd.Series(False, index=idx)
         for c in cols}, index=idx)
    first_after = after & ~after.shift(1, fill_value=False)
    filled = r.fillna(0.0)
    delistings = 0
    if delist_return is not None:
        hit = first_after & (held != 0)
        delistings = int(hit.to_numpy().sum())
        filled = filled.mask(first_after, delist_return)
        # The position is gone after the delisting date.
        held = held.mask(after & ~first_after, 0.0)
    gross = (held * filled).sum(axis=1)
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
        delistings=delistings,
    )
