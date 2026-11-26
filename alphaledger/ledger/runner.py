"""Register, test and record a hypothesis in one step.

This is the only path the Week 5 agent gateway will expose: a hypothesis is
logged before it runs, and its outcome is logged whether it succeeds,
fails or raises.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from ..backtest import run_backtest
from .store import Ledger, data_fingerprint


def run_logged(
    ledger: Ledger,
    spec: dict,
    build_weights: Callable[[pd.DataFrame], pd.DataFrame],
    returns: pd.DataFrame,
    parent_id: str | None = None,
    author: str = "agent",
    rationale: str = "",
    lag: int = 1,
    cost_bps: float = 0.0,
) -> tuple[str, dict | None]:
    """Returns (hypothesis_id, stats or None if the run failed)."""
    hid = ledger.register(spec, parent_id=parent_id, author=author, rationale=rationale)
    dh = data_fingerprint(returns)
    try:
        weights = build_weights(returns)
        res = run_backtest(weights, returns, lag=lag, cost_bps=cost_bps)
    except Exception as exc:  # failures are trials too
        ledger.record(hid, stats={}, data_hash=dh, status="error",
                      error=f"{type(exc).__name__}: {exc}")
        return hid, None
    stats = res.stats
    ledger.record(hid, stats=stats, returns=res.returns, data_hash=dh)
    return hid, stats
