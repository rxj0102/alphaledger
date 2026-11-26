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
    audit_leaks: bool = False,
    scores_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[str, dict | None]:
    """Returns (hypothesis_id, stats or None if the run failed).

    With ``audit_leaks=True`` the leakage checks run first. A trial that fails
    a hard check is recorded with status "leak": it still counts as a trial
    but is excluded from the returns used for evaluation.
    """
    hid = ledger.register(spec, parent_id=parent_id, author=author, rationale=rationale)
    dh = data_fingerprint(returns)
    try:
        weights = build_weights(returns)
        res = run_backtest(weights, returns, lag=lag, cost_bps=cost_bps)
        report = None
        if audit_leaks:
            from ..leakage import audit

            report = audit(build_weights, returns, scores_fn=scores_fn)
    except Exception as exc:  # failures are trials too
        ledger.record(hid, stats={}, data_hash=dh, status="error",
                      error=f"{type(exc).__name__}: {exc}")
        return hid, None
    stats = dict(res.stats)
    status = "ok"
    net = res.returns.dropna()
    if len(net) < 3 or not net.std() > 0:
        # A signal that never takes a position (or a flat book) measured nothing
        # useful, but it was still tried, so it stays in the trial count.
        status = "degenerate"
    if report is not None:
        stats["leak_audit"] = report.to_dict()
        if not report.passed:
            status = "leak"  # a leak outranks a degenerate result
    ledger.record(hid, stats=stats, returns=res.returns, data_hash=dh, status=status)
    return hid, stats
