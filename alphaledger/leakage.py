"""Leakage detectors.

A signal or weight function ``f(returns) -> DataFrame`` is causal if its row
for date t depends only on data up to and including t. The hard checks test
that directly; the soft checks flag patterns that usually mean trouble.

Hard checks (severity "fail"):

* truncation: ``f(returns[:t]).loc[t]`` must equal ``f(returns).loc[t]``.
  Catches shift(-k), centered windows, full-sample normalization, backward
  fills, fitting on the whole sample, and use of the last row.
* future perturbation: scrambling every return after t must not change any
  row at or before t.

Soft checks (severity "warn"):

* information coefficient implausibility: a mean daily rank IC far above
  what real signals achieve usually means the target leaked in.
* lag fragility: an edge that collapses when execution is delayed one more
  day is often an artefact of same-bar timing.
* survivorship: a long sample in which no asset ever stops trading is
  probably built from today's survivors.
* holdout reuse: a declared holdout window evaluated by more than one trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import run_backtest

SignalFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class Finding:
    check: str
    severity: str  # "fail" or "warn"
    detail: str


@dataclass
class LeakReport:
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "fail" for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "findings": [f.__dict__ for f in self.findings],
            "metrics": self.metrics,
        }


def _rows_equal(a: pd.Series, b: pd.Series, atol: float = 1e-10) -> bool:
    a, b = a.align(b, join="outer")
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    both_nan = np.isnan(av) & np.isnan(bv)
    close = np.isclose(av, bv, atol=atol, rtol=1e-9)
    return bool(np.all(both_nan | close))


def _cut_dates(index: pd.DatetimeIndex, n: int, seed: int, min_frac: float = 0.3) -> list:
    lo = int(len(index) * min_frac)
    candidates = index[lo:-1]
    if len(candidates) == 0:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    return sorted(candidates[i] for i in pick)


# ---------------------------------------------------------------- hard checks
def truncation_check(fn: SignalFn, returns: pd.DataFrame, n_cuts: int = 6, seed: int = 0) -> Finding | None:
    full = fn(returns)
    for t in _cut_dates(full.index, n_cuts, seed):
        part = fn(returns.loc[:t])
        if t not in part.index or not _rows_equal(full.loc[t], part.loc[t]):
            return Finding(
                "truncation", "fail",
                f"row {t.date()} changes when data after it is removed: the function uses future data",
            )
    return None


def perturbation_check(fn: SignalFn, returns: pd.DataFrame, n_cuts: int = 4, seed: int = 0) -> Finding | None:
    full = fn(returns)
    rng = np.random.default_rng(seed + 1)
    for t in _cut_dates(full.index, n_cuts, seed + 7):
        shocked = returns.copy()
        future = shocked.index > t
        noise = rng.normal(0, returns.std().mean() * 3, size=(int(future.sum()), returns.shape[1]))
        shocked.loc[future] = shocked.loc[future].to_numpy() * -2.0 + noise
        after = fn(shocked)
        past = full.index[full.index <= t]
        for d in past[-60:]:  # recent past is where leaks show; keeps it fast
            if d not in after.index or not _rows_equal(full.loc[d], after.loc[d]):
                return Finding(
                    "future_perturbation", "fail",
                    f"changing returns after {t.date()} changed the row for {d.date()}",
                )
    return None


# ---------------------------------------------------------------- soft checks
def mean_rank_ic(scores: pd.DataFrame, returns: pd.DataFrame, horizon: int = 1) -> float:
    """Mean daily Spearman correlation of scores at t with returns over t+1..t+horizon."""
    fwd = returns.rolling(horizon).sum().shift(-horizon)
    s, f = scores.align(fwd, join="inner")
    sr = s.rank(axis=1)
    fr = f.where(s.notna()).rank(axis=1)
    sr = sr.where(fr.notna())
    ic = sr.corrwith(fr, axis=1, method="pearson")
    return float(ic.dropna().mean()) if ic.notna().any() else float("nan")


def ic_check(scores: pd.DataFrame, returns: pd.DataFrame, max_ic: float = 0.1) -> tuple[Finding | None, float]:
    ic = mean_rank_ic(scores, returns)
    if np.isfinite(ic) and abs(ic) > max_ic:
        return Finding(
            "ic_implausible", "warn",
            f"mean daily rank IC {ic:.3f} exceeds {max_ic}; real daily signals rarely pass 0.05",
        ), ic
    return None, ic


def lag_check(weights: pd.DataFrame, returns: pd.DataFrame, min_sharpe: float = 1.0,
              keep_frac: float = 0.3) -> tuple[Finding | None, dict]:
    s = {lag: run_backtest(weights, returns, lag=lag).stats["sharpe"] for lag in (1, 2, 5)}
    if s[1] > min_sharpe and s[2] < keep_frac * s[1]:
        return Finding(
            "lag_fragility", "warn",
            f"Sharpe {s[1]:.2f} at lag 1 falls to {s[2]:.2f} at lag 2; check execution timing",
        ), s
    return None, s


def survivorship_check(returns: pd.DataFrame, min_years: float = 3.0, min_assets: int = 20) -> Finding | None:
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    if years < min_years or returns.shape[1] < min_assets:
        return None
    last_valid = returns.apply(lambda c: c.last_valid_index())
    exits = int((last_valid < returns.index[-30]).sum())
    if exits == 0:
        return Finding(
            "survivorship", "warn",
            f"none of {returns.shape[1]} assets stops trading over {years:.1f} years; "
            "universe is probably today's survivors",
        )
    return None


def holdout_check(ledger, start, end) -> Finding | None:
    """Flag a holdout window that more than one distinct trial has been evaluated on."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    m = ledger.returns_matrix()
    if m.empty:
        return None
    window = m.loc[(m.index >= start) & (m.index <= end)]
    touched = [c for c in window.columns if window[c].notna().any()]
    if len(touched) > 1:
        return Finding(
            "holdout_reuse", "warn",
            f"{len(touched)} trials were evaluated on the holdout {start.date()} to {end.date()}; "
            "it is no longer out of sample",
        )
    return None


# ---------------------------------------------------------------- audit
def audit(weights_fn: SignalFn, returns: pd.DataFrame, scores_fn: SignalFn | None = None,
          n_cuts: int = 6, seed: int = 0) -> LeakReport:
    """Run every signal-level check. ``scores_fn`` (raw scores) improves the IC check;
    without it the weights are used as scores."""
    rep = LeakReport()
    for chk in (truncation_check(weights_fn, returns, n_cuts, seed),
                perturbation_check(weights_fn, returns, max(2, n_cuts // 2), seed)):
        if chk:
            rep.findings.append(chk)
    if scores_fn is not None:
        for chk in (truncation_check(scores_fn, returns, n_cuts, seed + 3),):
            if chk:
                chk.detail = "scores: " + chk.detail
                rep.findings.append(chk)
    weights = weights_fn(returns)
    scores = scores_fn(returns) if scores_fn is not None else weights
    f, ic = ic_check(scores, returns)
    rep.metrics["mean_rank_ic"] = ic
    if f:
        rep.findings.append(f)
    f, lags = lag_check(weights, returns)
    rep.metrics["sharpe_by_lag"] = {str(k): v for k, v in lags.items()}
    if f:
        rep.findings.append(f)
    f = survivorship_check(returns)
    if f:
        rep.findings.append(f)
    return rep
