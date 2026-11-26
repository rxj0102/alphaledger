"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Reference: Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017).
The Probability of Backtest Overfitting. Journal of Computational Finance 20(4).

Procedure: split the T x N matrix of trial returns into S equal row blocks.
For every way of choosing S/2 blocks as in-sample (the rest out-of-sample),
pick the trial with the best in-sample Sharpe and record its relative rank
out-of-sample. PBO is the share of splits where that pick lands at or below
the out-of-sample median (logit <= 0).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = x.mean(axis=0) / sd
    return np.where(np.isfinite(s), s, -np.inf)


@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_splits: int
    n_blocks: int
    n_trials: int
    is_best_sharpe: np.ndarray  # in-sample Sharpe of the selected trial, per split
    oos_sharpe: np.ndarray  # its out-of-sample Sharpe, per split

    @property
    def oos_loss_prob(self) -> float:
        """Share of splits where the in-sample winner lost money out of sample."""
        return float(np.mean(self.oos_sharpe < 0))

    @property
    def degradation_slope(self) -> float:
        """Slope of OOS Sharpe on IS Sharpe across splits (negative means overfit)."""
        if np.std(self.is_best_sharpe) == 0:
            return float("nan")
        return float(np.polyfit(self.is_best_sharpe, self.oos_sharpe, 1)[0])


def pbo(returns: pd.DataFrame | np.ndarray, n_blocks: int = 16, max_splits: int | None = None,
        seed: int = 0) -> PBOResult:
    x = returns.dropna(how="any").to_numpy() if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    t, n = x.shape
    if n < 2:
        raise ValueError("need at least 2 trials")
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    if t < n_blocks * 2:
        raise ValueError("too few observations for this many blocks")
    blocks = np.array_split(np.arange(t), n_blocks)
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]

    logits, is_best, oos = [], [], []
    for c in combos:
        is_idx = np.concatenate([blocks[i] for i in c])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in c])
        s_is = _sharpe_cols(x[is_idx])
        s_oos = _sharpe_cols(x[oos_idx])
        best = int(np.argmax(s_is))
        # Relative rank of the IS winner among OOS Sharpes, in (0, 1).
        rank = (s_oos < s_oos[best]).sum() + 0.5 * ((s_oos == s_oos[best]).sum() - 1) + 1
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
        is_best.append(s_is[best])
        oos.append(s_oos[best])

    logits = np.array(logits)
    return PBOResult(
        pbo=float(np.mean(logits <= 0)), logits=logits, n_splits=len(combos),
        n_blocks=n_blocks, n_trials=n,
        is_best_sharpe=np.array(is_best), oos_sharpe=np.array(oos),
    )
