"""Evaluate every trial in a ledger with trial-adjusted statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .multiple import adjust, haircut_sharpe, p_from_t
from .pbo import PBOResult, pbo
from .sharpe import expected_max_sr, moments, psr


def participation_ratio(returns: pd.DataFrame, min_overlap: int = 60) -> float:
    """Spectral estimate of independent trials: N^2 / sum_ij C_ij^2.

    It is N when trials are uncorrelated and 1 when they are identical.
    Off-diagonal squared correlations are reduced by their sampling noise, 1/T,
    so unrelated trials over a finite sample are not mistaken for related ones.
    On its own it can be gamed: adding many copies of one trial drives it
    towards 1. ``effective_trials`` guards against that.
    """
    n = returns.shape[1]
    if n <= 1:
        return float(n)
    x = returns.to_numpy(dtype=float)
    mask = ~np.isnan(x)
    overlap = mask.T.astype(float) @ mask.astype(float)
    corr = returns.corr(min_periods=min_overlap).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        noise = np.where(overlap > 0, 1.0 / overlap, 0.0)
    sq = np.clip(np.nan_to_num(corr, nan=0.0) ** 2 - noise, 0.0, 1.0)
    np.fill_diagonal(sq, 1.0)
    return float(n * n / sq.sum())


def correlation_clusters(returns: pd.DataFrame, threshold: float = 0.5, min_overlap: int = 60) -> int:
    """Number of clusters from average-linkage clustering on correlation.

    Two groups merge while their average correlation is at least
    ``threshold``. Copies of one trial always land in one cluster, so this
    count cannot be lowered by repeating a trial. Pairs with too little
    overlap count as uncorrelated. O(N^3) in numpy; fine for a few hundred
    trials.
    """
    n = returns.shape[1]
    if n <= 1:
        return n
    corr = np.nan_to_num(returns.corr(min_periods=min_overlap).to_numpy(), nan=0.0)
    sim = corr.copy()
    np.fill_diagonal(sim, -np.inf)
    sizes = np.ones(n)
    active = np.ones(n, dtype=bool)
    clusters = n
    while clusters > 1:
        masked = np.where(active[:, None] & active[None, :], sim, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        if masked[i, j] < threshold:
            break
        # Average linkage: similarity of the merged group is the size-weighted mean.
        merged = (sizes[i] * sim[i] + sizes[j] * sim[j]) / (sizes[i] + sizes[j])
        sim[i, :] = merged
        sim[:, i] = merged
        sim[i, i] = -np.inf
        sizes[i] += sizes[j]
        active[j] = False
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf
        clusters -= 1
    return int(clusters)


def effective_trials(returns: pd.DataFrame, threshold: float = 0.5, min_overlap: int = 60) -> float:
    """Effective number of independent trials.

    The larger of the spectral estimate (``participation_ratio``) and the
    cluster count (``correlation_clusters``), following the clustering and
    spectral approaches suggested by Lopez de Prado. Taking the larger keeps it
    conservative and stops the count from being lowered by padding the ledger
    with copies of one trial.
    """
    if returns.shape[1] <= 1:
        return float(returns.shape[1])
    return max(participation_ratio(returns, min_overlap),
               float(correlation_clusters(returns, threshold, min_overlap)))


@dataclass
class Evaluation:
    table: pd.DataFrame  # one row per tested hypothesis, sorted by naive Sharpe
    n_trials: int
    var_trial_sr: float  # per-period, across trials
    sr0: float  # per-period hurdle
    pbo: PBOResult | None
    alpha: float
    median_t: int = 0
    n_eff: float = float("nan")

    def summary(self, periods: int = 252) -> dict:
        best = self.table.iloc[0]
        return {
            "n_trials": self.n_trials,
            "best_hypothesis": best.name,
            "best_naive_sharpe": float(best["sr_annual"]),
            "best_dsr": float(best["dsr"]),
            "best_dsr_analytic": float(best["dsr_analytic"]),
            "hurdle_sharpe_annual": self.sr0 * math.sqrt(periods),
            "hurdle_sharpe_annual_analytic": (
                expected_max_sr(self.n_trials, 1 / (self.median_t - 1)) * math.sqrt(periods)
                if self.median_t > 1 else float("nan")),
            "n_significant_raw": int((self.table["p_value"] <= self.alpha).sum()),
            "n_significant_bh": int(self.table["sig_bh"].sum()),
            "n_pass_dsr_95": int((self.table["dsr"] >= 0.95).sum()),
            "n_pass_dsr_analytic_95": int((self.table["dsr_analytic"] >= 0.95).sum()),
            "n_effective_trials": self.n_eff,
            "best_dsr_effective": float(best["dsr_effective"]),
            "hurdle_sharpe_annual_effective": (
                expected_max_sr(max(int(round(self.n_eff)), 1), 1 / (self.median_t - 1)) * math.sqrt(periods)
                if self.median_t > 1 else float("nan")),
            "n_pass_dsr_effective_95": int((self.table["dsr_effective"] >= 0.95).sum()),
            "pbo": None if self.pbo is None else self.pbo.pbo,
        }


def evaluate(ledger, alpha: float = 0.05, periods: int = 252, n_blocks: int = 16,
             max_splits: int | None = 2000) -> Evaluation:
    """Deflate every recorded trial.

    * N is ``ledger.n_trials()``, which includes failed runs (conservative).
    * Significance is one-sided: the question is whether a trial has positive
      skill. A large negative Sharpe (for example a high-turnover signal eaten
      by costs) is not a discovery.
    * Two deflated Sharpe ratios are reported:
      - ``dsr`` uses the variance of Sharpe ratios across the trials, as in
        Bailey and Lopez de Prado (2014). If trials differ for systematic
        reasons (costs, sign flips of the same idea) this variance, and so the
        hurdle, can be far larger than luck alone would produce.
      - ``dsr_analytic`` uses the sampling variance of a zero-skill Sharpe
        estimate, 1 / (T - 1). This is the hurdle for N independent trials
        with no skill, and is not inflated by systematic dispersion. It
        ignores correlation between trials, which makes it conservative when
        trials are highly correlated.
      - ``dsr_effective`` is the analytic version with N replaced by the
        effective number of independent trials (``effective_trials``), plus
        every failed, flat or leaky trial counted as fully independent. This
        is the recommended headline number.
    * PBO uses the dates every trial has in common.
    """
    m = ledger.returns_matrix()
    if m.shape[1] == 0:
        raise ValueError("ledger has no successful trials with returns")
    n_trials = max(ledger.n_trials(), m.shape[1])

    rows = {}
    for hid in m.columns:
        try:
            sr, skew, kurt, t = moments(m[hid])
        except ValueError:
            continue  # flat or too short; still counted in n_trials
        rows[hid] = {"sr": sr, "skew": skew, "kurt": kurt, "t_obs": t}
    if not rows:
        raise ValueError("ledger has no trials with usable returns")
    tab = pd.DataFrame.from_dict(rows, orient="index")
    var_sr = float(tab["sr"].var(ddof=1)) if len(tab) > 1 else 0.0
    sr0 = expected_max_sr(n_trials, var_sr)

    tab["sr_annual"] = tab["sr"] * math.sqrt(periods)
    tab["t_stat"] = tab["sr"] * np.sqrt(tab["t_obs"])
    tab["p_value"] = [p_from_t(x, two_sided=False) for x in tab["t_stat"]]
    # Failed trials have no p-value; treat them as p = 1 so they still count in m.
    n_missing = n_trials - len(tab)
    padded = np.concatenate([tab["p_value"].to_numpy(), np.ones(n_missing)])
    tab["p_bh"] = adjust(padded, "bh")[: len(tab)]
    tab["sig_bh"] = tab["p_bh"] <= alpha
    tab["psr"] = [psr(r.sr, r.t_obs, r.skew, r.kurt, 0.0) for r in tab.itertuples()]
    tab["dsr"] = [psr(r.sr, r.t_obs, r.skew, r.kurt, sr0) for r in tab.itertuples()]
    tab["sr0_analytic"] = [expected_max_sr(n_trials, 1 / (r.t_obs - 1)) for r in tab.itertuples()]
    tab["dsr_analytic"] = [psr(r.sr, r.t_obs, r.skew, r.kurt, r.sr0_analytic) for r in tab.itertuples()]
    n_eff = effective_trials(m[list(tab.index)]) + (n_trials - len(tab))
    n_eff = min(max(n_eff, 1.0), float(n_trials))
    k = max(int(round(n_eff)), 1)
    tab["sr0_effective"] = [expected_max_sr(k, 1 / (r.t_obs - 1)) for r in tab.itertuples()]
    tab["dsr_effective"] = [psr(r.sr, r.t_obs, r.skew, r.kurt, r.sr0_effective) for r in tab.itertuples()]
    haircuts = []
    for i, r in enumerate(tab.itertuples()):
        others = np.delete(padded, i)
        haircuts.append(haircut_sharpe(r.sr_annual, r.t_obs, n_trials, "holm", periods, others,
                                       two_sided=False)["sr_haircut"])
    tab["sr_haircut_holm"] = haircuts
    tab = tab.sort_values("sr_annual", ascending=False)
    tab.index.name = "hypothesis_id"

    common = m[list(tab.index)].dropna(how="any")
    pbo_res = None
    if m.shape[1] >= 2 and len(common) >= 2 * n_blocks:
        pbo_res = pbo(common, n_blocks=n_blocks, max_splits=max_splits)

    return Evaluation(table=tab, n_trials=n_trials, var_trial_sr=var_sr, sr0=sr0,
                      pbo=pbo_res, alpha=alpha, median_t=int(tab["t_obs"].median()), n_eff=n_eff)
