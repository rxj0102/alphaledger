"""Multiple-testing corrections and Sharpe haircuts.

References
----------
Benjamini, Y. and Hochberg, Y. (1995). Controlling the False Discovery Rate.
JRSS B 57(1).
Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery
rate in multiple testing under dependency. Annals of Statistics 29(4).
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
Harvey, C. R. and Liu, Y. (2015). Backtesting. Journal of Portfolio Management 42(1).
Harvey, C. R., Liu, Y. and Zhu, H. (2016). ... and the Cross-Section of
Expected Returns. Review of Financial Studies 29(1).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
METHODS = ("bonferroni", "holm", "bh", "by")


def p_from_t(t_stat: float, two_sided: bool = True) -> float:
    """Normal approximation p-value for a t-statistic (fine for daily samples)."""
    # erfc keeps precision in the far tail, where 1 - cdf would round to zero.
    if two_sided:
        p = math.erfc(abs(t_stat) / math.sqrt(2))
    else:
        p = 0.5 * math.erfc(t_stat / math.sqrt(2))
    return float(min(1.0, max(p, 1e-300)))


def t_from_p(p: float, two_sided: bool = True) -> float:
    p = min(max(p, 1e-300), 1.0)
    if p >= 1.0:
        return 0.0
    # Use the lower tail so tiny p-values do not round 1 - p to exactly 1.
    return float(-_N.inv_cdf(p / 2) if two_sided else -_N.inv_cdf(p))


def adjust(pvals, method: str = "bh") -> np.ndarray:
    """Adjusted p-values (compare each to alpha). Same order as input."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    order = np.argsort(p)
    ps = p[order]
    if method == "bonferroni":
        adj = np.minimum(ps * m, 1.0)
    elif method == "holm":
        adj = np.maximum.accumulate(ps * (m - np.arange(m)))
        adj = np.minimum(adj, 1.0)
    else:
        c = 1.0 if method == "bh" else np.sum(1.0 / np.arange(1, m + 1))
        raw = ps * m * c / np.arange(1, m + 1)
        adj = np.minimum.accumulate(raw[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def reject(pvals, alpha: float = 0.05, method: str = "bh") -> np.ndarray:
    return adjust(pvals, method) <= alpha


def haircut_sharpe(sr_annual: float, n_obs: int, n_trials: int, method: str = "bonferroni",
                   periods: int = 252, other_pvals=None, two_sided: bool = True) -> dict:
    """Harvey and Liu style Sharpe haircut.

    Convert the Sharpe ratio to a t-stat (t = SR_annual * sqrt(n_obs / periods)),
    adjust its p-value for multiple testing, convert back to a t-stat and then to
    the Sharpe ratio that t-stat implies. For Holm, BH and BY the other trials'
    p-values matter; if ``other_pvals`` is not given, the other N-1 trials are
    assumed insignificant (p = 1), which makes Holm equal to Bonferroni and is
    the conservative choice.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    years = n_obs / periods
    t_stat = sr_annual * math.sqrt(years)
    if not two_sided and t_stat <= 0:
        # One-sided test of positive skill: a non-positive Sharpe has nothing to keep.
        return {"sr_annual": sr_annual, "t_stat": t_stat, "p_value": p_from_t(t_stat, False),
                "p_adjusted": 1.0, "sr_haircut": min(sr_annual, 0.0), "haircut_pct": float("nan")}
    p = p_from_t(t_stat, two_sided)
    others = list(other_pvals) if other_pvals is not None else [1.0] * (n_trials - 1)
    if len(others) != n_trials - 1:
        raise ValueError("other_pvals must have n_trials - 1 entries")
    all_p = np.array([p] + others)
    p_adj = float(adjust(all_p, method)[0])
    t_adj = t_from_p(p_adj, two_sided)
    sr_adj = math.copysign(t_adj / math.sqrt(years), sr_annual) if t_adj > 0 else 0.0
    sr_adj = min(abs(sr_adj), abs(sr_annual)) * (1 if sr_annual >= 0 else -1)
    return {
        "sr_annual": sr_annual,
        "t_stat": t_stat,
        "p_value": p,
        "p_adjusted": p_adj,
        "sr_haircut": sr_adj,
        "haircut_pct": (1 - sr_adj / sr_annual) if sr_annual else float("nan"),
    }
