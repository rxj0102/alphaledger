"""Probabilistic and Deflated Sharpe Ratios.

References
----------
Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
Frontier. Journal of Risk 15(2).
Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
Journal of Portfolio Management 40(5), 94-107.

All Sharpe ratios here are per period (not annualized) and kurtosis is
Pearson kurtosis (3 for a normal distribution), as in the papers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def moments(r) -> tuple[float, float, float, int]:
    """Per-period Sharpe, skewness and Pearson kurtosis of a return series, plus T."""
    x = np.asarray(pd.Series(r).dropna(), dtype=float)
    t = len(x)
    if t < 3:
        raise ValueError("need at least 3 observations")
    sd = x.std(ddof=1)
    if sd == 0:
        raise ValueError("zero volatility")
    sr = x.mean() / sd
    z = (x - x.mean()) / x.std(ddof=0)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    return float(sr), skew, kurt, t


def psr(sr: float, t: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """Probability that the true Sharpe exceeds ``sr_benchmark`` given the estimate."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        raise ValueError("non-normality term is not positive; check skew/kurtosis inputs")
    z = (sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)
    return _N.cdf(z)


def expected_max_sr_factor(n_trials: int) -> float:
    """E[max of N standard normals], approximated as in Bailey and Lopez de Prado (2014)."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    return (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * _N.inv_cdf(
        1 - 1 / (n_trials * math.e)
    )


def expected_max_sr(n_trials: int, var_trial_sr: float, mean_trial_sr: float = 0.0) -> float:
    """Hurdle SR0: the best Sharpe expected from N zero-skill trials."""
    if var_trial_sr < 0:
        raise ValueError("variance must be non-negative")
    return mean_trial_sr + math.sqrt(var_trial_sr) * expected_max_sr_factor(n_trials)


def dsr(sr: float, t: int, n_trials: int, var_trial_sr: float, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio: PSR against the multiple-testing hurdle SR0."""
    return psr(sr, t, skew, kurt, expected_max_sr(n_trials, var_trial_sr))


def min_track_record_length(sr: float, skew: float = 0.0, kurt: float = 3.0,
                            sr_benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Observations needed for PSR(sr_benchmark) to reach ``confidence``."""
    if sr <= sr_benchmark:
        return float("inf")
    z = _N.inv_cdf(confidence)
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - sr_benchmark)) ** 2


@dataclass
class DeflatedSharpe:
    sr: float  # per period
    sr_annual: float
    t: int
    skew: float
    kurt: float
    n_trials: int
    sr0: float  # per-period hurdle
    psr: float  # vs zero
    dsr: float


def deflate(r, n_trials: int, var_trial_sr: float, periods: int = 252) -> DeflatedSharpe:
    """Full DSR calculation for one return series."""
    sr, skew, kurt, t = moments(r)
    sr0 = expected_max_sr(n_trials, var_trial_sr)
    return DeflatedSharpe(
        sr=sr, sr_annual=sr * math.sqrt(periods), t=t, skew=skew, kurt=kurt,
        n_trials=n_trials, sr0=sr0, psr=psr(sr, t, skew, kurt, 0.0),
        dsr=psr(sr, t, skew, kurt, sr0),
    )
