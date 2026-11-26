import numpy as np
import pandas as pd
import pytest

from alphaledger.leakage import (
    audit,
    holdout_check,
    perturbation_check,
    survivorship_check,
    truncation_check,
)
from alphaledger.ledger import Ledger, run_logged
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.stats import evaluate
from alphaledger.synthetic import make_market

R = make_market(n_assets=80, n_days=252 * 4, drift_vol=0.0008, seed=0)


def mom(x):
    return momentum(x, 126, 5)


# ---------------------------------------------------------------- planted leaks
def leak_tomorrow(x):
    return x.shift(-1)


def leak_centered_window(x):
    return x.rolling(21, center=True).mean()


def leak_full_sample_zscore(x):
    s = mom(x)
    return (s - s.mean()) / s.std()  # per-asset mean and std over the whole sample


def leak_backfill(x):
    s = mom(x)
    mask = pd.DataFrame(np.arange(len(s)) % 10 == 0, index=s.index).reindex(columns=[0])
    holes = s.mask(np.repeat(mask.to_numpy(), s.shape[1], axis=1))
    return holes.bfill()  # each hole is filled with the next (future) value


def leak_month_last_value(x):
    s = mom(x)
    return s.groupby(s.index.to_period("M")).transform("last")  # month-end value on every day


def leak_sign_fit_full_sample(x):
    s = mom(x)
    fwd = x.shift(-1)
    sign = np.sign(s.stack().corr(fwd.stack()))  # direction chosen with hindsight
    return sign * s


LEAKS = [leak_tomorrow, leak_centered_window, leak_full_sample_zscore,
         leak_backfill, leak_month_last_value]


def test_known_limitation_discrete_hindsight_choice():
    # Choosing the signal's sign with full-sample hindsight is a leak, but when
    # the edge is strong the chosen sign is the same on every truncated sample,
    # so the output is identical and no data-level check can see it. The defence
    # is the ledger: trying both signs is two trials, and the stats deflate for it.
    assert truncation_check(leak_sign_fit_full_sample, R) is None
    assert perturbation_check(leak_sign_fit_full_sample, R) is None


@pytest.mark.parametrize("leak", LEAKS, ids=lambda f: f.__name__)
def test_every_planted_leak_fails_audit(leak):
    rep = audit(lambda x: quantile_long_short(leak(x)), R, scores_fn=leak)
    assert not rep.passed
    assert any(f.check in ("truncation", "future_perturbation") for f in rep.findings)


@pytest.mark.parametrize("leak", LEAKS, ids=lambda f: f.__name__)
def test_truncation_alone_catches_every_leak(leak):
    assert truncation_check(leak, R) is not None


# ---------------------------------------------------------------- clean signals pass
CLEAN = {
    "momentum_monthly": (lambda x: rebalance(quantile_long_short(mom(x))), mom),
    "momentum_daily": (lambda x: quantile_long_short(mom(x)), mom),
    "reversal_1d": (lambda x: quantile_long_short(-x), lambda x: -x),
    "trailing_zscore": (
        lambda x: quantile_long_short((mom(x) - mom(x).rolling(252).mean()) / mom(x).rolling(252).std()),
        None,
    ),
    "forward_fill": (lambda x: quantile_long_short(mom(x).ffill()), None),
}


@pytest.mark.parametrize("name", list(CLEAN))
def test_clean_signals_pass(name):
    w, s = CLEAN[name]
    rep = audit(w, R, scores_fn=s)
    assert rep.passed, [f.detail for f in rep.findings]
    assert not any(f.check in ("ic_implausible", "lag_fragility") for f in rep.findings)


def test_perturbation_catches_shift_but_not_clean():
    assert perturbation_check(leak_tomorrow, R) is not None
    assert perturbation_check(mom, R) is None


# ---------------------------------------------------------------- soft checks
def test_ic_and_lag_flags_on_tomorrow_leak():
    rep = audit(lambda x: quantile_long_short(leak_tomorrow(x)), R, scores_fn=leak_tomorrow)
    checks = {f.check for f in rep.findings}
    assert {"ic_implausible", "lag_fragility"} <= checks
    assert rep.metrics["mean_rank_ic"] > 0.9


def test_survivorship_flag():
    assert survivorship_check(R) is not None  # synthetic universe never delists
    delisted = R.copy()
    delisted.iloc[500:, :5] = np.nan
    assert survivorship_check(delisted) is None


def test_holdout_reuse():
    lg = Ledger()
    hold = (R.index[-252], R.index[-1])
    run_logged(lg, {"a": 1}, lambda x: rebalance(quantile_long_short(mom(x))), R)
    assert holdout_check(lg, *hold) is None
    run_logged(lg, {"a": 2}, lambda x: rebalance(quantile_long_short(momentum(x, 63, 5))), R)
    f = holdout_check(lg, *hold)
    assert f is not None and "2 trials" in f.detail


# ---------------------------------------------------------------- ledger integration
def test_leaky_trial_is_counted_but_excluded():
    lg = Ledger()
    good, _ = run_logged(lg, {"s": "mom"}, lambda x: quantile_long_short(mom(x)), R,
                         audit_leaks=True, scores_fn=mom)
    bad, stats = run_logged(lg, {"s": "tomorrow"}, lambda x: quantile_long_short(leak_tomorrow(x)), R,
                            audit_leaks=True, scores_fn=leak_tomorrow)
    t = lg.trials().set_index("hypothesis_id")
    assert t.loc[good, "status"] == "ok"
    assert t.loc[bad, "status"] == "leak"
    assert stats["leak_audit"]["passed"] is False
    assert lg.n_trials() == 2
    assert list(lg.returns_matrix().columns) == [good]
    assert evaluate(lg).n_trials == 2
    assert lg.verify().ok
