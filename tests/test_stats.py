import math

import numpy as np
import pandas as pd
import pytest

from alphaledger.ledger import Ledger, run_logged
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.stats import (
    adjust,
    deflate,
    dsr,
    evaluate,
    expected_max_sr,
    expected_max_sr_factor,
    haircut_sharpe,
    min_track_record_length,
    moments,
    pbo,
    psr,
    reject,
)
from alphaledger.synthetic import make_market

# ---------------------------------------------------------------- published examples


def test_dsr_matches_bailey_lopez_de_prado_2014_example():
    # Numerical example in the 2014 JPM paper: annualized SR 2.5 over 5 years
    # of daily data (T = 1250), skew -3, kurtosis 10, N = 100 trials, annualized
    # variance of trial Sharpes 0.5. The paper reports DSR = 0.9004.
    sr = 2.5 / math.sqrt(250)
    v = 0.5 / 250
    assert dsr(sr, 1250, 100, v, skew=-3, kurt=10) == pytest.approx(0.9004, abs=5e-5)


def test_dsr_matches_monthly_worked_example():
    # Worked example (quantt.co.uk DSR guide): annual SR 1.5 on 60 monthly
    # returns, skew -1, kurtosis 6, N = 100, per-period trial SR std 0.15.
    # Reported: bracket 2.531, SR0 0.3797, DSR 0.624.
    assert expected_max_sr_factor(100) == pytest.approx(2.531, abs=5e-4)
    assert expected_max_sr(100, 0.15**2) == pytest.approx(0.3797, abs=5e-4)
    # The guide rounds the z-score to 0.317 before taking Phi, hence the looser tolerance.
    assert dsr(1.5 / math.sqrt(12), 60, 100, 0.15**2, -1, 6) == pytest.approx(0.624, abs=1e-3)


def test_benjamini_hochberg_1995_example():
    # The 15 p-values from Benjamini and Hochberg (1995). At q = 0.05, BH rejects
    # 4 hypotheses; Bonferroni and Holm reject 3.
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
         0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
    assert reject(p, 0.05, "bh").sum() == 4
    assert reject(p, 0.05, "bonferroni").sum() == 3
    assert reject(p, 0.05, "holm").sum() == 3


@pytest.mark.parametrize("ours,theirs", [("bonferroni", "bonferroni"), ("holm", "holm"),
                                         ("bh", "fdr_bh"), ("by", "fdr_by")])
def test_adjust_matches_statsmodels(ours, theirs):
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 90)])
    rng.shuffle(p)
    expected = sm.multipletests(p, method=theirs)[1]
    np.testing.assert_allclose(adjust(p, ours), expected, rtol=1e-12)


# ---------------------------------------------------------------- properties


def test_psr_basic_properties():
    assert psr(0.0, 1000) == pytest.approx(0.5)
    assert psr(0.1, 1000) > psr(0.1, 100)
    # Negative skew and fat tails lower confidence for a positive Sharpe.
    assert psr(0.1, 500, skew=-2, kurt=9) < psr(0.1, 500)


def test_dsr_falls_as_trials_grow():
    vals = [dsr(0.1, 1000, n, 0.002) for n in (1, 10, 100, 1000)]
    assert vals == sorted(vals, reverse=True)
    assert dsr(0.1, 1000, 1, 0.002) == pytest.approx(psr(0.1, 1000))


def test_moments_of_normal_sample():
    x = np.random.default_rng(1).normal(0.001, 0.01, 200_000)
    sr, skew, kurt, t = moments(x)
    assert sr == pytest.approx(0.1, abs=0.005)
    assert skew == pytest.approx(0, abs=0.02)
    assert kurt == pytest.approx(3, abs=0.05)


def test_min_track_record_consistent_with_psr():
    n = min_track_record_length(0.1, confidence=0.95)
    assert psr(0.1, math.ceil(n)) >= 0.95
    assert psr(0.1, math.floor(n) - 1) < 0.95


def test_haircut_properties():
    one = haircut_sharpe(1.0, 252 * 10, 1)
    assert one["sr_haircut"] == pytest.approx(1.0, rel=1e-6)
    cuts = [haircut_sharpe(1.0, 252 * 10, n)["sr_haircut"] for n in (1, 10, 100, 1000)]
    assert cuts == sorted(cuts, reverse=True)
    weak = haircut_sharpe(0.3, 252 * 3, 200)
    assert weak["sr_haircut"] == 0.0


def test_extreme_t_stats_do_not_underflow():
    # Regression: a t-stat around 10 used to give p = 0 and crash the haircut.
    from alphaledger.stats import p_from_t, t_from_p

    for t in (2.0, 9.0, 12.0, 30.0):
        assert 0 < p_from_t(t) < 1
        assert t_from_p(p_from_t(t)) == pytest.approx(t, rel=1e-6)
    h = haircut_sharpe(5.0, 252 * 8, 60)
    assert 0 < h["sr_haircut"] <= 5.0


def test_deflate_returns_consistent_fields():
    r = make_market(n_assets=1, drift_vol=0.0, seed=4).iloc[:, 0]
    d = deflate(r, n_trials=50, var_trial_sr=0.001)
    assert d.sr_annual == pytest.approx(d.sr * math.sqrt(252))
    assert d.dsr <= d.psr


# ---------------------------------------------------------------- PBO


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, (2000, 50))
    res = pbo(x, n_blocks=10)
    assert res.n_splits == 252
    assert 0.3 < res.pbo < 0.7


def test_pbo_near_zero_with_one_real_edge():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, (2000, 50))
    x[:, 7] += 0.002  # one strategy with a real daily edge (annual SR about 3)
    res = pbo(x, n_blocks=10)
    assert res.pbo < 0.05
    assert res.oos_loss_prob < 0.05


def test_pbo_input_checks():
    with pytest.raises(ValueError):
        pbo(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        pbo(np.zeros((100, 4)), n_blocks=5)


# ---------------------------------------------------------------- ledger evaluation


def sweep(drift, seed=5):
    r = make_market(n_assets=150, drift_vol=drift, seed=seed)
    lg = Ledger()
    for lb in (252, 189, 126, 63):
        for sk in (5, 21):
            for q in (0.1, 0.2, 0.3):
                run_logged(lg, {"lb": lb, "sk": sk, "q": q},
                           lambda x, lb=lb, sk=sk, q=q: rebalance(quantile_long_short(momentum(x, lb, sk), q)), r)
    return lg


def test_evaluate_on_noise_sweep_finds_nothing():
    ev = evaluate(sweep(0.0))
    s = ev.summary()
    assert s["n_trials"] == 24
    assert s["n_significant_bh"] == 0
    assert s["n_pass_dsr_95"] == 0
    assert s["best_dsr"] < 0.95


def test_evaluate_on_real_edge_keeps_it():
    ev = evaluate(sweep(0.0008))
    s = ev.summary()
    assert s["n_significant_bh"] >= 1
    assert s["best_dsr"] > 0.95


def test_failed_trials_raise_the_bar():
    lg = sweep(0.0)
    before = evaluate(lg)
    r = make_market(n_assets=10, n_days=300, seed=0)
    for i in range(30):
        run_logged(lg, {"broken": i}, lambda x: 1 / 0, r)
    after = evaluate(lg)
    assert after.n_trials == before.n_trials + 30
    assert after.sr0 > before.sr0
    best = before.table.index[0]
    assert after.table.loc[best, "dsr"] < before.table.loc[best, "dsr"]


# ---------------------------------------------------------------- one-sided and analytic null


def _ledger_from_matrix(x):
    from alphaledger.ledger import Ledger

    lg = Ledger()
    idx = pd.bdate_range("2015-01-01", periods=x.shape[0])
    for j in range(x.shape[1]):
        h = lg.register({"col": j})
        lg.record(h, {"sharpe": 0}, returns=pd.Series(x[:, j], index=idx))
    return lg


def test_losers_are_not_discoveries():
    rng = np.random.default_rng(8)
    x = rng.normal(0, 0.01, (1500, 20))
    x[:, 0] -= 0.003  # hugely negative, like a strategy eaten by costs
    ev = evaluate(_ledger_from_matrix(x))
    assert not ev.table.loc[ev.table["sr"] < 0, "sig_bh"].any()


def test_analytic_hurdle_matches_empirical_for_independent_noise():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 0.01, (1500, 300))
    ev = evaluate(_ledger_from_matrix(x))
    s = ev.summary()
    assert s["hurdle_sharpe_annual_analytic"] == pytest.approx(s["hurdle_sharpe_annual"], rel=0.12)


def test_systematic_dispersion_inflates_empirical_hurdle_only():
    rng = np.random.default_rng(10)
    x = rng.normal(0, 0.01, (1500, 40))
    x[:, :10] -= 0.002  # ten trials share a systematic drag
    # One real edge with realized annual Sharpe of exactly about 1.9 (demeaned
    # noise plus a fixed drift), so the test does not depend on a lucky draw.
    x[:, 39] = x[:, 39] - x[:, 39].mean() + 0.0012
    ev = evaluate(_ledger_from_matrix(x))
    s = ev.summary()
    assert s["hurdle_sharpe_annual"] > 2 * s["hurdle_sharpe_annual_analytic"]
    best = ev.table.iloc[0]
    assert best["dsr_analytic"] > 0.95 > best["dsr"]


# ---------------------------------------------------------------- effective number of trials


def test_effective_trials_independent_and_identical():
    from alphaledger.stats import effective_trials

    rng = np.random.default_rng(11)
    ind = pd.DataFrame(rng.normal(0, 0.01, (1500, 50)))
    assert effective_trials(ind) == pytest.approx(50, rel=0.1)
    base = rng.normal(0, 0.01, (1500, 1))
    same = pd.DataFrame(base + rng.normal(0, 0.0001, (1500, 50)))
    assert effective_trials(same) < 1.1


def test_effective_trials_two_clusters():
    from alphaledger.stats import effective_trials

    rng = np.random.default_rng(12)
    a, b = rng.normal(0, 0.01, (1500, 1)), rng.normal(0, 0.01, (1500, 1))
    x = np.hstack([a + rng.normal(0, 0.001, (1500, 20)), b + rng.normal(0, 0.001, (1500, 20))])
    assert 1.8 < effective_trials(pd.DataFrame(x)) < 2.3


def test_parameter_sweep_gets_lower_hurdle_than_raw_count():
    rng = np.random.default_rng(13)
    base = rng.normal(0, 0.01, (1500, 1))
    x = base + rng.normal(0, 0.003, (1500, 40))  # 40 tweaks of one idea
    x[:, 0] = x[:, 0] - x[:, 0].mean() + 0.0010  # the best tweak has a real edge
    ev = evaluate(_ledger_from_matrix(x))
    s = ev.summary()
    assert s["n_effective_trials"] < 5
    assert s["hurdle_sharpe_annual_effective"] < s["hurdle_sharpe_annual_analytic"]
    assert s["best_dsr_effective"] >= s["best_dsr_analytic"]


def test_effective_trials_cannot_be_lowered_by_padding_with_copies():
    from alphaledger.stats import correlation_clusters, effective_trials, participation_ratio

    rng = np.random.default_rng(14)
    ind = rng.normal(0, 0.01, (1500, 3))
    padded = np.hstack([ind, ind[:, [0]] + rng.normal(0, 0.001, (1500, 100))])
    df = pd.DataFrame(padded)
    assert participation_ratio(df) < 1.5  # the spectral estimate alone is gameable
    assert correlation_clusters(df) == 3
    assert effective_trials(df) >= 3
    assert effective_trials(df) >= effective_trials(pd.DataFrame(ind)) - 0.5


def test_clusters_on_known_blocks():
    from alphaledger.stats import correlation_clusters

    rng = np.random.default_rng(15)
    blocks = [rng.normal(0, 0.01, (1500, 1)) + rng.normal(0, 0.003, (1500, k)) for k in (5, 7, 3)]
    assert correlation_clusters(pd.DataFrame(np.hstack(blocks))) == 3
