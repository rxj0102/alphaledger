"""The README may only state numbers that code produces.

Fast demos (weeks 1, 3 and 4) are recomputed here. The long studies (about
10 to 15 minutes each) write their results to ``case_study/*.json`` when they
run; their README tables are checked against those files, so a README edit
that disagrees with the recorded results fails CI. Rerunning a study and
committing its JSON updates the source of truth.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
sys.path.insert(0, str(ROOT / "examples"))


def has_row(cells):
    row = "| " + " | ".join(cells) + " |"
    assert row in README, f"README is missing or disagrees with:\n{row}"


def pct(x):
    return f"{round(x * 100):.0f}%"


# ---------------------------------------------------------------- recomputed demos


def test_week1_table():
    from alphaledger.backtest import run_backtest
    from alphaledger.signals import momentum, quantile_long_short, rebalance
    from alphaledger.synthetic import make_market

    for label, drift in (("Planted momentum", 0.0008), ("Pure noise", 0.0)):
        r = make_market(n_assets=200, drift_vol=drift, seed=0)
        s = run_backtest(rebalance(quantile_long_short(momentum(r))), r, lag=1, cost_bps=5).stats
        has_row([label, f"{s['sharpe']:.2f}", f"{s['t_stat']:.2f}", f"{s['max_drawdown'] * 100:.1f}%"])


def test_week3_table():
    from week3_deflation_demo import sweep

    from alphaledger.stats import evaluate
    from alphaledger.synthetic import make_market

    out = []
    for drift in (0.0, 0.0008):
        out.append(evaluate(sweep(make_market(n_assets=150, drift_vol=drift, seed=21))).summary())
    n, p = out
    has_row(["Best naive annual Sharpe", f"{n['best_naive_sharpe']:.2f}", f"{p['best_naive_sharpe']:.2f}"])
    has_row(["Hurdle Sharpe for 60 trials (trial-spread version)",
             f"{n['hurdle_sharpe_annual']:.2f}", f"{p['hurdle_sharpe_annual']:.2f}"])
    has_row(["Deflated Sharpe Ratio of best (trial-spread version)", f"{n['best_dsr']:.3f}", f"{p['best_dsr']:.3f}"])
    has_row(["Significant before correction", str(n["n_significant_raw"]), str(p["n_significant_raw"])])
    has_row(["Significant after BH", str(n["n_significant_bh"]), str(p["n_significant_bh"])])
    has_row(["Passing DSR 0.95", str(n["n_pass_dsr_95"]), str(p["n_pass_dsr_95"])])
    has_row(["PBO", f"{n['pbo']:.2f}", f"{p['pbo']:.2f}"])


def test_week4_table():
    from week4_leak_demo import R, SIGNALS

    from alphaledger.backtest import run_backtest
    from alphaledger.leakage import audit
    from alphaledger.signals import quantile_long_short

    names = {"future_perturbation": "perturbation", "truncation": "truncation",
             "ic_implausible": "IC", "lag_fragility": "lag fragility"}
    for name, fn in SIGNALS.items():
        w = lambda x, fn=fn: quantile_long_short(fn(x))  # noqa: E731
        rep = audit(w, R, scores_fn=fn)
        hard = sorted(names[f.check] for f in rep.findings if f.severity == "fail")
        soft = [names[c] for c in ("ic_implausible", "lag_fragility")
                if any(f.check == c for f in rep.findings)]
        has_row([name, f"{run_backtest(w(R), R).stats['sharpe']:.2f}", "PASS" if rep.passed else "FAIL",
                 ", ".join(sorted(set(hard))) or "none", ", ".join(soft) or "none"])


# ---------------------------------------------------------------- recorded studies


def load(name):
    return json.loads((ROOT / "case_study" / name).read_text())


def test_controlled_study_table():
    s = load("controlled_study.json")["summary"]
    cols = [s[k] for k in ("planted/blind", "planted/preregistered", "noise/blind", "noise/preregistered")]
    has_row(["Trials (effective)"] + [f"{c['mean_trials']:.0f} ({c['mean_effective_trials']:.1f})" for c in cols])
    has_row(["Pick certified"] + [pct(c["certified_rate"]) for c in cols])
    has_row(["Any trial significant after BH"] + [pct(c["any_bh_significant_rate"]) for c in cols])
    has_row(["Raw p < 0.05, one-sided, per run"] + [f"{c['mean_raw_significant']:.1f}" for c in cols])
    has_row(["Pick in-sample Sharpe, mean"] + [f"{c['mean_pick_in_sample_sharpe']:.2f}" for c in cols])
    has_row(["Pick holdout Sharpe, mean"] + [f"{c['mean_pick_holdout_sharpe']:.2f}" for c in cols])
    has_row(["Pick holdout positive"] + [pct(c["pick_holdout_positive_rate"]) for c in cols])
    head = (f"got its best pick certified on {pct(cols[0]['certified_rate'])} of the markets that had a real edge; "
            f"an agent that tested 6 hypotheses written down in advance got it certified on "
            f"{pct(cols[1]['certified_rate'])}")
    assert head in README


def test_blind_case_study_table():
    noise, planted = load("results.json")
    assert (noise["market"], planted["market"]) == ("noise", "planted")
    a, b = noise["alphaledger"], planted["alphaledger"]
    has_row([f"Trials counted ({noise['proposals']} proposals; repeats and identical portfolios not counted)",
             str(noise["trials"]), str(planted["trials"])])
    has_row(["Effective independent trials", f"{a['effective_trials']:.1f}", f"{b['effective_trials']:.1f}"])
    has_row(["Naive pick: in-sample Sharpe", f"{noise['naive']['best_in_sample_sharpe']:.2f}",
             f"{planted['naive']['best_in_sample_sharpe']:.2f}"])
    has_row(["Raw p < 0.05, one-sided", str(noise["naive"]["significant_at_5pct_raw_one_sided"]),
             str(planted["naive"]["significant_at_5pct_raw_one_sided"])])
    has_row(["Hurdle Sharpe: effective / analytic / trial spread"] + [
        f"{x['hurdle_sharpe_effective']:.2f} / {x['hurdle_sharpe_analytic']:.2f} / {x['hurdle_sharpe']:.2f}"
        for x in (a, b)])
    has_row(["Best DSR: effective / analytic / trial spread"] + [
        f"{x['best_dsr_effective']:.2f} / {x['best_dsr_analytic']:.2f} / {x['best_dsr']:.3f}" for x in (a, b)])
    has_row(["PBO", f"{a['pbo']:.2f}", f"{b['pbo']:.2f}"])
    has_row(["Holdout Sharpe of the naive pick", f"{noise['holdout_of_naive_pick']['sharpe']:.2f}",
             f"{planted['holdout_of_naive_pick']['sharpe']:.2f}"])


def test_real_data_table():
    pre, blind = load("real_data_study.json")
    assert (pre["agent"], blind["agent"]) == ("preregistered", "blind")
    has_row(["Effective independent trials", f"{pre['effective_trials']:.1f}", f"{blind['effective_trials']:.1f}"])
    has_row(["Raw p < 0.05, one-sided", str(pre["raw_significant_one_sided"]),
             str(blind["raw_significant_one_sided"])])
    has_row(["Pick in-sample Sharpe", f"{pre['pick_in_sample_sharpe']:.2f}", f"{blind['pick_in_sample_sharpe']:.2f}"])
    has_row(["Hurdle Sharpe (effective trials)", f"{pre['hurdle_sharpe_effective']:.2f}",
             f"{blind['hurdle_sharpe_effective']:.2f}"])
    has_row(["Pick DSR: effective / analytic / trial spread"] + [
        f"{x['pick_dsr_effective']:.2f} / {x['pick_dsr_analytic']:.2f} / {x['pick_dsr_trial_spread']:.2f}"
        for x in (pre, blind)])
    has_row(["Certified (DSR effective at least 0.95)"] + ["yes" if x["certified"] else "no" for x in (pre, blind)])
    has_row(["PBO", f"{pre['pbo']:.2f}", f"{blind['pbo']:.2f}"])
    has_row(["Pick holdout Sharpe, 2019 to 2026", f"{pre['pick_holdout_sharpe']:.2f}",
             f"{blind['pick_holdout_sharpe']:.2f}"])
    assert pre["pick"] == "rank(momentum(252, 21))"
    assert f"`{pre['pick']}`" in README
    assert pre["holdout"].startswith("2019-") and pre["holdout"].endswith("2026-09-30")


@pytest.mark.parametrize("name", ["controlled_study.json", "results.json", "real_data_study.json",
                                  "claude_session.json"])
def test_recorded_results_exist(name):
    assert (ROOT / "case_study" / name).exists()


def test_claude_session_table():
    d = load("claude_session.json")
    rows = {r["expression"]: r for r in d["leaderboard"]}
    labels = {"0.5 * rank(momentum(126, 5)) + 0.5 * rank(ts_zscore(price, 252))": "blend of trials 4 and 5"}
    for i, expr in enumerate(d["expressions_in_order"], start=1):
        r = rows[expr]
        has_row([str(i), labels.get(expr, f"`{expr}`"), f"{r['naive_sharpe']:.2f}",
                 f"{r['deflated_sharpe_ratio_effective']:.3f}", f"{r['deflated_sharpe_ratio_analytic']:.3f}",
                 f"{r['deflated_sharpe_ratio_trial_spread']:.3f}"])
    assert f"(Values from the final leaderboard: {d['trials']} trials, {d['effective_trials']:.1f} effective.)" in README
    assert f"**The holdout came back at {d['holdout_sharpe']:.2f}**" in README
    o = d["other_seeds_holdout"]
    assert (f"holdout Sharpe of {o['mean']:.2f} on average (sd {o['sd']:.2f}, lowest {o['min']:.2f}), "
            f"so this was a {d['sd_below_mean']:.1f} sd bad draw") in README
