"""Week 6: replay of a research session in which Claude acted as the agent.

Claude used the gateway interactively, one decision at a time, on a synthetic
market (seed 303) where a momentum-type edge is planted. This script replays
the exact calls in order, with the rationale Claude gave for each, so every
number in the README write-up comes from running it.

Claude knew the generator plants a continuation edge, so this session shows
the workflow (priors first, refinements linked by lineage, deflated
numbers, one pre-chosen holdout test), not an unbiased discovery.

Usage:
    python examples/week6_claude_session.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alphaledger.backtest import run_backtest
from alphaledger.gateway import ResearchGateway
from alphaledger.report import build_report
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.synthetic import make_market

SESSION = [
    # (expression, q, rebalance, parent index or None, rationale)
    ("rank(momentum(252, 21))", 0.2, "M", None,
     "Prior: cross-sectional 12-1 momentum is the best documented equity anomaly; test it first as the baseline."),
    ("rank(-ts_sum(returns, 5))", 0.2, "W", None,
     "Prior: short-term reversal from liquidity provision; weekly horizon, so rebalance weekly."),
    ("rank(-ts_std(returns, 63))", 0.2, "M", None,
     "Prior: low-volatility anomaly; 3-month realized vol."),
    ("rank(momentum(126, 5))", 0.2, "M", 0,
     "Reversal lost money with high turnover, which suggests continuation at short horizons; "
     "test a 6-month lookback with a short skip as a refinement of the baseline."),
    ("rank(ts_zscore(price, 252))", 0.2, "M", 0,
     "Same continuation idea measured as a trend z-score instead of compounded return, "
     "to check the edge is not specific to one definition."),
    ("0.5 * rank(momentum(126, 5)) + 0.5 * rank(ts_zscore(price, 252))", 0.2, "M", 3,
     "Two continuation measures with similar Sharpe; averaging should cut noise if they are not identical."),
]
HOLDOUT_PICK = 3  # chosen before opening the holdout: strongest prior and best in-sample


def main() -> None:
    returns = make_market(n_assets=100, n_days=252 * 8, drift_vol=0.0006, seed=303)
    # The session was run when the gateway traded at the same close (lag 1);
    # the replay keeps that setting so it reproduces the decisions as made.
    gw = ResearchGateway(returns, holdout_frac=0.2, holdout_budget=2, cost_bps=5,
                         data_label="synthetic seed 303", execution_lag=1)
    ids = []
    print("Session")
    for expr, q, freq, parent, why in SESSION:
        out = gw.test_signal(expr, q=q, rebalance_freq=freq, rationale=why,
                             parent_id=ids[parent] if parent is not None else None)
        ids.append(out["hypothesis_id"])
        n, a = out["naive"], out["trial_adjusted"]
        print(f"  {len(ids)}. {expr}  [{freq}]")
        print(f"     naive Sharpe {n['sharpe']:.2f}, trials {a['trials_so_far']}, "
              f"effective trials {a['effective_independent_trials']}, "
              f"DSR effective {a['deflated_sharpe_ratio_effective']:.3f}, "
              f"analytic {a['deflated_sharpe_ratio_analytic']:.3f}, "
              f"trial spread {a['deflated_sharpe_ratio_trial_spread']:.3f}")

    board = gw.leaderboard(top=len(SESSION))
    print(f"\nLeaderboard: {board['n_trials']} trials ({board['effective_independent_trials']} effective), "
          f"PBO {board['probability_of_backtest_overfitting']}, "
          f"{board['n_significant_after_fdr']} significant after BH (one-sided)")
    for row in board["top"]:
        print(f"  {row['expression']}: naive {row['naive_sharpe']:.2f}, "
              f"DSR effective {row['deflated_sharpe_ratio_effective']:.3f}, "
              f"analytic {row['deflated_sharpe_ratio_analytic']:.3f}, "
              f"trial spread {row['deflated_sharpe_ratio_trial_spread']:.3f}, "
              f"BH {row['significant_after_fdr']}, haircut {row['haircut_sharpe']:.2f}")

    ho = gw.evaluate_holdout(ids[HOLDOUT_PICK])
    print(f"\nHoldout for {SESSION[HOLDOUT_PICK][0]}: Sharpe {ho['holdout_sharpe']:.2f} over {ho['holdout_days']} days")
    print(f"Lineage of the blend: {[x['expression'] for x in gw.lineage(ids[5])['lineage']]}")
    print(f"Ledger verifies: {gw.verify()['chain_ok']}")
    out = Path(__file__).resolve().parent.parent / "case_study"
    out.mkdir(exist_ok=True)
    (out / "report_claude_session.html").write_text(
        build_report(gw.ledger, "Claude as the agent, synthetic seed 303"))

    # How unusual is that holdout result for a strategy whose edge is real?
    sr = []
    for s in range(30):
        r = make_market(n_assets=100, n_days=252 * 8, drift_vol=0.0006, seed=1000 + s)
        net = run_backtest(rebalance(quantile_long_short(momentum(r, 126, 5))), r, cost_bps=5).returns
        tail = net.iloc[-ho["holdout_days"]:]
        sr.append(tail.mean() / tail.std() * np.sqrt(252))
    sr = np.array(sr)
    print(f"\nSame strategy, same holdout length, 30 other seeds of the same market model:")
    print(f"  holdout Sharpe mean {sr.mean():.2f}, sd {sr.std():.2f}, min {sr.min():.2f}, "
          f"share below zero {np.mean(sr < 0):.2f}")
    print(f"  this session's result is {(sr.mean() - ho['holdout_sharpe']) / sr.std():.1f} sd below the mean")


if __name__ == "__main__":
    main()
