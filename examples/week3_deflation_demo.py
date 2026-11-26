"""Week 3 demo: naive vs trial-adjusted results for the same parameter sweep.

Runs 60 momentum variants through the ledger on two synthetic markets
(pure noise, and noise plus a planted momentum edge), then evaluates every
trial with the deflated Sharpe ratio, Benjamini-Hochberg FDR control,
Holm haircut Sharpe and PBO.

Usage:
    python examples/week3_deflation_demo.py
"""

from __future__ import annotations

from alphaledger.ledger import Ledger, run_logged
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.stats import evaluate
from alphaledger.synthetic import make_market


def sweep(returns) -> Ledger:
    lg = Ledger()
    for lb in (252, 210, 168, 126, 84, 42):
        for sk in (1, 5, 21):
            if sk >= lb:
                continue
            for q in (0.1, 0.2, 0.3):
                for period in ("M", "W"):
                    if len(lg.hypotheses()) >= 60:
                        return lg
                    spec = {"signal": "momentum", "lookback": lb, "skip": sk, "q": q, "rebalance": period}
                    run_logged(
                        lg, spec,
                        lambda r, lb=lb, sk=sk, q=q, p=period: rebalance(quantile_long_short(momentum(r, lb, sk), q), p),
                        returns, cost_bps=5,
                    )
    return lg


def report(name: str, returns) -> None:
    ev = evaluate(sweep(returns))
    s = ev.summary()
    print(f"\n{name}")
    print(f"  trials: {s['n_trials']}")
    print(f"  best naive annual Sharpe: {s['best_naive_sharpe']:.2f}")
    print(f"  hurdle Sharpe for {s['n_trials']} trials: {s['hurdle_sharpe_annual']:.2f}")
    print(f"  deflated Sharpe ratio of best: {s['best_dsr']:.3f}")
    print(f"  significant at 5% before correction: {s['n_significant_raw']}")
    print(f"  significant after BH FDR control: {s['n_significant_bh']}")
    print(f"  passing DSR >= 0.95: {s['n_pass_dsr_95']}")
    print(f"  probability of backtest overfitting: {s['pbo']:.2f}")


def main() -> None:
    report("Pure noise (seed 21)", make_market(n_assets=150, drift_vol=0.0, seed=21))
    report("Planted momentum edge (seed 21)", make_market(n_assets=150, drift_vol=0.0008, seed=21))


if __name__ == "__main__":
    main()
