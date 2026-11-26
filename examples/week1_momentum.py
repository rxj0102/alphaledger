"""Week 1 sanity check: 12-1 momentum, monthly rebalance, 1-day execution lag.

Usage:
    python examples/week1_momentum.py                     # synthetic market with a planted edge
    python examples/week1_momentum.py --noise             # synthetic pure noise (edge should vanish)
    python examples/week1_momentum.py --yfinance AAPL MSFT ...   # real prices (needs network, pip install yfinance)
"""

from __future__ import annotations

import argparse

from alphaledger.backtest import run_backtest
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.synthetic import make_market


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true", help="synthetic market with no edge")
    ap.add_argument("--yfinance", nargs="+", metavar="TICKER", help="use real yfinance prices")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.yfinance:
        from alphaledger.data.sources import fetch_yfinance_close, prices_to_returns

        returns = prices_to_returns(fetch_yfinance_close(args.yfinance, args.start))
        label = f"yfinance, {returns.shape[1]} tickers (survivorship-biased universe)"
    else:
        drift = 0.0 if args.noise else 0.0008
        returns = make_market(n_assets=200, drift_vol=drift, seed=args.seed)
        label = f"synthetic, {'pure noise' if args.noise else 'planted momentum'}, seed {args.seed}"

    weights = rebalance(quantile_long_short(momentum(returns)))
    res = run_backtest(weights, returns, lag=1, cost_bps=args.cost_bps)

    print(f"Data: {label}")
    print(f"Period: {res.returns.index[0].date()} to {res.returns.index[-1].date()}")
    for k, v in res.stats.items():
        print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")


if __name__ == "__main__":
    main()
