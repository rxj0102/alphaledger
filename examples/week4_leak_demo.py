"""Week 4 demo: audit planted leaks and clean signals, print what each check found.

Usage:
    python examples/week4_leak_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaledger.backtest import run_backtest
from alphaledger.leakage import audit
from alphaledger.signals import momentum, quantile_long_short
from alphaledger.synthetic import make_market

R = make_market(n_assets=80, n_days=252 * 4, drift_vol=0.0008, seed=0)


def mom(x):
    return momentum(x, 126, 5)


def backfill(x):
    s = mom(x)
    holes = s.mask(np.repeat((np.arange(len(s)) % 10 == 0)[:, None], s.shape[1], axis=1))
    return holes.bfill()


SIGNALS = {
    "clean: 6-1 momentum": mom,
    "clean: 1-day reversal": lambda x: -x,
    "leak: tomorrow's return": lambda x: x.shift(-1),
    "leak: centered 21-day window": lambda x: x.rolling(21, center=True).mean(),
    "leak: full-sample z-score": lambda x: (mom(x) - mom(x).mean()) / mom(x).std(),
    "leak: backward fill": backfill,
    "leak: month-end value all month": lambda x: mom(x).groupby(mom(x).index.to_period("M")).transform("last"),
}


def main() -> None:
    rows = []
    for name, fn in SIGNALS.items():
        w = lambda x, fn=fn: quantile_long_short(fn(x))
        rep = audit(w, R, scores_fn=fn)
        hard = sorted({f.check for f in rep.findings if f.severity == "fail"})
        soft = sorted({f.check for f in rep.findings if f.severity == "warn" and f.check != "survivorship"})
        rows.append({
            "signal": name,
            "naive Sharpe": round(run_backtest(w(R), R).stats["sharpe"], 2),
            "audit": "PASS" if rep.passed else "FAIL",
            "hard checks failed": ", ".join(hard) or "-",
            "warnings": ", ".join(soft) or "-",
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nEvery run also warns about survivorship: synthetic assets never delist.")


if __name__ == "__main__":
    main()
