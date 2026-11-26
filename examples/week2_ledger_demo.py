"""Week 2 demo: log a small parameter sweep, then show tampering being caught.

Usage:
    python examples/week2_ledger_demo.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from alphaledger.ledger import Ledger, run_logged
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.synthetic import make_market


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "ledger.db"
    returns = make_market(n_assets=100, drift_vol=0.0, seed=11)  # pure noise on purpose

    with Ledger(db) as lg:
        parent = None
        for lookback in (252, 189, 126, 63, 21):
            for skip in (0, 5, 21):
                if skip >= lookback:
                    continue
                spec = {"signal": "momentum", "lookback": lookback, "skip": skip, "q": 0.2}
                hid, _ = run_logged(
                    lg, spec,
                    lambda r, lb=lookback, sk=skip: rebalance(quantile_long_short(momentum(r, lb, sk))),
                    returns, parent_id=parent, cost_bps=5,
                )
                parent = parent or hid
        trials = lg.trials()
        head = lg.head_hash()
        print(f"Trials logged: {lg.n_trials()}")
        best = trials.sort_values("sharpe", ascending=False).iloc[0]
        print(f"Best naive Sharpe on pure noise: {best['sharpe']:.2f} (hypothesis {best['hypothesis_id']})")
        print(f"Chain verifies: {lg.verify().ok}")
        print(f"Published head hash: {head[:16]}...")

    # Tamper with a copy: delete the worst trial, as someone hiding failures might.
    copy = tmp / "tampered.db"
    shutil.copy(db, copy)
    with Ledger(copy) as lg:
        worst_seq = int(lg.trials().sort_values("sharpe").iloc[0]["result_seq"])
        lg.conn.execute("DROP TRIGGER entries_no_delete")
        lg.conn.execute("DELETE FROM entries WHERE seq=?", (worst_seq,))
        rep = lg.verify(expected_head=head)
        print(f"After deleting entry {worst_seq}: verifies={rep.ok}, first bad entry={rep.first_bad_seq}, reason: {rep.reason}")


if __name__ == "__main__":
    main()
