import sqlite3

import pandas as pd
import pytest

from alphaledger.ledger import Ledger, LedgerError, data_fingerprint, run_logged
from alphaledger.signals import momentum, quantile_long_short, rebalance
from alphaledger.synthetic import make_market


def fixed_clock():
    n = {"i": 0}

    def clock():
        n["i"] += 1
        return f"2026-01-01T00:00:{n['i']:02d}+00:00"

    return clock


def filled(path=":memory:"):
    lg = Ledger(path, clock=fixed_clock())
    a = lg.register({"signal": "mom", "lookback": 252})
    lg.record(a, {"sharpe": 0.8})
    b = lg.register({"signal": "mom", "lookback": 126}, parent_id=a)
    lg.record(b, {"sharpe": 1.1})
    c = lg.register({"signal": "mom", "lookback": 63}, parent_id=b)
    lg.record(c, {"sharpe": float("nan")}, status="error", error="boom")
    return lg, (a, b, c)


def raw(lg):
    """Bypass the append-only triggers, as an attacker editing the file would."""
    conn = lg.conn
    conn.execute("DROP TRIGGER entries_no_update")
    conn.execute("DROP TRIGGER entries_no_delete")
    return conn


def test_clean_chain_verifies():
    lg, _ = filled()
    rep = lg.verify()
    assert rep.ok and rep.n_entries == 6


def test_triggers_block_update_and_delete():
    lg, _ = filled()
    with pytest.raises(sqlite3.DatabaseError):
        lg.conn.execute("UPDATE entries SET payload='{}' WHERE seq=2")
    with pytest.raises(sqlite3.DatabaseError):
        lg.conn.execute("DELETE FROM entries WHERE seq=2")


def test_edited_result_is_detected():
    lg, _ = filled()
    conn = raw(lg)
    conn.execute("UPDATE entries SET payload=replace(payload, '0.8', '2.8') WHERE seq=2")
    rep = lg.verify()
    assert not rep.ok and rep.first_bad_seq == 2


def test_edit_with_rehash_breaks_next_link():
    # A smarter attacker recomputes the edited entry's own hash. The next
    # entry still points at the old hash, so the chain breaks there.
    from alphaledger.ledger.store import entry_hash

    lg, _ = filled()
    conn = raw(lg)
    row = conn.execute("SELECT seq, ts, kind, hypothesis_id, parent_id, payload, prev_hash FROM entries WHERE seq=2").fetchone()
    new_payload = row[5].replace("0.8", "2.8")
    new_hash = entry_hash(row[0], row[1], row[2], row[3], row[4], new_payload, row[6])
    conn.execute("UPDATE entries SET payload=?, hash=? WHERE seq=2", (new_payload, new_hash))
    rep = lg.verify()
    assert not rep.ok and rep.first_bad_seq == 3


def test_deleted_failure_is_detected():
    lg, _ = filled()
    conn = raw(lg)
    conn.execute("DELETE FROM entries WHERE seq=4")
    rep = lg.verify()
    assert not rep.ok and rep.first_bad_seq == 4


def test_truncation_caught_only_with_published_head():
    lg, _ = filled()
    head = lg.head_hash()
    conn = raw(lg)
    conn.execute("DELETE FROM entries WHERE seq>=5")  # drop the failed trial at the end
    assert lg.verify().ok  # a shorter chain is internally valid
    rep = lg.verify(expected_head=head)
    assert not rep.ok and "not found" in rep.reason


def test_published_head_still_valid_after_more_appends():
    lg, (a, _, _) = filled()
    head = lg.head_hash()
    lg.note("later work", hypothesis_id=a)
    assert lg.verify(expected_head=head).ok


def test_persistence_across_reopen(tmp_path):
    p = tmp_path / "ledger.db"
    lg, _ = filled(p)
    head = lg.head_hash()
    lg.close()
    with Ledger(p) as lg2:
        assert lg2.verify(expected_head=head).ok
        assert len(lg2.hypotheses()) == 3


def test_lineage_and_children():
    lg, (a, b, c) = filled()
    assert lg.lineage(c) == [a, b, c]
    assert lg.children(a) == [b]
    with pytest.raises(LedgerError):
        lg.register({"x": 1}, parent_id="h_doesnotexist")


def test_failures_count_as_trials_and_duplicates_do_not():
    lg, (a, _, _) = filled()
    assert lg.n_trials() == 3  # includes the errored run
    d = lg.register({"lookback": 252, "signal": "mom"})  # same spec, different key order
    lg.record(d, {"sharpe": 0.8})
    assert lg.hypotheses().set_index("hypothesis_id").loc[d, "duplicate_of"] == a
    assert lg.n_trials() == 3


def test_registered_but_untested_does_not_count():
    lg, _ = filled()
    lg.register({"signal": "value"})
    assert lg.n_trials() == 3
    t = lg.trials()
    assert t["result_seq"].isna().sum() == 1


def test_record_unknown_hypothesis_raises():
    lg, _ = filled()
    with pytest.raises(LedgerError):
        lg.record("h_nope", {"sharpe": 1})


def test_run_logged_end_to_end():
    r = make_market(n_assets=60, n_days=600, seed=2)
    lg = Ledger()
    hid, stats = run_logged(
        lg, {"signal": "mom", "lookback": 120, "skip": 5},
        lambda x: rebalance(quantile_long_short(momentum(x, 120, 5))), r,
    )
    assert stats is not None
    bad, bad_stats = run_logged(lg, {"signal": "broken"}, lambda x: 1 / 0, r)
    assert bad_stats is None
    t = lg.trials().set_index("hypothesis_id")
    assert t.loc[bad, "status"] == "error"
    assert t.loc[hid, "data_hash"] == data_fingerprint(r)
    m = lg.returns_matrix()
    assert list(m.columns) == [hid]
    from alphaledger.backtest import run_backtest

    fresh = run_backtest(rebalance(quantile_long_short(momentum(r, 120, 5))), r).returns
    pd.testing.assert_series_equal(m[hid], fresh, check_names=False, check_freq=False)
    assert lg.verify().ok


def test_data_fingerprint_changes_with_data():
    r = make_market(n_assets=5, n_days=50, seed=0)
    r2 = r.copy()
    r2.iloc[10, 2] += 1e-9
    assert data_fingerprint(r) == data_fingerprint(r.copy())
    assert data_fingerprint(r) != data_fingerprint(r2)


def test_duplicates_mixed_with_originals_keep_clean_ids():
    # Regression: with pandas 3, None in duplicate_of became NaN, NaN is truthy,
    # and returns_matrix produced a NaN column that crashed the report.
    r = make_market(n_assets=30, n_days=400, seed=4)
    lg = Ledger()
    build = lambda x: quantile_long_short(momentum(x, 60, 5))
    a, _ = run_logged(lg, {"s": "a"}, build, r)
    b, _ = run_logged(lg, {"s": "b"}, build, r)
    run_logged(lg, {"s": "a"}, build, r)  # duplicate of a
    m = lg.returns_matrix()
    assert sorted(m.columns) == sorted([a, b])
    h = lg.hypotheses()
    assert h["duplicate_of"].tolist()[:2] == [None, None]
    from alphaledger.report import build_report

    assert "<svg" in build_report(lg)
