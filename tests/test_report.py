import re

import pytest

from alphaledger.gateway import ResearchGateway
from alphaledger.ledger import Ledger
from alphaledger.report import build_report, ledger_view
from alphaledger.synthetic import make_market

R = make_market(n_assets=40, n_days=252 * 3, drift_vol=0.0008, seed=0)


@pytest.fixture
def ledger_path(tmp_path):
    p = tmp_path / "l.db"
    gw = ResearchGateway(R, p)
    a = gw.test_signal("rank(momentum(120, 5))", rationale="root")["hypothesis_id"]
    gw.test_signal("rank(momentum(60, 5))", rationale="child", parent_id=a)
    gw.test_signal("rank(-ts_sum(returns, 5))", rationale="reversal")
    gw.test_signal("0.5 * rank(returns) - 0.5 * rank(returns)", rationale="flat")
    gw.ledger.close()
    return p


def test_view_counts(ledger_path):
    with Ledger(ledger_path) as lg:
        v = ledger_view(lg)
    assert v["n_hypotheses"] == 4
    assert v["status_counts"] == {"ok": 3, "degenerate": 1}
    assert v["chain_ok"]
    assert v["evaluation"]["summary"]["n_trials"] == 4


def test_report_is_self_contained_and_complete(ledger_path):
    with Ledger(ledger_path) as lg:
        page = build_report(lg, "test report")
    assert page.startswith("<!DOCTYPE html>")
    assert not re.search(r"(src|href)=['\"]https?://", page)  # no external assets
    assert page.count("<circle") == 3
    assert "rank(momentum(120, 5))" in page
    assert "verified" in page
    assert "prefers-color-scheme: dark" in page


def test_report_on_empty_ledger():
    page = build_report(Ledger(), "empty")
    assert "Hypotheses logged" in page and "<svg" not in page


def test_dashboard_renders(ledger_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("ALPHALEDGER_DB", str(ledger_path))
    at = testing.AppTest.from_function(_run_app, default_timeout=60)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Trials counted" in labels and "Best DSR" in labels
    values = {m.label: m.value for m in at.metric}
    assert values["Chain"] == "verified"
    assert values["Trials counted"] == "4"
    assert len(at.dataframe) == 1


def _run_app():
    from alphaledger.dashboard import app

    app()
