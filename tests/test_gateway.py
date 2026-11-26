import asyncio
import json
import sys

import numpy as np
import pandas as pd
import pytest

from alphaledger.gateway import ExpressionError, ResearchGateway, compile_expression
from alphaledger.gateway.expr import FUNCTIONS
from alphaledger.leakage import perturbation_check, truncation_check
from alphaledger.synthetic import make_market

R = make_market(n_assets=40, n_days=252 * 3, drift_vol=0.0008, seed=0)

# ---------------------------------------------------------------- expression language

EVERY_OPERATOR = [
    "ts_sum(returns, 20)", "ts_mean(returns, 20)", "ts_std(returns, 20)", "ts_min(returns, 20)",
    "ts_max(returns, 20)", "ts_zscore(price, 60)", "ts_rank(price, 20)", "ewm(returns, 10)",
    "delay(returns, 3)", "delta(price, 5)", "momentum(120, 5)", "rank(returns)",
    "zscore(returns)", "demean(returns)", "abs(returns)", "sign(returns)",
    "clip(returns, -0.02, 0.02)", "rank(momentum(60, 5)) - 0.5 * zscore(ts_std(returns, 20))",
    "-returns / (1 + ts_std(returns, 10))",
]


def test_operator_list_is_covered():
    used = {f for e in EVERY_OPERATOR for f in FUNCTIONS if f + "(" in e}
    assert used == set(FUNCTIONS)


@pytest.mark.parametrize("expr", EVERY_OPERATOR)
def test_every_operator_is_causal(expr):
    c = compile_expression(expr)
    assert truncation_check(c, R, n_cuts=4) is None
    assert perturbation_check(c, R, n_cuts=2) is None


@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')",
    "returns.shift(-1)",
    "open('/etc/passwd')",
    "(lambda: 1)()",
    "[x for x in returns]",
    "returns[0]",
    "delay(returns, -1)",
    "ts_mean(returns, 0)",
    "ts_mean(returns, 99999)",
    "ts_mean(returns, n)",
    "ts_mean(returns, 2.5)",
    "rank(returns, 3)",
    "rank(x=returns)",
    "momentum(5, 10)",
    "foo(returns)",
    "1 + 2",
    "returns ** 2",
    "returns if True else price",
    "",
    "rank(" * 50 + "returns" + ")" * 50,
])
def test_rejected_expressions(bad):
    with pytest.raises(ExpressionError):
        compile_expression(bad)


def test_canonical_form_dedupes_whitespace():
    a = compile_expression("rank( momentum(252,21) )")
    b = compile_expression("rank(momentum(252, 21))")
    assert a.canonical == b.canonical


def test_expression_values_match_pandas():
    c = compile_expression("ts_mean(returns, 10) - delay(returns, 2)")
    expected = R.rolling(10, min_periods=10).mean() - R.shift(2)
    pd.testing.assert_frame_equal(c(R), expected)


# ---------------------------------------------------------------- gateway


@pytest.fixture
def gw():
    return ResearchGateway(R, holdout_frac=0.25, holdout_budget=2)


def test_agent_never_sees_holdout(gw):
    d = gw.describe()
    assert pd.Timestamp(d["in_sample_end"]) < gw.holdout_start
    out = gw.test_signal("rank(momentum(120, 5))", rationale="test")
    m = gw.ledger.returns_matrix()
    assert m.index.max() < gw.holdout_start
    assert out["status"] == "ok"


def test_every_compiled_test_is_a_trial(gw):
    gw.test_signal("rank(momentum(120, 5))", rationale="a")
    gw.test_signal("rank(-ts_sum(returns, 5))", rationale="b")
    gw.test_signal("not valid ((", rationale="c")
    assert gw.ledger.n_trials() == 2  # malformed input measured nothing
    gw.test_signal("rank(momentum( 120 ,5 ))", rationale="same idea again")
    assert gw.ledger.n_trials() == 2  # duplicate spec is not a new trial


def test_response_always_has_trial_adjusted_context(gw):
    for e in ("rank(momentum(120, 5))", "rank(returns)", "rank(ts_std(returns, 20))"):
        out = gw.test_signal(e, rationale="x")
        assert "trial_adjusted" in out
        assert {"trials_so_far", "hurdle_sharpe_annual", "deflated_sharpe_ratio"} <= set(out["trial_adjusted"])
    assert out["trial_adjusted"]["trials_so_far"] == 3


def test_parent_must_exist_and_lineage_works(gw):
    assert gw.test_signal("rank(returns)", rationale="x", parent_id="h_nope")["status"] == "rejected"
    a = gw.test_signal("rank(momentum(120, 5))", rationale="root")["hypothesis_id"]
    b = gw.test_signal("rank(momentum(60, 5))", rationale="shorter", parent_id=a)["hypothesis_id"]
    lin = gw.lineage(b)
    assert [x["hypothesis_id"] for x in lin["lineage"]] == [a, b]


def test_holdout_budget_is_enforced_and_logged(gw):
    a = gw.test_signal("rank(momentum(120, 5))", rationale="x")["hypothesis_id"]
    first = gw.evaluate_holdout(a)
    assert first["status"] == "ok" and first["holdout_evaluations_left"] == 1
    assert gw.evaluate_holdout(a)["status"] == "ok"
    assert gw.evaluate_holdout(a)["status"] == "refused"
    notes = gw.ledger.entries().query("kind == 'note'")
    assert len(notes) == 2
    assert gw.verify()["chain_ok"]


def test_budget_survives_restart(tmp_path):
    p = tmp_path / "l.db"
    g1 = ResearchGateway(R, p, holdout_budget=1)
    a = g1.test_signal("rank(momentum(120, 5))", rationale="x")["hypothesis_id"]
    assert g1.evaluate_holdout(a)["status"] == "ok"
    g1.ledger.close()
    g2 = ResearchGateway(R, p, holdout_budget=1)
    assert g2.evaluate_holdout(a)["status"] == "refused"


def test_leaderboard_reports_deflated_numbers(gw):
    for lb in (252, 126, 63, 21):
        gw.test_signal(f"rank(momentum({lb}, 5))", rationale="sweep")
    lb = gw.leaderboard(top=3)
    assert lb["n_trials"] == 4 and len(lb["top"]) == 3
    assert all("deflated_sharpe_ratio" in r for r in lb["top"])


# ---------------------------------------------------------------- MCP


def test_mcp_tools_in_process():
    from alphaledger.gateway.mcp_server import build_server

    server = build_server(ResearchGateway(R))

    async def go():
        names = [t.name for t in await server.list_tools()]
        res = await server.call_tool("test_signal", {"expression": "rank(momentum(120, 5))", "rationale": "x"})
        return names, json.loads(res.content[0].text)

    names, out = asyncio.run(go())
    assert set(names) == {"describe_data", "expression_language", "test_signal", "leaderboard",
                          "lineage", "verify_ledger", "evaluate_holdout"}
    assert out["status"] == "ok" and "trial_adjusted" in out


def test_mcp_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "alphaledger.gateway.mcp_server", "--ledger", str(tmp_path / "s.db"), "--synthetic"],
    )

    async def go():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool("test_signal", {"expression": "rank(momentum(252, 21))",
                                                        "rationale": "classic momentum"})
                ver = await s.call_tool("verify_ledger", {})
                return json.loads(res.content[0].text), json.loads(ver.content[0].text)

    out, ver = asyncio.run(asyncio.wait_for(go(), timeout=120))
    assert out["status"] == "ok"
    assert ver["chain_ok"] and ver["entries"] == 2


def test_flat_signal_is_degenerate_not_a_crash(gw):
    # Regression: an expression that cancels to zero used to crash the gateway.
    out = gw.test_signal("0.5 * rank(momentum(120, 5)) - 0.5 * rank(momentum(120, 5))", rationale="x")
    assert out["status"] == "degenerate"
    assert gw.ledger.n_trials() == 1
    good = gw.test_signal("rank(momentum(120, 5))", rationale="y")
    assert good["status"] == "ok" and good["trial_adjusted"]["trials_so_far"] == 2
    assert gw.leaderboard()["n_trials"] == 2
