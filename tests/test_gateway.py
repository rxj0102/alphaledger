import asyncio
import json
import sys

import numpy as np
import pandas as pd
import pytest

from alphaledger.gateway import ExpressionError, ResearchGateway, compile_expression
from alphaledger.gateway.expr import FUNCTIONS
from alphaledger.leakage import perturbation_check, truncation_check
from alphaledger.signals import quantile_long_short
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
        assert {"trials_so_far", "effective_independent_trials", "deflated_sharpe_ratio_effective",
                "deflated_sharpe_ratio_analytic", "deflated_sharpe_ratio_trial_spread"} <= set(out["trial_adjusted"])
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
    assert sum("holdout_eval" in n for n in gw.ledger.notes()) == 2
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
    assert all("deflated_sharpe_ratio_effective" in r for r in lb["top"])


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
    assert ver["chain_ok"] and ver["entries"] == 3  # config note, hypothesis, result


def test_flat_signal_is_degenerate_not_a_crash(gw):
    # Regression: an expression that cancels to zero used to crash the gateway.
    out = gw.test_signal("0.5 * rank(momentum(120, 5)) - 0.5 * rank(momentum(120, 5))", rationale="x")
    assert out["status"] == "degenerate"
    assert gw.ledger.n_trials() == 1
    good = gw.test_signal("rank(momentum(120, 5))", rationale="y")
    assert good["status"] == "ok" and good["trial_adjusted"]["trials_so_far"] == 2
    assert gw.leaderboard()["n_trials"] == 2



# ---------------------------------------------------------------- locking down trial accounting


def test_ledger_bound_to_config(tmp_path):
    p = tmp_path / "l.db"
    g = ResearchGateway(R, p, holdout_budget=1)
    g.test_signal("rank(momentum(120, 5))", rationale="x")
    g.ledger.close()
    with pytest.raises(Exception, match="different dataset or configuration"):
        ResearchGateway(R, p, holdout_budget=5)  # trying to buy more holdout looks
    with pytest.raises(Exception, match="different dataset or configuration"):
        ResearchGateway(R.iloc[:-10], p, holdout_budget=1)
    ResearchGateway(R, p, holdout_budget=1)  # same setup reopens fine


def test_registry_blocks_fresh_ledger_reset(tmp_path):
    reg = tmp_path / "registry.jsonl"
    g1 = ResearchGateway(R, tmp_path / "a.db", holdout_budget=1, registry_path=reg)
    h = g1.test_signal("rank(momentum(120, 5))", rationale="x")["hypothesis_id"]
    assert g1.evaluate_holdout(h)["status"] == "ok"
    with pytest.raises(Exception, match="already bound"):
        ResearchGateway(R, tmp_path / "b.db", holdout_budget=1, registry_path=reg)
    g1.ledger.close()
    g3 = ResearchGateway(R, tmp_path / "a.db", holdout_budget=1, registry_path=reg)
    assert g3.evaluate_holdout(h)["status"] == "refused"


def test_signed_and_anchored_gateway(tmp_path):
    anchors = tmp_path / "anchors.jsonl"
    g = ResearchGateway(R, tmp_path / "s.db", key="k", anchor_path=anchors)
    g.test_signal("rank(momentum(120, 5))", rationale="x")
    v = g.verify()
    assert v["chain_ok"] and v["signed"] and v["anchors_ok"]


def test_note_text_cannot_spend_or_refund_budget(gw):
    gw.ledger.note("holdout_eval")  # text alone is not a holdout evaluation
    assert gw._holdout_remaining() == gw.holdout_budget


def test_identical_portfolio_flagged(gw):
    a = gw.test_signal("rank(momentum(120, 5))", rationale="x")
    b = gw.test_signal("rank(rank(momentum(120, 5)))", rationale="same thing")
    assert b["same_trial_as"] == a["hypothesis_id"]
    assert gw.ledger.n_trials() == 1


def test_default_execution_is_next_day(gw):
    from alphaledger.backtest import run_backtest

    assert gw.execution_lag == 2
    out = gw.test_signal("rank(momentum(120, 5))", rationale="x", rebalance_freq="D")
    w = quantile_long_short(compile_expression("rank(momentum(120, 5))")(gw._in_sample))
    lag2 = run_backtest(w, gw._in_sample, lag=2, cost_bps=5).stats["sharpe"]
    assert out["naive"]["sharpe"] == pytest.approx(lag2, abs=1e-4)


def test_survivorship_reported_once_not_per_trial():
    gw = ResearchGateway(make_market(n_assets=40, n_days=252 * 5, seed=3))
    assert gw.describe()["data_warnings"]
    out = gw.test_signal("rank(momentum(120, 5))", rationale="x")
    assert not any(f.startswith("survivorship") for f in out["leak_audit"]["findings"])


# ---------------------------------------------------------------- point-in-time variables


def _macro_store():
    from alphaledger.data import PITStore

    # A monthly macro series released 25 days after the month it describes.
    periods = pd.date_range(R.index[0] - pd.Timedelta(days=60), R.index[-1], freq="ME")
    rng = np.random.default_rng(0)
    s = PITStore()
    s.add(pd.DataFrame({"entity": "US", "field": "IP", "period": periods,
                        "available_at": periods + pd.Timedelta(days=25),
                        "value": rng.normal(0, 1, len(periods))}))
    return s


def test_pit_variable_only_uses_released_values():
    store = _macro_store()
    g = ResearchGateway(R, pit=store, pit_fields={"ip": "IP"})
    panel = g._extra_vars["ip"](g._in_sample)
    raw = store.to_frame()
    for d in g._in_sample.index[::50]:
        v = panel.loc[d].iloc[0]
        if pd.isna(v):
            continue
        released = raw[(raw["available_at"] <= d)].sort_values("period").iloc[-1]
        assert v == released["value"]


def test_pit_variable_in_expression_is_causal_and_logged():
    g = ResearchGateway(R, pit=_macro_store(), pit_fields={"ip": "IP"})
    assert "ip" in g.describe()["point_in_time_variables"]
    out = g.test_signal("rank(momentum(120, 5)) * sign(ip)", rationale="momentum only in expansions")
    assert out["status"] == "ok" and out["leak_audit"]["passed"]


def test_pit_fields_are_part_of_the_binding(tmp_path):
    p = tmp_path / "l.db"
    ResearchGateway(R, p, pit=_macro_store(), pit_fields={"ip": "IP"}).ledger.close()
    with pytest.raises(Exception, match="different dataset or configuration"):
        ResearchGateway(R, p)


def test_mcp_locked_down_deployment_over_stdio(tmp_path, monkeypatch):
    import os

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    args = ["-m", "alphaledger.gateway.mcp_server", "--ledger", str(tmp_path / "s.db"), "--synthetic",
            "--registry", str(tmp_path / "reg.jsonl"), "--anchors", str(tmp_path / "anchors.jsonl")]
    env = dict(os.environ, ALPHALEDGER_KEY="operator-secret")
    params = StdioServerParameters(command=sys.executable, args=args, env=env)

    async def go():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                await s.call_tool("test_signal", {"expression": "rank(momentum(252, 21))", "rationale": "x"})
                return json.loads((await s.call_tool("verify_ledger", {})).content[0].text)

    ver = asyncio.run(asyncio.wait_for(go(), timeout=120))
    assert ver["chain_ok"] and ver["signed"] and ver["anchors_ok"]
    from alphaledger.ledger import Ledger

    assert not Ledger(tmp_path / "s.db").verify(require_signatures=True).ok  # no key, no proof
    assert Ledger(tmp_path / "s.db", key="operator-secret").verify().ok
