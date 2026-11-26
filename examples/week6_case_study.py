"""Week 6 case study: a blind hypothesis-spamming agent, with and without alphaledger.

The agent proposes 200 random signal expressions from a grammar of common
quant building blocks (momentum, reversal, volatility, trend z-scores and
pairwise blends), as an LLM agent told to find alpha might. Every proposal
goes through the research gateway: hidden holdout, leak audit, ledger.

Two markets, both synthetic so the truth is known:
* noise: no signal has any edge. Anything that looks good is luck.
* planted: momentum-type signals have a real edge, everything else is noise.

For each market we compare what a naive researcher would conclude (pick the
best in-sample Sharpe, count raw p < 0.05) with what alphaledger reports, and
then open the hidden holdout for the naive pick.

Usage:
    python examples/week6_case_study.py            # writes case_study/results.json
    python examples/week6_case_study.py --trials 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from alphaledger.gateway import ResearchGateway
from alphaledger.report import build_report
from alphaledger.stats import evaluate
from alphaledger.synthetic import make_market

OUT = Path(__file__).resolve().parent.parent / "case_study"


def atom(rng: random.Random) -> str:
    kind = rng.choice(["mom", "rev", "vol", "trend", "range"])
    if kind == "mom":
        lb = rng.choice([42, 63, 126, 189, 252])
        sk = rng.choice([1, 5, 21])
        return f"rank(momentum({lb}, {sk}))"
    if kind == "rev":
        return f"rank(-ts_sum(returns, {rng.choice([1, 3, 5, 10, 21])}))"
    if kind == "vol":
        return f"rank(-ts_std(returns, {rng.choice([10, 21, 63, 126])}))"
    if kind == "trend":
        return f"rank(ts_zscore(price, {rng.choice([21, 63, 126, 252])}))"
    n = rng.choice([21, 63])
    return f"rank((price - ts_min(price, {n})) / (ts_max(price, {n}) - ts_min(price, {n})))"


def propose(rng: random.Random) -> tuple[str, float, str]:
    if rng.random() < 0.5:
        expr = atom(rng)
    else:
        a, b = atom(rng), atom(rng)
        w = rng.choice([0.3, 0.5, 0.7])
        sign = rng.choice(["+", "-"])
        expr = f"{w} * {a} {sign} {round(1 - w, 1)} * {b}"
    return expr, rng.choice([0.1, 0.2, 0.3]), rng.choice(["D", "W", "M"])


def run_market(name: str, returns, n_trials: int, seed: int) -> dict:
    gw = ResearchGateway(returns, holdout_frac=0.2, holdout_budget=1, cost_bps=5,
                         data_label=f"synthetic {name}")
    rng = random.Random(seed)
    tested = 0
    leaks = 0
    while gw.ledger.n_trials() < n_trials and tested < n_trials * 3:
        expr, q, freq = propose(rng)
        out = gw.test_signal(expr, q=q, rebalance_freq=freq, rationale="blind sweep")
        tested += 1
        leaks += out.get("status") == "leak"

    ev = evaluate(gw.ledger)
    s = ev.summary()
    tab = ev.table
    best_id = tab.index[0]
    spec = gw.ledger.hypotheses().set_index("hypothesis_id").loc[best_id, "spec"]
    ho = gw.evaluate_holdout(best_id)
    OUT.mkdir(exist_ok=True)
    (OUT / f"report_{name}.html").write_text(
        build_report(gw.ledger, f"Blind sweep agent, {n_trials} trials, synthetic {name} market"))
    return {
        "market": name,
        "proposals": tested,
        "trials": s["n_trials"],
        "leak_audit_failures": int(leaks),
        "naive": {
            "best_expression": spec["expression"],
            "best_q": spec["q"],
            "best_rebalance": spec["rebalance"],
            "best_in_sample_sharpe": round(s["best_naive_sharpe"], 3),
            "significant_at_5pct_raw_one_sided": s["n_significant_raw"],
        },
        "alphaledger": {
            "effective_trials": round(s["n_effective_trials"], 1),
            "hurdle_sharpe_effective": round(s["hurdle_sharpe_annual_effective"], 3),
            "best_dsr_effective": round(s["best_dsr_effective"], 6),
            "pass_dsr_effective_95": s["n_pass_dsr_effective_95"],
            "hurdle_sharpe": round(s["hurdle_sharpe_annual"], 3),
            "hurdle_sharpe_analytic": round(s["hurdle_sharpe_annual_analytic"], 3),
            "best_dsr": round(s["best_dsr"], 6),
            "best_dsr_analytic": round(s["best_dsr_analytic"], 6),
            "pass_dsr_analytic_95": s["n_pass_dsr_analytic_95"],
            "significant_after_bh": s["n_significant_bh"],
            "pass_dsr_95": s["n_pass_dsr_95"],
            "pbo": None if s["pbo"] is None else round(s["pbo"], 3),
            "best_haircut_sharpe_holm": round(float(tab.iloc[0]["sr_haircut_holm"]), 3),
        },
        "holdout_of_naive_pick": {
            "sharpe": ho["holdout_sharpe"],
            "psr_vs_zero": ho["holdout_psr_vs_zero"],
            "days": ho["holdout_days"],
        },
        "ledger_verifies": gw.verify()["chain_ok"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    markets = {
        "noise": make_market(n_assets=100, n_days=252 * 8, drift_vol=0.0, seed=101),
        "planted": make_market(n_assets=100, n_days=252 * 8, drift_vol=0.0006, seed=102),
    }
    results = [run_market(k, v, args.trials, args.seed) for k, v in markets.items()]
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
