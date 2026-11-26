"""Controlled study: blind search vs pre-registered hypotheses on identical markets.

Both agents face exactly the same markets (same seeds), the same gateway
rules (hidden 20% holdout, next-day execution, 5 bps costs) and the same
selection rule (take the trial with the best in-sample result, then open the
holdout once).

* Blind agent: 100 random expressions from a grammar of common building
  blocks, as an LLM told to find alpha might produce.
* Pre-registered agent: 6 expressions written down before any data was seen,
  from documented anomalies (momentum, reversal, low volatility, trend) and
  one blend. The list is fixed below and identical for every market.

Markets: 20 seeds with a planted continuation edge and 20 seeds of pure noise.
On planted markets, certifying the pick is a true positive; on noise markets
it is a false positive. "Certified" means the pick's deflated Sharpe ratio
(effective trials) is at least 0.95.

The leak audit is switched off here for speed; every expression in both
lists comes from the causal expression language, whose operators are tested
for causality in the test suite.

Usage:
    python examples/controlled_study.py            # about 10 minutes, 80 runs
    python examples/controlled_study.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from alphaledger.gateway import ResearchGateway
from alphaledger.stats import evaluate
from alphaledger.synthetic import make_market

PRE_REGISTERED = [
    "rank(momentum(252, 21))",
    "rank(-ts_sum(returns, 5))",
    "rank(-ts_std(returns, 63))",
    "rank(momentum(126, 5))",
    "rank(ts_zscore(price, 252))",
    "0.5 * rank(momentum(126, 5)) + 0.5 * rank(ts_zscore(price, 252))",
]


def atom(rng: random.Random) -> str:
    kind = rng.choice(["mom", "rev", "vol", "trend", "range"])
    if kind == "mom":
        return f"rank(momentum({rng.choice([42, 63, 126, 189, 252])}, {rng.choice([1, 5, 21])}))"
    if kind == "rev":
        return f"rank(-ts_sum(returns, {rng.choice([1, 3, 5, 10, 21])}))"
    if kind == "vol":
        return f"rank(-ts_std(returns, {rng.choice([10, 21, 63, 126])}))"
    if kind == "trend":
        return f"rank(ts_zscore(price, {rng.choice([21, 63, 126, 252])}))"
    n = rng.choice([21, 63])
    return f"rank((price - ts_min(price, {n})) / (ts_max(price, {n}) - ts_min(price, {n})))"


def blind_proposals(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n * 3):
        if rng.random() < 0.5:
            expr = atom(rng)
        else:
            w = rng.choice([0.3, 0.5, 0.7])
            expr = f"{w} * {atom(rng)} {rng.choice(['+', '-'])} {round(1 - w, 1)} * {atom(rng)}"
        yield expr, rng.choice([0.1, 0.2, 0.3]), rng.choice(["D", "W", "M"])


def run(agent: str, returns, seed: int, n_blind: int) -> dict:
    gw = ResearchGateway(returns, holdout_frac=0.2, holdout_budget=1, cost_bps=5, audit_leaks=False)
    if agent == "blind":
        for expr, q, freq in blind_proposals(seed, n_blind):
            if gw.ledger.n_trials() >= n_blind:
                break
            gw.test_signal(expr, q=q, rebalance_freq=freq, rationale="blind")
    else:
        for expr in PRE_REGISTERED:
            gw.test_signal(expr, q=0.2, rebalance_freq="M", rationale="pre-registered")
    ev = evaluate(gw.ledger)
    best_id = ev.table.index[0]  # best in-sample naive Sharpe, same rule for both agents
    best = ev.table.loc[best_id]
    ho = gw.evaluate_holdout(best_id)
    return {
        "agent": agent,
        "seed": seed,
        "trials": ev.n_trials,
        "n_eff": round(ev.n_eff, 1),
        "pick_in_sample_sharpe": round(float(best["sr_annual"]), 3),
        "pick_dsr_effective": round(float(best["dsr_effective"]), 4),
        "certified": bool(best["dsr_effective"] >= 0.95),
        "any_bh_significant": bool(ev.table["sig_bh"].any()),
        "raw_significant": int((ev.table["p_value"] <= 0.05).sum()),
        "pick_holdout_sharpe": ho["holdout_sharpe"],
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for market in ("planted", "noise"):
        for agent in ("blind", "preregistered"):
            sel = [r for r in rows if r["market"] == market and r["agent"] == agent]
            if not sel:
                continue
            ho = np.array([r["pick_holdout_sharpe"] for r in sel], dtype=float)
            out[f"{market}/{agent}"] = {
                "runs": len(sel),
                "mean_trials": float(np.mean([r["trials"] for r in sel])),
                "mean_effective_trials": round(float(np.mean([r["n_eff"] for r in sel])), 1),
                "certified_rate": float(np.mean([r["certified"] for r in sel])),
                "any_bh_significant_rate": float(np.mean([r["any_bh_significant"] for r in sel])),
                "mean_raw_significant": float(np.mean([r["raw_significant"] for r in sel])),
                "mean_pick_in_sample_sharpe": round(float(np.mean([r["pick_in_sample_sharpe"] for r in sel])), 3),
                "mean_pick_holdout_sharpe": round(float(ho.mean()), 3),
                "pick_holdout_positive_rate": float(np.mean(ho > 0)),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--blind-trials", type=int, default=100)
    args = ap.parse_args()
    rows = []
    for i in range(args.seeds):
        for market, drift in (("planted", 0.0006), ("noise", 0.0)):
            seed = 5000 + i
            returns = make_market(n_assets=100, n_days=252 * 8, drift_vol=drift, seed=seed)
            for agent in ("blind", "preregistered"):
                r = run(agent, returns, seed, args.blind_trials)
                r["market"] = market
                rows.append(r)
                print(json.dumps(r), flush=True)
    summary = summarize(rows)
    out = Path(__file__).resolve().parent.parent / "case_study"
    out.mkdir(exist_ok=True)
    (out / "controlled_study.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
