"""Real-data study: blind search against pre-registered hypotheses on 49 US industries.

Data: Kenneth French's 49 value-weighted industry portfolios, daily, from
1990 to the latest month in the library. Industry portfolios are rebuilt
every June from the stocks alive at the time, so the universe has no
survivorship bias, and the returns are free.

Both agents use the same gateway (hidden final 20% of dates, next-day
execution, 5 bps costs, leak audit on) and the same selection rule: take the
best in-sample trial, then open the holdout once.

* Blind agent: 100 random expressions from the same grammar as the
  synthetic controlled study.
* Pre-registered agent: the same 6 expressions as the controlled study,
  fixed before this data was downloaded.

Unlike the synthetic studies, nobody knows the true edge here. Industry
momentum is documented (Moskowitz and Grinblatt, 1999) but was published long
before this sample starts, and published anomalies tend to weaken afterwards.

Usage:
    python examples/real_data_study.py        # downloads about 4 MB once, then about 15 minutes
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controlled_study import PRE_REGISTERED, blind_proposals  # noqa: E402

from alphaledger.data.sources import KEN_FRENCH_INDUSTRY_URL, parse_ken_french_industries  # noqa: E402
from alphaledger.gateway import ResearchGateway  # noqa: E402
from alphaledger.report import build_report  # noqa: E402
from alphaledger.stats import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache" / "49_Industry_Portfolios_daily_CSV.zip"
OUT = ROOT / "case_study"


def load() -> pd.DataFrame:
    """Cache the original zip, not a re-saved table, so every run parses the
    same bytes (a CSV round trip changes the last bits of some floats)."""
    import io
    import urllib.request
    import zipfile

    if not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(KEN_FRENCH_INDUSTRY_URL.format(n=49), timeout=120) as r:
            CACHE.write_bytes(r.read())
    with zipfile.ZipFile(io.BytesIO(CACHE.read_bytes())) as z:
        return parse_ken_french_industries(z.read(z.namelist()[0]).decode("latin-1"))


def run(agent: str, returns: pd.DataFrame, seed: int = 2026) -> dict:
    gw = ResearchGateway(returns, holdout_frac=0.2, holdout_budget=1, cost_bps=5,
                         data_label="Ken French 49 industries, value weighted, daily", survivor_free=True)
    if agent == "blind":
        for expr, q, freq in blind_proposals(seed, 100):
            if gw.ledger.n_trials() >= 100:
                break
            gw.test_signal(expr, q=q, rebalance_freq=freq, rationale="blind")
    else:
        for expr in PRE_REGISTERED:
            gw.test_signal(expr, q=0.2, rebalance_freq="M", rationale="pre-registered")
    ev = evaluate(gw.ledger)
    s = ev.summary()
    best_id = ev.table.index[0]
    best = ev.table.loc[best_id]
    spec = gw.ledger.hypotheses().set_index("hypothesis_id").loc[best_id, "spec"]
    leaks = int((gw.ledger.trials()["status"] == "leak").sum())
    ho = gw.evaluate_holdout(best_id)
    OUT.mkdir(exist_ok=True)
    (OUT / f"report_real_{agent}.html").write_text(
        build_report(gw.ledger, f"{agent.capitalize()} agent on 49 US industries (real data)"))
    top = ev.table.head(5)
    specs = gw.ledger.hypotheses().set_index("hypothesis_id")["spec"]
    return {
        "agent": agent,
        "in_sample": f"{returns.index[0].date()} to {gw._in_sample.index[-1].date()}",
        "holdout": f"{gw.holdout_start.date()} to {returns.index[-1].date()}",
        "trials": ev.n_trials,
        "effective_trials": round(ev.n_eff, 1),
        "leak_audit_failures": leaks,
        "raw_significant_one_sided": s["n_significant_raw"],
        "significant_after_bh": s["n_significant_bh"],
        "pick": spec["expression"],
        "pick_q": spec["q"],
        "pick_rebalance": spec["rebalance"],
        "pick_in_sample_sharpe": round(float(best["sr_annual"]), 3),
        "pick_dsr_effective": round(float(best["dsr_effective"]), 4),
        "pick_dsr_analytic": round(float(best["dsr_analytic"]), 4),
        "pick_dsr_trial_spread": round(float(best["dsr"]), 4),
        "hurdle_sharpe_effective": round(s["hurdle_sharpe_annual_effective"], 3),
        "certified": bool(best["dsr_effective"] >= 0.95),
        "pbo": None if s["pbo"] is None else round(s["pbo"], 3),
        "pick_holdout_sharpe": ho["holdout_sharpe"],
        "pick_holdout_days": ho["holdout_days"],
        "top5": [{"expression": specs[h]["expression"], "q": specs[h]["q"], "rebalance": specs[h]["rebalance"],
                  "in_sample_sharpe": round(float(r.sr_annual), 3),
                  "dsr_effective": round(float(r.dsr_effective), 4)} for h, r in top.iterrows()],
    }


def main() -> None:
    data = load().loc["1990-01-01":]
    results = [run(agent, data) for agent in ("preregistered", "blind")]
    (OUT / "real_data_study.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
