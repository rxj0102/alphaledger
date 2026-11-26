"""Research gateway: the only door between an agent and the data.

* The agent sees only the in-sample period. The final ``holdout_frac`` of
  dates is hidden and can be opened a limited number of times, each logged.
* Every signal is an expression in the safe language (``expr.py``), compiled,
  leak-audited, backtested and recorded through ``run_logged``. There is no
  way to get a backtest number without creating a ledger entry.
* Every response reports trial-adjusted context (trial count, hurdle Sharpe,
  deflated Sharpe) next to the naive numbers, so the agent is never shown a
  raw Sharpe on its own.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..backtest import run_backtest
from ..ledger import Ledger, LedgerError, run_logged
from ..signals import quantile_long_short, rebalance
from ..stats import evaluate, psr, moments
from ..stats.sharpe import expected_max_sr
from .expr import FUNCTIONS, VARIABLES, ExpressionError, compile_expression

REBALANCE = {"D": None, "W": "W", "M": "M"}


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


class ResearchGateway:
    def __init__(self, returns: pd.DataFrame, ledger: Ledger | str | Path = ":memory:",
                 holdout_frac: float = 0.2, holdout_budget: int = 3, cost_bps: float = 5.0,
                 audit_leaks: bool = True, data_label: str = "unlabelled"):
        if not 0 < holdout_frac < 0.5:
            raise ValueError("holdout_frac must be in (0, 0.5)")
        returns = returns.sort_index()
        cut = int(len(returns) * (1 - holdout_frac))
        self._in_sample = returns.iloc[:cut]
        self._full = returns
        self.holdout_start = returns.index[cut]
        self.holdout_budget = holdout_budget
        self.cost_bps = cost_bps
        self.audit_leaks = audit_leaks
        self.data_label = data_label
        self.ledger = ledger if isinstance(ledger, Ledger) else Ledger(ledger)

    # ------------------------------------------------------------ info
    def describe(self) -> dict:
        r = self._in_sample
        return {
            "data": self.data_label,
            "n_assets": int(r.shape[1]),
            "in_sample_start": str(r.index[0].date()),
            "in_sample_end": str(r.index[-1].date()),
            "n_days_in_sample": int(len(r)),
            "holdout": f"final {len(self._full) - len(r)} days are hidden; "
                       f"{self._holdout_remaining()} of {self.holdout_budget} holdout evaluations left",
            "variables": sorted(VARIABLES),
            "functions": FUNCTIONS,
            "portfolio": "quantile long-short, dollar neutral, gross 1.0, traded with a 1 day lag",
            "cost_bps": self.cost_bps,
            "trials_so_far": self.ledger.n_trials(),
        }

    # ------------------------------------------------------------ testing
    def _weights_fn(self, compiled, q: float, rebalance_freq: str):
        if rebalance_freq not in REBALANCE:
            raise ExpressionError("rebalance must be one of D, W, M")
        period = REBALANCE[rebalance_freq]

        def build(r):
            w = quantile_long_short(compiled(r), q)
            return rebalance(w, period) if period else w

        return build

    def _context(self, sr_daily: float, t_obs: int, skew: float, kurt: float) -> dict:
        n = max(self.ledger.n_trials(), 1)
        m = self.ledger.returns_matrix()
        sharpes = []
        for c in m.columns:
            try:
                sharpes.append(moments(m[c])[0])
            except ValueError:
                pass  # flat series carry no Sharpe
        var = float(pd.Series(sharpes).var(ddof=1)) if len(sharpes) > 1 else 0.0
        sr0 = expected_max_sr(n, var)
        sr0a = expected_max_sr(n, 1 / (t_obs - 1))
        return {
            "trials_so_far": n,
            "hurdle_sharpe_annual": _r(sr0 * math.sqrt(252), 3),
            "deflated_sharpe_ratio": _r(psr(sr_daily, t_obs, skew, kurt, sr0)),
            "hurdle_sharpe_annual_analytic": _r(sr0a * math.sqrt(252), 3),
            "deflated_sharpe_ratio_analytic": _r(psr(sr_daily, t_obs, skew, kurt, sr0a)),
            "note": ("Deflated Sharpe above 0.95 means the result likely survives the number of trials "
                     "run so far. The first version uses the spread of your own trial Sharpes (strict when "
                     "trials differ for systematic reasons such as costs); the analytic version assumes "
                     "independent zero-skill trials."),
        }

    def test_signal(self, expression: str, q: float = 0.2, rebalance_freq: str = "M",
                    rationale: str = "", parent_id: str | None = None) -> dict:
        try:
            compiled = compile_expression(expression)
        except ExpressionError as exc:
            # Malformed expressions are not trials: nothing was measured.
            return {"status": "rejected", "error": str(exc)}
        if not 0 < q <= 0.5:
            return {"status": "rejected", "error": "q must be in (0, 0.5]"}
        if parent_id is not None:
            try:
                self.ledger.lineage(parent_id)
            except LedgerError as exc:
                return {"status": "rejected", "error": str(exc)}
        spec = {"expression": compiled.canonical, "q": q, "rebalance": rebalance_freq,
                "portfolio": "quantile_long_short", "lag": 1, "cost_bps": self.cost_bps,
                "data": self.data_label}
        hid, stats = run_logged(
            self.ledger, spec, self._weights_fn(compiled, q, rebalance_freq), self._in_sample,
            parent_id=parent_id, rationale=rationale, cost_bps=self.cost_bps,
            audit_leaks=self.audit_leaks, scores_fn=compiled,
        )
        trial = self.ledger.trials().set_index("hypothesis_id").loc[hid]
        out = {"hypothesis_id": hid, "status": trial["status"], "expression": compiled.canonical}
        if stats is None:
            out["error"] = "the signal raised an error while running; it still counts as a trial"
            return out
        audit = stats.get("leak_audit", {})
        out["naive"] = {k: _r(stats.get(k)) for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown",
                                                      "t_stat", "avg_daily_turnover", "n_obs")}
        out["leak_audit"] = {"passed": audit.get("passed", True),
                             "findings": [f"{f['check']}: {f['detail']}" for f in audit.get("findings", [])]}
        dup = trial["duplicate_of"] if isinstance(trial["duplicate_of"], str) else None
        if dup:
            out["duplicate_of"] = dup
            out["note"] = "identical to an earlier test; logged but not counted as a new trial"
        if trial["status"] == "ok":
            col = dup or hid
            r = self.ledger.returns_matrix()[col].dropna()
            sr, skew, kurt, t_obs = moments(r)
            out["trial_adjusted"] = self._context(sr, t_obs, skew, kurt)
        return out

    # ------------------------------------------------------------ ledger views
    def leaderboard(self, top: int = 10) -> dict:
        try:
            ev = evaluate(self.ledger)
        except ValueError as exc:
            return {"error": str(exc)}
        tab = ev.table.head(top)
        specs = self.ledger.hypotheses().set_index("hypothesis_id")["spec"]
        rows = [{
            "hypothesis_id": hid,
            "expression": specs[hid].get("expression"),
            "naive_sharpe": _r(row.sr_annual, 3),
            "deflated_sharpe_ratio": _r(row.dsr),
            "deflated_sharpe_ratio_analytic": _r(row.dsr_analytic),
            "significant_after_fdr": bool(row.sig_bh),
            "haircut_sharpe": _r(row.sr_haircut_holm, 3),
        } for hid, row in tab.iterrows()]
        s = ev.summary()
        return {
            "n_trials": s["n_trials"],
            "hurdle_sharpe_annual": _r(s["hurdle_sharpe_annual"], 3),
            "hurdle_sharpe_annual_analytic": _r(s["hurdle_sharpe_annual_analytic"], 3),
            "probability_of_backtest_overfitting": _r(s["pbo"], 3),
            "n_significant_after_fdr": s["n_significant_bh"],
            "top": rows,
        }

    def lineage(self, hypothesis_id: str) -> dict:
        try:
            chain = self.ledger.lineage(hypothesis_id)
        except LedgerError as exc:
            return {"error": str(exc)}
        specs = self.ledger.hypotheses().set_index("hypothesis_id")["spec"]
        return {"lineage": [{"hypothesis_id": h, "expression": specs[h].get("expression")} for h in chain],
                "children": self.ledger.children(hypothesis_id)}

    def verify(self) -> dict:
        rep = self.ledger.verify()
        return {"chain_ok": rep.ok, "entries": rep.n_entries, "head_hash": self.ledger.head_hash(),
                "first_bad_entry": rep.first_bad_seq, "reason": rep.reason}

    # ------------------------------------------------------------ holdout
    def _holdout_uses(self) -> int:
        e = self.ledger.entries()
        return int(((e["kind"] == "note") & e["payload"].str.contains('"holdout_eval"')).sum())

    def _holdout_remaining(self) -> int:
        return max(self.holdout_budget - self._holdout_uses(), 0)

    def evaluate_holdout(self, hypothesis_id: str) -> dict:
        """Open the hidden period for one tested hypothesis. Limited and logged."""
        if self._holdout_remaining() <= 0:
            return {"status": "refused", "error": "holdout budget used up; the holdout is no longer clean"}
        h = self.ledger.hypotheses().set_index("hypothesis_id")
        if hypothesis_id not in h.index:
            return {"status": "refused", "error": f"unknown hypothesis {hypothesis_id}"}
        trial = self.ledger.trials().set_index("hypothesis_id").loc[hypothesis_id]
        if trial["status"] != "ok":
            return {"status": "refused", "error": f"hypothesis status is {trial['status']}, not ok"}
        spec = h.loc[hypothesis_id, "spec"]
        compiled = compile_expression(spec["expression"])
        build = self._weights_fn(compiled, spec["q"], spec["rebalance"])
        res = run_backtest(build(self._full), self._full, lag=1, cost_bps=self.cost_bps)
        ho = res.returns.loc[res.returns.index >= self.holdout_start]
        sr, skew, kurt, t_obs = moments(ho)
        result = {
            "holdout_start": str(self.holdout_start.date()),
            "holdout_days": int(t_obs),
            "holdout_sharpe": _r(sr * math.sqrt(252), 3),
            "holdout_psr_vs_zero": _r(psr(sr, t_obs, skew, kurt, 0.0)),
        }
        self.ledger.note("holdout_eval", hypothesis_id=hypothesis_id, data={"holdout_eval": result})
        result["holdout_evaluations_left"] = self._holdout_remaining()
        return {"status": "ok", "hypothesis_id": hypothesis_id, **result}
