"""Research gateway: the only door between an agent and the data.

* The agent sees only the in-sample period. The final ``holdout_frac`` of
  dates is hidden and can be opened a limited number of times, each logged.
* Every signal is an expression in the safe language (``expr.py``), compiled,
  leak-audited, backtested and recorded through ``run_logged``. There is no
  way to get a backtest number from the gateway without creating a ledger entry.
* Every response reports trial-adjusted context next to the naive numbers.

What the gateway can and cannot enforce
---------------------------------------
The gateway controls what passes through it. It cannot stop an agent that has
its own copy of the data from backtesting elsewhere; that has to be enforced
by deployment (the operator runs the server, holds the data and the signing
key, and the agent only gets the MCP connection). Inside that setup:

* the ledger is bound to the dataset and configuration on first use, so it
  cannot be reopened with a bigger holdout budget or a different holdout;
* with a ``registry`` file, a dataset can only ever be served from one ledger,
  so starting a fresh ledger does not reset the trial count or the budget;
* with a signing ``key`` the history cannot be rewritten without detection,
  and with an ``anchor_path`` every write is anchored outside the ledger.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import run_backtest
from ..data.pit import PITStore
from ..leakage import survivorship_check
from ..ledger import Ledger, LedgerError, data_fingerprint, run_logged
from ..ledger.store import canonical, sha256
from ..signals import quantile_long_short, rebalance
from ..stats import effective_trials, evaluate, moments, psr
from ..stats.sharpe import expected_max_sr
from .expr import FUNCTIONS, VARIABLES, ExpressionError, compile_expression

REBALANCE = {"D": None, "W": "W", "M": "M"}


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


SKETCH_K = 1024


def data_sketch(df: pd.DataFrame, k: int = SKETCH_K, decimals: int = 6) -> tuple[list[int], int]:
    """Bottom-k MinHash sketch of a returns panel's (date, asset, value) cells.

    Two panels that share most of their cells have similar sketches even if
    one drops a day, drops assets or adds a few rows, so a registry can
    recognise a trimmed or subset copy of a dataset it has already bound to a
    ledger. Values are rounded to ``decimals`` so float noise does not hide a
    copy; deliberate changes larger than that are not recognised.
    """
    long = df.stack().dropna()
    if long.empty:
        return [], 0
    dates = long.index.get_level_values(0)
    names = long.index.get_level_values(1).astype(str)
    cells = pd.DataFrame({
        "d": pd.DatetimeIndex(dates).strftime("%Y%m%d").astype("int64"),
        "a": pd.util.hash_array(names.to_numpy(dtype=object)),
        "v": long.to_numpy(dtype=float).round(decimals),
    })
    h = pd.util.hash_pandas_object(cells, index=False).to_numpy()
    u = np.unique(h)
    return [int(x) for x in u[:k]], int(len(u))


def sketch_containment(a: list[int], na: int, b: list[int], nb: int, k: int = SKETCH_K) -> float:
    """Estimated share of the smaller panel's cells that also appear in the other."""
    if not a or not b:
        return 0.0
    union_k = sorted(set(a) | set(b))[:k]
    sa, sb = set(a), set(b)
    j = sum(1 for x in union_k if x in sa and x in sb) / len(union_k)
    inter = j * (na + nb) / (1 + j)
    return float(inter / min(na, nb))


def pit_variable(pit: PITStore, field: str, lag_days: int = 0):
    """Expression variable backed by the point-in-time store.

    On each date it holds the latest value released on or before that date
    (minus ``lag_days``). A field stored for one entity (for example "US" for
    a macro series) is broadcast to every asset.
    """

    def fn(r: pd.DataFrame) -> pd.DataFrame:
        panel = pit.panel(field, r.index, lag_days=lag_days)
        if panel.shape[1] == 1 and panel.columns[0] not in r.columns:
            col = panel.iloc[:, 0].to_numpy()
            return pd.DataFrame({c: col for c in r.columns}, index=r.index)
        return panel.reindex(columns=r.columns)

    return fn


class ResearchGateway:
    def __init__(self, returns: pd.DataFrame, ledger: Ledger | str | Path = ":memory:",
                 holdout_frac: float = 0.2, holdout_budget: int = 3, cost_bps: float = 5.0,
                 audit_leaks: bool = True, data_label: str = "unlabelled", execution_lag: int = 2,
                 pit: PITStore | None = None, pit_fields: dict[str, str] | None = None,
                 pit_lag_days: int = 0, key: str | bytes | None = None,
                 anchor_path: str | Path | None = None, registry_path: str | Path | None = None,
                 survivor_free: bool = False, signer=None):
        if not 0 < holdout_frac < 0.5:
            raise ValueError("holdout_frac must be in (0, 0.5)")
        if execution_lag < 1:
            raise ValueError("execution_lag must be at least 1")
        returns = returns.sort_index()
        cut = int(len(returns) * (1 - holdout_frac))
        self._in_sample = returns.iloc[:cut]
        self._full = returns
        self.holdout_start = returns.index[cut]
        self.holdout_budget = holdout_budget
        self.cost_bps = cost_bps
        self.audit_leaks = audit_leaks
        self.data_label = data_label
        self.execution_lag = execution_lag
        self.anchor_path = anchor_path

        self._extra_vars = {}
        if pit_fields:
            if pit is None:
                raise ValueError("pit_fields given without a pit store")
            self._extra_vars = {name: pit_variable(pit, f, pit_lag_days) for name, f in pit_fields.items()}

        self.config = {
            "data_hash": data_fingerprint(returns),
            "holdout_start": str(self.holdout_start.date()),
            "holdout_budget": holdout_budget,
            "cost_bps": cost_bps,
            "execution_lag": execution_lag,
            "pit_fields": dict(sorted((pit_fields or {}).items())),
            "pit_lag_days": pit_lag_days,
            "pit_hash": data_fingerprint(pit.to_frame()) if pit is not None and pit_fields else None,
            "survivor_free": survivor_free,
        }
        self.name = "gw_" + sha256(canonical(self.config))[:16]
        if isinstance(ledger, Ledger):
            self.ledger = ledger
        else:
            self.ledger = Ledger(ledger, key=key, signer=signer)
        if registry_path is not None:
            self._bind_registry(registry_path)
        self._bind_config()
        # The survivorship heuristic cannot tell a survivor-free universe that
        # never delists (industry portfolios, indices) from today's survivors,
        # so the operator can declare it; the declaration is part of the binding.
        self._data_warnings = [] if survivor_free else [
            f.detail for f in [survivorship_check(self._in_sample)] if f]
        self.survivor_free = survivor_free

    # ------------------------------------------------------------ binding
    def _bind_config(self) -> None:
        existing = [n["gateway_config"] for n in self.ledger.notes() if "gateway_config" in n]
        if existing:
            if existing[0] != self.config:
                raise LedgerError(
                    "this ledger was created for a different dataset or configuration "
                    "(data, holdout, budget, costs, lag or data fields); refusing to reuse it"
                )
            return
        if len(self.ledger.hypotheses()) > 0:
            raise LedgerError("ledger already holds trials but no gateway configuration; refusing to adopt it")
        self.ledger.note("gateway_config", data={"gateway_config": self.config})
        self._anchor()

    def _bind_registry(self, registry_path, overlap: float = 0.5) -> None:
        """Bind this dataset to this ledger, refusing a second ledger for the same data.

        A dataset counts as the same if its exact hash matches, or if at least
        ``overlap`` of the smaller panel's cells appear in a registered one, so
        trimming a day or dropping assets does not buy a fresh ledger.
        """
        if self.ledger.path == ":memory:":
            raise ValueError("a registry needs a ledger file, not an in-memory ledger")
        me = os.path.abspath(self.ledger.path)
        key = self.config["data_hash"]
        sketch, n_cells = data_sketch(self._full)
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    same = rec["data_hash"] == key
                    if not same and rec.get("sketch"):
                        c = sketch_containment(sketch, n_cells, rec["sketch"], rec["n_cells"])
                        same = c >= overlap
                    if same:
                        if rec["ledger"] != me:
                            raise LedgerError(
                                f"this dataset (or most of it) is already bound to the ledger at "
                                f"{rec['ledger']}; starting a new ledger would reset the trial count "
                                "and holdout budget. Use that ledger."
                            )
                        if rec["data_hash"] == key:
                            return
        with open(registry_path, "a") as f:
            f.write(canonical({"data_hash": key, "ledger": me, "label": self.data_label,
                               "sketch": sketch, "n_cells": n_cells}) + "\n")

    def _anchor(self) -> None:
        if self.anchor_path is not None:
            self.ledger.anchor(self.anchor_path, self.name)

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
            "variables": sorted(VARIABLES) + sorted(self._extra_vars),
            "point_in_time_variables": sorted(self._extra_vars),
            "functions": FUNCTIONS,
            "portfolio": (f"quantile long-short, dollar neutral, gross 1.0; a signal computed from the "
                          f"close of day t is traded at the close of day t+{self.execution_lag - 1}"),
            "cost_bps": self.cost_bps,
            "data_warnings": self._data_warnings,
            "universe": ("declared survivor-free by the operator" if self.survivor_free
                         else "not declared survivor-free"),
            "trials_so_far": self.ledger.n_trials(),
            "ledger_signed": self.ledger.signed,
        }

    # ------------------------------------------------------------ testing
    def _compile(self, expression: str):
        return compile_expression(expression, self._extra_vars)

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
        n_eff = effective_trials(m) + (n - m.shape[1]) if m.shape[1] else float(n)
        n_eff = min(max(n_eff, 1.0), float(n))
        sr0 = expected_max_sr(n, var)
        sr0a = expected_max_sr(n, 1 / (t_obs - 1))
        sr0e = expected_max_sr(max(int(round(n_eff)), 1), 1 / (t_obs - 1))
        return {
            "trials_so_far": n,
            "effective_independent_trials": _r(n_eff, 1),
            "deflated_sharpe_ratio_effective": _r(psr(sr_daily, t_obs, skew, kurt, sr0e)),
            "hurdle_sharpe_annual_effective": _r(sr0e * math.sqrt(252), 3),
            "deflated_sharpe_ratio_analytic": _r(psr(sr_daily, t_obs, skew, kurt, sr0a)),
            "hurdle_sharpe_annual_analytic": _r(sr0a * math.sqrt(252), 3),
            "deflated_sharpe_ratio_trial_spread": _r(psr(sr_daily, t_obs, skew, kurt, sr0)),
            "hurdle_sharpe_annual_trial_spread": _r(sr0 * math.sqrt(252), 3),
            "note": ("Judge results by deflated_sharpe_ratio_effective: 0.95 or more means the result "
                     "likely survives the search so far, counting correlated trials once. The analytic "
                     "version treats every trial as independent; the trial-spread version follows the "
                     "original paper and is very strict when trials differ for systematic reasons."),
        }

    MAX_RATIONALE = 4000

    def _first_holdout_seq(self) -> int | None:
        seqs = [n["seq"] for n in self.ledger.notes() if isinstance(n.get("holdout_eval"), dict)]
        return min(seqs) if seqs else None

    def test_signal(self, expression: str, q: float = 0.2, rebalance_freq: str = "M",
                    rationale: str = "", parent_id: str | None = None) -> dict:
        if len(rationale or "") > self.MAX_RATIONALE:
            return {"status": "rejected", "error": f"rationale longer than {self.MAX_RATIONALE} characters"}
        try:
            compiled = self._compile(expression)
        except ExpressionError as exc:
            # Malformed expressions are not trials: nothing was measured.
            return {"status": "rejected", "error": str(exc)}
        if not 0 < q <= 0.5:
            return {"status": "rejected", "error": "q must be in (0, 0.5]"}
        if rebalance_freq not in REBALANCE:
            return {"status": "rejected", "error": "rebalance must be one of D, W, M"}
        if parent_id is not None:
            try:
                self.ledger.lineage(parent_id)
            except LedgerError as exc:
                return {"status": "rejected", "error": str(exc)}
        spec = {"expression": compiled.canonical, "q": q, "rebalance": rebalance_freq,
                "portfolio": "quantile_long_short", "lag": self.execution_lag,
                "cost_bps": self.cost_bps, "data": self.data_label}
        hid, stats = run_logged(
            self.ledger, spec, self._weights_fn(compiled, q, rebalance_freq), self._in_sample,
            parent_id=parent_id, rationale=rationale, lag=self.execution_lag, cost_bps=self.cost_bps,
            audit_leaks=self.audit_leaks, scores_fn=compiled,
        )
        self._anchor()
        trial = self.ledger.trials().set_index("hypothesis_id").loc[hid]
        out = {"hypothesis_id": hid, "status": trial["status"], "expression": compiled.canonical}
        if self._first_holdout_seq() is not None:
            out["after_holdout_look"] = True
            out["holdout_warning"] = ("this test was run after the holdout was opened, so it may be shaped by "
                                      "what the holdout showed; its holdout result would not be out of sample")
        if stats is None:
            out["error"] = "the signal raised an error while running; it still counts as a trial"
            return out
        audit = stats.get("leak_audit", {})
        out["naive"] = {k: _r(stats.get(k)) for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown",
                                                      "t_stat", "avg_daily_turnover", "n_obs")}
        out["leak_audit"] = {
            "passed": audit.get("passed", True),
            # Data-level warnings are reported once in describe_data, not on every trial.
            "findings": [f"{f['check']}: {f['detail']}" for f in audit.get("findings", [])
                         if f["check"] != "survivorship"],
        }
        same = self.ledger.canonical_id(hid)
        if same != hid:
            out["same_trial_as"] = same
            out["note"] = "same portfolio as an earlier test; logged but not counted as a new trial"
        if trial["status"] == "ok":
            r = self.ledger.returns_matrix()[same].dropna()
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
        hyp = self.ledger.hypotheses().set_index("hypothesis_id")
        specs = hyp["spec"]
        first_look = self._first_holdout_seq()
        rows = [{
            "hypothesis_id": hid,
            "expression": specs[hid].get("expression"),
            "naive_sharpe": _r(row.sr_annual, 3),
            "deflated_sharpe_ratio_effective": _r(row.dsr_effective),
            "deflated_sharpe_ratio_analytic": _r(row.dsr_analytic),
            "deflated_sharpe_ratio_trial_spread": _r(row.dsr),
            "significant_after_fdr": bool(row.sig_bh),
            "haircut_sharpe": _r(row.sr_haircut_holm, 3),
            "tested_after_holdout_look": bool(first_look is not None and hyp.loc[hid, "seq"] > first_look),
        } for hid, row in tab.iterrows()]
        s = ev.summary()
        return {
            "n_trials": s["n_trials"],
            "effective_independent_trials": _r(s["n_effective_trials"], 1),
            "hurdle_sharpe_annual_effective": _r(s["hurdle_sharpe_annual_effective"], 3),
            "hurdle_sharpe_annual_analytic": _r(s["hurdle_sharpe_annual_analytic"], 3),
            "hurdle_sharpe_annual_trial_spread": _r(s["hurdle_sharpe_annual"], 3),
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
        out = {"chain_ok": rep.ok, "signed": self.ledger.signed, "entries": rep.n_entries,
               "head_hash": self.ledger.head_hash(), "first_bad_entry": rep.first_bad_seq,
               "reason": rep.reason}
        if self.anchor_path is not None and os.path.exists(self.anchor_path):
            a = self.ledger.verify_anchors(self.anchor_path, self.name)
            out["anchors_ok"], out["anchors"] = a.ok, a.reason
        return out

    # ------------------------------------------------------------ holdout
    def _holdout_uses(self) -> int:
        return sum(1 for n in self.ledger.notes() if isinstance(n.get("holdout_eval"), dict))

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
        compiled = self._compile(spec["expression"])
        build = self._weights_fn(compiled, spec["q"], spec["rebalance"])
        res = run_backtest(build(self._full), self._full, lag=spec.get("lag", self.execution_lag),
                           cost_bps=self.cost_bps)
        ho = res.returns.loc[res.returns.index >= self.holdout_start]
        sr, skew, kurt, t_obs = moments(ho)
        result = {
            "holdout_start": str(self.holdout_start.date()),
            "holdout_days": int(t_obs),
            "holdout_sharpe": _r(sr * math.sqrt(252), 3),
            "holdout_psr_vs_zero": _r(psr(sr, t_obs, skew, kurt, 0.0)),
        }
        self.ledger.note("holdout_eval", hypothesis_id=hypothesis_id, data={"holdout_eval": result})
        self._anchor()
        result["holdout_evaluations_left"] = self._holdout_remaining()
        return {"status": "ok", "hypothesis_id": hypothesis_id, **result}
