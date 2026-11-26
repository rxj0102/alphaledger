"""Append-only, hash-chained hypothesis ledger.

Every hypothesis an agent or researcher tries is registered before it is
tested, and every result is recorded against it, including failures. Entries
form a hash chain: each entry's hash covers its own content plus the previous
entry's hash. Editing, deleting or reordering any past entry breaks the chain
from that point on, and ``verify()`` reports where.

Two layers of protection:

1. SQLite triggers reject UPDATE and DELETE through normal SQL.
2. The hash chain detects changes made by any other route (dropping the
   triggers, editing the file directly, restoring an old copy). Publishing
   ``head_hash()`` somewhere external (a commit message, a report) anchors
   the whole history up to that point.

Trial counting: the number of distinct hypothesis specs tested is what the
stats engine (Week 3) uses to deflate performance. Re-registering an
identical spec is logged but does not add a new trial.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GENESIS = "0" * 64
KINDS = ("hypothesis", "result", "note")

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    seq           INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,
    kind          TEXT NOT NULL,
    hypothesis_id TEXT,
    parent_id     TEXT,
    payload       TEXT NOT NULL,
    prev_hash     TEXT NOT NULL,
    hash          TEXT NOT NULL UNIQUE
);
CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""


class LedgerError(RuntimeError):
    pass


# ---------------------------------------------------------------- hashing
def _clean(obj):
    """Make an object JSON-canonical: NaN/inf to None, numpy to python, tuples to lists."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def canonical(obj) -> str:
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def entry_hash(seq, ts, kind, hypothesis_id, parent_id, payload_json, prev_hash) -> str:
    body = canonical(
        {
            "seq": seq,
            "ts": ts,
            "kind": kind,
            "hypothesis_id": hypothesis_id,
            "parent_id": parent_id,
            "payload": payload_json,
            "prev_hash": prev_hash,
        }
    )
    return sha256(body)


def spec_hash(spec: dict) -> str:
    return sha256(canonical(spec))


def data_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame's index, columns and values."""
    h = hashlib.sha256()
    h.update(canonical([str(c) for c in df.columns]).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- verify report
@dataclass
class VerifyReport:
    ok: bool
    n_entries: int
    first_bad_seq: int | None = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- ledger
class Ledger:
    def __init__(self, path: str | Path = ":memory:", clock=None):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat(timespec="microseconds"))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ low level
    def _append(self, kind: str, payload: dict, hypothesis_id=None, parent_id=None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        payload_json = canonical(payload)
        with self.conn:  # one transaction per append
            row = self.conn.execute(
                "SELECT seq, hash FROM entries ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            seq = 1 if row is None else row[0] + 1
            prev = GENESIS if row is None else row[1]
            ts = self._clock()
            h = entry_hash(seq, ts, kind, hypothesis_id, parent_id, payload_json, prev)
            self.conn.execute(
                "INSERT INTO entries (seq, ts, kind, hypothesis_id, parent_id, payload, prev_hash, hash)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (seq, ts, kind, hypothesis_id, parent_id, payload_json, prev, h),
            )
        return {"seq": seq, "ts": ts, "hash": h}

    def entries(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM entries ORDER BY seq", self.conn)
        return df

    def head_hash(self) -> str:
        row = self.conn.execute("SELECT hash FROM entries ORDER BY seq DESC LIMIT 1").fetchone()
        return GENESIS if row is None else row[0]

    def verify(self, expected_head: str | None = None) -> VerifyReport:
        """Recompute the whole chain. Optionally check it ends at a published head hash."""
        rows = self.conn.execute(
            "SELECT seq, ts, kind, hypothesis_id, parent_id, payload, prev_hash, hash"
            " FROM entries ORDER BY seq"
        ).fetchall()
        prev = GENESIS
        expected_seq = 1
        for seq, ts, kind, hid, pid, payload, prev_hash, h in rows:
            if seq != expected_seq:
                return VerifyReport(False, len(rows), expected_seq, f"missing entry {expected_seq}")
            if prev_hash != prev:
                return VerifyReport(False, len(rows), seq, "prev_hash does not match previous entry")
            if entry_hash(seq, ts, kind, hid, pid, payload, prev_hash) != h:
                return VerifyReport(False, len(rows), seq, "entry content does not match its hash")
            prev = h
            expected_seq += 1
        if expected_head is not None and prev != expected_head:
            # Covers truncation of the newest entries, which leaves a valid shorter chain.
            known = {r[7] for r in rows}
            reason = (
                "published head is not the last entry (entries appended since)"
                if expected_head in known
                else "published head not found (history truncated or replaced)"
            )
            return VerifyReport(expected_head in known, len(rows), None, reason)
        return VerifyReport(True, len(rows))

    # ------------------------------------------------------------ hypotheses
    def register(self, spec: dict, parent_id: str | None = None, author: str = "agent",
                 rationale: str = "") -> str:
        """Register a hypothesis before testing it. Returns its id.

        ``spec`` should fully describe the signal (formula or code reference,
        parameters, universe, data fields). The id is derived from the spec
        hash plus the entry hash, so two registrations never share an id.
        """
        if not isinstance(spec, dict) or not spec:
            raise ValueError("spec must be a non-empty dict")
        if parent_id is not None and not self._exists(parent_id):
            raise LedgerError(f"unknown parent hypothesis {parent_id}")
        sh = spec_hash(spec)
        duplicate_of = self._first_with_spec(sh)
        payload = {
            "spec": spec,
            "spec_hash": sh,
            "author": author,
            "rationale": rationale,
            "duplicate_of": duplicate_of,
        }
        # The id must be known before the hash, so derive it from spec hash and sequence.
        nxt = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM entries").fetchone()[0]
        hid = "h_" + sha256(f"{sh}:{nxt}")[:12]
        self._append("hypothesis", payload, hypothesis_id=hid, parent_id=parent_id)
        return hid

    def record(self, hypothesis_id: str, stats: dict, returns: pd.Series | None = None,
               data_hash: str | None = None, status: str = "ok", error: str = "") -> None:
        """Record the outcome of testing a hypothesis (also for failed runs)."""
        if not self._exists(hypothesis_id):
            raise LedgerError(f"unknown hypothesis {hypothesis_id}")
        payload = {"stats": stats, "status": status, "error": error, "data_hash": data_hash}
        if returns is not None:
            r = pd.Series(returns).dropna()
            payload["returns"] = {
                "dates": [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(r.index)],
                "values": [float(x) for x in r.to_numpy()],
            }
        self._append("result", payload, hypothesis_id=hypothesis_id)

    def note(self, text: str, hypothesis_id: str | None = None) -> None:
        self._append("note", {"text": text}, hypothesis_id=hypothesis_id)

    # ------------------------------------------------------------ queries
    def _exists(self, hid: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM entries WHERE kind='hypothesis' AND hypothesis_id=? LIMIT 1", (hid,)
        ).fetchone() is not None

    def _first_with_spec(self, sh: str) -> str | None:
        for hid, payload in self.conn.execute(
            "SELECT hypothesis_id, payload FROM entries WHERE kind='hypothesis' ORDER BY seq"
        ):
            if json.loads(payload)["spec_hash"] == sh:
                return hid
        return None

    def hypotheses(self) -> pd.DataFrame:
        rows = []
        for seq, ts, hid, pid, payload in self.conn.execute(
            "SELECT seq, ts, hypothesis_id, parent_id, payload FROM entries"
            " WHERE kind='hypothesis' ORDER BY seq"
        ):
            p = json.loads(payload)
            rows.append({
                "hypothesis_id": hid, "parent_id": pid, "seq": seq, "ts": ts,
                "spec_hash": p["spec_hash"], "author": p["author"],
                "duplicate_of": p["duplicate_of"], "spec": p["spec"],
            })
        return pd.DataFrame(rows, columns=["hypothesis_id", "parent_id", "seq", "ts", "spec_hash",
                                           "author", "duplicate_of", "spec"])

    def results(self) -> pd.DataFrame:
        """Latest result per hypothesis, with stats expanded into columns."""
        rows = []
        for seq, hid, payload in self.conn.execute(
            "SELECT seq, hypothesis_id, payload FROM entries WHERE kind='result' ORDER BY seq"
        ):
            p = json.loads(payload)
            rows.append({"hypothesis_id": hid, "result_seq": seq, "status": p["status"],
                         "data_hash": p["data_hash"], **(p["stats"] or {})})
        if not rows:
            return pd.DataFrame(columns=["hypothesis_id", "result_seq", "status", "data_hash"])
        df = pd.DataFrame(rows)
        return df.groupby("hypothesis_id", as_index=False).last()

    def trials(self) -> pd.DataFrame:
        """Every hypothesis with its latest result (NaN where never tested)."""
        h = self.hypotheses().drop(columns=["spec"])
        return h.merge(self.results(), on="hypothesis_id", how="left")

    def n_trials(self) -> int:
        """Distinct specs that have at least one recorded result (any status)."""
        t = self.trials()
        tested = t[t["result_seq"].notna()]
        return int(tested["spec_hash"].nunique())

    def returns_matrix(self, status: str = "ok") -> pd.DataFrame:
        """Dates x hypotheses matrix of recorded daily returns (latest result each).

        This is the input the Week 3 stats engine needs (PBO, deflated Sharpe).
        Duplicate specs are collapsed to their first registration.
        """
        h = self.hypotheses().set_index("hypothesis_id")
        cols = {}
        for hid, payload in self.conn.execute(
            "SELECT hypothesis_id, payload FROM entries WHERE kind='result' ORDER BY seq"
        ):
            p = json.loads(payload)
            if p["status"] != status or "returns" not in p:
                continue
            canonical_id = h.loc[hid, "duplicate_of"] or hid
            cols[canonical_id] = pd.Series(
                p["returns"]["values"], index=pd.to_datetime(p["returns"]["dates"])
            )
        return pd.DataFrame(cols).sort_index()

    def lineage(self, hypothesis_id: str) -> list[str]:
        """Ancestors from the root down to ``hypothesis_id``."""
        parents = dict(
            self.conn.execute(
                "SELECT hypothesis_id, parent_id FROM entries WHERE kind='hypothesis'"
            ).fetchall()
        )
        if hypothesis_id not in parents:
            raise LedgerError(f"unknown hypothesis {hypothesis_id}")
        chain = [hypothesis_id]
        while parents.get(chain[-1]) is not None:
            chain.append(parents[chain[-1]])
        return chain[::-1]

    def children(self, hypothesis_id: str) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT hypothesis_id FROM entries WHERE kind='hypothesis' AND parent_id=? ORDER BY seq",
            (hypothesis_id,),
        )]
