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
import hmac
import os
import json
import math
import sqlite3
import threading
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
    hash          TEXT NOT NULL UNIQUE,
    sig           TEXT
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
def returns_hash(r: pd.Series) -> str:
    """Fingerprint of a return series, used to spot identical portfolios.

    Values are rounded to 1e-12 so that float noise from algebraically equal
    expressions (``x`` and ``x * 1``) does not create a new trial.
    """
    r = pd.Series(r).dropna()
    body = canonical({"d": [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(r.index)],
                      "v": [round(float(x), 12) + 0.0 for x in r.to_numpy()]})
    return sha256(body)


def load_key(value: str | bytes | None = None, env: str = "ALPHALEDGER_KEY") -> bytes | None:
    """Signing key from an explicit value or an environment variable."""
    if value is None:
        value = os.environ.get(env)
    if value is None or value == "":
        return None
    return value if isinstance(value, bytes) else value.encode()


def generate_signing_keypair(private_path: str | Path, public_path: str | Path) -> None:
    """Write a new Ed25519 key pair as PEM files (needs the ``cryptography`` package)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    Path(private_path).write_bytes(priv.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    os.chmod(private_path, 0o600)
    Path(public_path).write_bytes(priv.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))


def load_signer(path: str | Path):
    from cryptography.hazmat.primitives import serialization

    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)


def load_verify_key(path: str | Path):
    from cryptography.hazmat.primitives import serialization

    return serialization.load_pem_public_key(Path(path).read_bytes())


class Ledger:
    """Append-only, hash-chained, optionally signed hypothesis ledger.

    Two ways to sign every entry's hash:

    * ``key``: a shared secret (HMAC-SHA256). Only someone holding the same
      secret can verify, so it proves integrity to the operator but not to an
      outside auditor.
    * ``signer``: an Ed25519 private key. Anyone with the matching public key
      (``verify_key``) can verify, without being able to sign. Use this when a
      third party (an investment committee, an auditor, a client) must be able
      to check the ledger.

    Without a key, someone who rewrites the file can recompute every hash and
    the chain alone will still verify. Keep secrets and private keys outside
    the agent's reach. A ledger opened with only ``verify_key`` is read-only.
    """

    def __init__(self, path: str | Path = ":memory:", clock=None, key: str | bytes | None = None,
                 signer=None, verify_key=None):
        self.path = str(path)
        # Servers call tools from worker threads; appends are serialized by a lock.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(entries)")}
        if "sig" not in cols:  # ledgers created before signing existed
            self.conn.execute("ALTER TABLE entries ADD COLUMN sig TEXT")
        self._key = key if isinstance(key, (bytes, type(None))) else key.encode()
        self._signer = signer
        self._verify_key = verify_key or (signer.public_key() if signer is not None else None)
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat(timespec="microseconds"))
        self._reset_cache()

    @property
    def signed(self) -> bool:
        """True if signatures can be checked with what this ledger was opened with."""
        return self._key is not None or self._verify_key is not None

    @property
    def read_only(self) -> bool:
        return self._verify_key is not None and self._signer is None and self._key is None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ cache
    # Queries are served from an in-memory index that is synced incrementally
    # from the database, so cost per query does not grow with every reparse.
    def _reset_cache(self) -> None:
        self._last_seq = 0
        self._hyp: dict[str, dict] = {}  # hid -> hypothesis record
        self._hyp_order: list[str] = []
        self._first_spec: dict[str, str] = {}
        self._results: dict[str, dict] = {}  # hid -> latest result record
        self._notes: list[dict] = []

    def _sync(self) -> None:
        with self._lock:
            rows = self.conn.execute(
                "SELECT seq, ts, kind, hypothesis_id, parent_id, payload FROM entries"
                " WHERE seq > ? ORDER BY seq", (self._last_seq,)
            ).fetchall()
            for seq, ts, kind, hid, pid, payload in rows:
                self._ingest(seq, ts, kind, hid, pid, json.loads(payload))

    def _ingest(self, seq, ts, kind, hid, pid, p) -> None:
        if seq <= self._last_seq:
            return
        self._last_seq = seq
        if kind == "hypothesis":
            self._hyp[hid] = {"hypothesis_id": hid, "parent_id": pid, "seq": seq, "ts": ts,
                              "spec_hash": p["spec_hash"], "author": p["author"],
                              "duplicate_of": p["duplicate_of"], "spec": p["spec"],
                              "rationale": p.get("rationale", "")}
            self._hyp_order.append(hid)
            self._first_spec.setdefault(p["spec_hash"], hid)
        elif kind == "result":
            rec = {"seq": seq, "status": p["status"], "data_hash": p["data_hash"],
                   "stats": p["stats"] or {}, "returns_hash": p.get("returns_hash"),
                   "returns": None}
            if "returns" in p:
                rec["returns"] = pd.Series(p["returns"]["values"],
                                           index=pd.to_datetime(p["returns"]["dates"]), dtype=float)
            self._results[hid] = rec
        else:
            self._notes.append({"seq": seq, "hypothesis_id": hid, **p})

    # ------------------------------------------------------------ low level
    def _sign(self, h: str) -> str | None:
        if self._signer is not None:
            return "ed25519:" + self._signer.sign(h.encode()).hex()
        return hmac.new(self._key, h.encode(), hashlib.sha256).hexdigest() if self._key else None

    def _check_sig(self, h: str, sig: str | None) -> bool:
        if sig is None:
            return False
        if sig.startswith("ed25519:"):
            if self._verify_key is None:
                return False
            try:
                self._verify_key.verify(bytes.fromhex(sig[8:]), h.encode())
                return True
            except Exception:
                return False
        if self._key is None:
            return False
        return hmac.compare_digest(sig, hmac.new(self._key, h.encode(), hashlib.sha256).hexdigest())

    def _append(self, kind: str, payload: dict, hypothesis_id=None, parent_id=None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.read_only:
            raise LedgerError("ledger opened with a public key only is read-only")
        payload_json = canonical(payload)
        with self._lock, self.conn:  # one serialized transaction per append
            row = self.conn.execute(
                "SELECT seq, hash FROM entries ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            seq = 1 if row is None else row[0] + 1
            prev = GENESIS if row is None else row[1]
            ts = self._clock()
            h = entry_hash(seq, ts, kind, hypothesis_id, parent_id, payload_json, prev)
            self.conn.execute(
                "INSERT INTO entries (seq, ts, kind, hypothesis_id, parent_id, payload, prev_hash, hash, sig)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (seq, ts, kind, hypothesis_id, parent_id, payload_json, prev, h, self._sign(h)),
            )
        self._sync()
        return {"seq": seq, "ts": ts, "hash": h}

    def entries(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM entries ORDER BY seq", self.conn)

    def head(self) -> tuple[int, str]:
        row = self.conn.execute("SELECT seq, hash FROM entries ORDER BY seq DESC LIMIT 1").fetchone()
        return (0, GENESIS) if row is None else (int(row[0]), row[1])

    def head_hash(self) -> str:
        return self.head()[1]

    # ------------------------------------------------------------ verification
    def verify(self, expected_head: str | None = None, require_signatures: bool | None = None) -> VerifyReport:
        """Recompute the whole chain from the database (never from the cache).

        If the ledger was opened with a key, every entry's signature is checked
        too, and an unsigned entry is an error. ``expected_head`` checks the
        chain against a head hash published elsewhere.
        """
        if require_signatures is None:
            require_signatures = self.signed
        rows = self.conn.execute(
            "SELECT seq, ts, kind, hypothesis_id, parent_id, payload, prev_hash, hash, sig"
            " FROM entries ORDER BY seq"
        ).fetchall()
        prev = GENESIS
        expected_seq = 1
        for seq, ts, kind, hid, pid, payload, prev_hash, h, sig in rows:
            if seq != expected_seq:
                return VerifyReport(False, len(rows), expected_seq, f"missing entry {expected_seq}")
            if prev_hash != prev:
                return VerifyReport(False, len(rows), seq, "prev_hash does not match previous entry")
            if entry_hash(seq, ts, kind, hid, pid, payload, prev_hash) != h:
                return VerifyReport(False, len(rows), seq, "entry content does not match its hash")
            if require_signatures:
                if not self.signed:
                    return VerifyReport(False, len(rows), seq, "signatures required but no key given")
                if not self._check_sig(h, sig):
                    return VerifyReport(False, len(rows), seq, "signature missing or invalid")
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

    def has_signatures(self) -> bool:
        return self.conn.execute("SELECT 1 FROM entries WHERE sig IS NOT NULL LIMIT 1").fetchone() is not None

    def integrity(self) -> dict:
        """Plain-language integrity status, for reports.

        A hash chain that verifies without the signing key only shows the file
        is internally consistent; it does not show it was not rewritten.
        """
        rep = self.verify()
        sigs = self.has_signatures()
        if not rep.ok:
            status = "BROKEN"
        elif self.signed:
            status = "verified with signing key"
        elif sigs:
            status = "chain intact, signatures NOT checked (no key)"
        else:
            status = "chain intact, unsigned (a full rewrite would not be detected)"
        return {"ok": rep.ok, "status": status, "signed_entries": sigs, "key_checked": self.signed,
                "entries": rep.n_entries, "reason": rep.reason}

    # ------------------------------------------------------------ anchoring
    def anchor(self, anchor_path: str | Path, name: str) -> dict:
        """Append the current head to an external, append-only anchor file.

        Keep the anchor file somewhere the agent cannot write (a git repository
        that gets pushed, a different machine, a WORM bucket). Any later rewrite
        of the ledger up to an anchored entry changes that entry's hash, and
        ``verify_anchors`` reports it, with or without a signing key.
        """
        seq, h = self.head()
        rec = {"name": name, "seq": seq, "hash": h, "ts": self._clock()}
        with open(anchor_path, "a") as f:
            f.write(canonical(rec) + "\n")
        return rec

    def verify_anchors(self, anchor_path: str | Path, name: str) -> VerifyReport:
        base = self.verify()
        if not base.ok:
            return base
        hashes = dict(self.conn.execute("SELECT seq, hash FROM entries").fetchall())
        n = 0
        with open(anchor_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("name") != name:
                    continue
                n += 1
                if rec["seq"] == 0:
                    continue
                if rec["seq"] not in hashes:
                    return VerifyReport(False, base.n_entries, rec["seq"],
                                        f"anchored entry {rec['seq']} is missing (history truncated)")
                if hashes[rec["seq"]] != rec["hash"]:
                    return VerifyReport(False, base.n_entries, rec["seq"],
                                        f"entry {rec['seq']} differs from its anchor (history rewritten)")
        if n == 0:
            return VerifyReport(False, base.n_entries, None, f"no anchors named {name!r}")
        return VerifyReport(True, base.n_entries, None, f"{n} anchors match")

    # ------------------------------------------------------------ hypotheses
    def register(self, spec: dict, parent_id: str | None = None, author: str = "agent",
                 rationale: str = "") -> str:
        """Register a hypothesis before testing it. Returns its id.

        ``spec`` should fully describe the signal (formula or code reference,
        parameters, universe, data fields). The id is derived from the spec
        hash plus the sequence number, so two registrations never share an id.
        """
        if not isinstance(spec, dict) or not spec:
            raise ValueError("spec must be a non-empty dict")
        with self._lock:
            self._sync()
            if parent_id is not None and parent_id not in self._hyp:
                raise LedgerError(f"unknown parent hypothesis {parent_id}")
            sh = spec_hash(spec)
            payload = {
                "spec": spec,
                "spec_hash": sh,
                "author": author,
                "rationale": rationale,
                "duplicate_of": self._first_spec.get(sh),
            }
            hid = "h_" + sha256(f"{sh}:{self.head()[0] + 1}")[:12]
            self._append("hypothesis", payload, hypothesis_id=hid, parent_id=parent_id)
        return hid

    def record(self, hypothesis_id: str, stats: dict, returns: pd.Series | None = None,
               data_hash: str | None = None, status: str = "ok", error: str = "") -> dict:
        """Record the outcome of testing a hypothesis (also for failed runs).

        Returns {"duplicate_portfolio_of": hid or None}: an earlier hypothesis
        whose recorded returns are identical, i.e. the same portfolio.
        """
        with self._lock:
            self._sync()
            if hypothesis_id not in self._hyp:
                raise LedgerError(f"unknown hypothesis {hypothesis_id}")
            payload = {"stats": stats, "status": status, "error": error, "data_hash": data_hash}
            same = None
            if returns is not None:
                r = pd.Series(returns).dropna()
                payload["returns"] = {
                    "dates": [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(r.index)],
                    "values": [float(x) for x in r.to_numpy()],
                }
                rh = returns_hash(r)
                payload["returns_hash"] = rh
                for hid in self._hyp_order:
                    res = self._results.get(hid)
                    if hid != hypothesis_id and res and res["returns_hash"] == rh:
                        same = hid
                        break
            payload["duplicate_portfolio_of"] = same
            self._append("result", payload, hypothesis_id=hypothesis_id)
        return {"duplicate_portfolio_of": same}

    def note(self, text: str, hypothesis_id: str | None = None, data: dict | None = None) -> None:
        payload = {"text": text}
        if data:
            payload.update(data)
        self._append("note", payload, hypothesis_id=hypothesis_id)

    # ------------------------------------------------------------ queries
    def notes(self) -> list[dict]:
        self._sync()
        return list(self._notes)

    def hypotheses(self) -> pd.DataFrame:
        self._sync()
        cols = ["hypothesis_id", "parent_id", "seq", "ts", "spec_hash", "author", "duplicate_of", "spec"]
        df = pd.DataFrame([{k: self._hyp[h][k] for k in cols} for h in self._hyp_order], columns=cols)
        # Keep missing links as None. pandas may turn None into NaN, and NaN is
        # truthy, which silently breaks "duplicate_of or hypothesis_id" lookups.
        for col in ("parent_id", "duplicate_of"):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        return df

    def results(self) -> pd.DataFrame:
        """Latest result per hypothesis, with stats expanded into columns."""
        self._sync()
        rows = [{"hypothesis_id": h, "result_seq": r["seq"], "status": r["status"],
                 "data_hash": r["data_hash"], "returns_hash": r["returns_hash"], **r["stats"]}
                for h, r in self._results.items()]
        if not rows:
            return pd.DataFrame(columns=["hypothesis_id", "result_seq", "status", "data_hash", "returns_hash"])
        return pd.DataFrame(rows)

    def trials(self) -> pd.DataFrame:
        """Every hypothesis with its latest result (NaN where never tested)."""
        h = self.hypotheses().drop(columns=["spec"])
        return h.merge(self.results(), on="hypothesis_id", how="left")

    def _trial_key(self, hid: str) -> str:
        r = self._results[hid]
        if r["status"] in ("ok", "leak") and r["returns_hash"]:
            return "r:" + r["returns_hash"]  # same portfolio, same trial
        return "s:" + self._hyp[hid]["spec_hash"]

    def n_trials(self) -> int:
        """Distinct tested portfolios (failed runs counted by distinct spec).

        Identical specs count once, and so do different expressions that
        produce exactly the same returns (``x``, ``x * 1``, ``rank(rank(x))``).
        Correlated but different trials still count separately; the effective
        number of independent trials is estimated in ``stats.evaluate``.
        """
        self._sync()
        return len({self._trial_key(h) for h in self._results})

    def returns_matrix(self, status: str = "ok") -> pd.DataFrame:
        """Dates x hypotheses matrix of recorded daily returns (latest result each).

        One column per distinct portfolio, named after the first hypothesis
        that produced it.
        """
        self._sync()
        cols, seen = {}, set()
        for hid in self._hyp_order:
            r = self._results.get(hid)
            if r is None or r["status"] != status or r["returns"] is None:
                continue
            key = self._trial_key(hid)
            if key in seen:
                continue
            seen.add(key)
            cols[hid] = r["returns"]
        return pd.DataFrame(cols).sort_index()

    def canonical_id(self, hid: str) -> str:
        """First hypothesis that produced the same trial (itself if none)."""
        self._sync()
        if hid not in self._results:
            return hid
        key = self._trial_key(hid)
        for h in self._hyp_order:
            if h in self._results and self._trial_key(h) == key:
                return h
        return hid

    def lineage(self, hypothesis_id: str) -> list[str]:
        """Ancestors from the root down to ``hypothesis_id``."""
        self._sync()
        if hypothesis_id not in self._hyp:
            raise LedgerError(f"unknown hypothesis {hypothesis_id}")
        chain = [hypothesis_id]
        while self._hyp[chain[-1]]["parent_id"] is not None:
            chain.append(self._hyp[chain[-1]]["parent_id"])
        return chain[::-1]

    def children(self, hypothesis_id: str) -> list[str]:
        self._sync()
        return [h for h in self._hyp_order if self._hyp[h]["parent_id"] == hypothesis_id]


def keygen_main(argv=None) -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(prog="alphaledger-keygen",
                                 description="Create an Ed25519 key pair for signing ledgers")
    ap.add_argument("private_key")
    ap.add_argument("public_key")
    a = ap.parse_args(argv)
    generate_signing_keypair(a.private_key, a.public_key)
    print(f"private key: {a.private_key} (keep it from the agent)\npublic key: {a.public_key} (share with auditors)")
