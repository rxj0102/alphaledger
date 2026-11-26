"""A small, safe, causal expression language for signals.

Agents describe signals as expressions such as::

    rank(momentum(252, 21)) - rank(ts_std(returns, 63))

instead of arbitrary code. Expressions are parsed with Python's ``ast`` module
and only a whitelist of node types, names and functions is accepted, so
nothing can be imported, no attribute can be read and no file can be touched.
Every operator only looks backwards in time (there is no negative delay), so
a well-formed expression cannot use future data. The leakage audit still runs
on every trial as a second line of defence.

Variables
---------
returns   daily simple returns (dates x assets)
price     cumulative return index, (1 + returns).cumprod()

Time-series operators (window ``n`` is an integer constant, 1 to 2520)
---------------------------------------------------------------------
ts_sum(x, n)  ts_mean(x, n)  ts_std(x, n)  ts_min(x, n)  ts_max(x, n)
ts_zscore(x, n)    trailing z-score of x against its own last n values
ts_rank(x, n)      percentile of today's x within its last n values
delay(x, d)        x from d days ago (d >= 0)
delta(x, d)        x - delay(x, d)
ewm(x, h)          exponentially weighted mean with half-life h
momentum(L, S)     compounded return from L days ago to S days ago

Cross-sectional operators (each date separately)
------------------------------------------------
rank(x)  zscore(x)  demean(x)

Element-wise
------------
abs(x)  sign(x)  clip(x, lo, hi)  + - * /  and numeric constants
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..signals import momentum as _momentum

MAX_NODES = 80
MAX_WINDOW = 2520


class ExpressionError(ValueError):
    pass


Frame = pd.DataFrame
Fn = Callable[[Frame], Frame]


def _win(n, name: str, lo: int = 1) -> int:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise ExpressionError(f"{name}: window must be an integer constant")
    if not lo <= n <= MAX_WINDOW:
        raise ExpressionError(f"{name}: window must be between {lo} and {MAX_WINDOW}")
    return int(n)


def _ts_rank(x: Frame, n: int) -> Frame:
    def last_pct(a):
        return (a <= a[-1]).mean()

    return x.rolling(n, min_periods=n).apply(last_pct, raw=True)


def _xs_zscore(x: Frame) -> Frame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)


# name -> (number of frame args, number of int args, builder)
TS_OPS = {
    "ts_sum": lambda x, n: x.rolling(n, min_periods=n).sum(),
    "ts_mean": lambda x, n: x.rolling(n, min_periods=n).mean(),
    "ts_std": lambda x, n: x.rolling(n, min_periods=max(n, 2)).std(),
    "ts_min": lambda x, n: x.rolling(n, min_periods=n).min(),
    "ts_max": lambda x, n: x.rolling(n, min_periods=n).max(),
    "ts_zscore": lambda x, n: (x - x.rolling(n, min_periods=n).mean())
    / x.rolling(n, min_periods=max(n, 2)).std().replace(0, np.nan),
    "ts_rank": _ts_rank,
    "ewm": lambda x, h: x.ewm(halflife=h, min_periods=h).mean(),
}
LAG_OPS = {
    "delay": lambda x, d: x.shift(d),
    "delta": lambda x, d: x - x.shift(d),
}
XS_OPS = {
    "rank": lambda x: x.rank(axis=1, pct=True),
    "zscore": _xs_zscore,
    "demean": lambda x: x.sub(x.mean(axis=1), axis=0),
    "abs": lambda x: x.abs(),
    "sign": lambda x: np.sign(x),
}
VARIABLES = {
    "returns": lambda r: r,
    "price": lambda r: (1 + r.fillna(0)).cumprod().where(r.notna()),
}
FUNCTIONS = sorted(set(TS_OPS) | set(LAG_OPS) | set(XS_OPS) | {"momentum", "clip"})


@dataclass
class Compiled:
    canonical: str  # normalized expression text, used for duplicate detection
    fn: Fn
    n_nodes: int

    def __call__(self, returns: Frame) -> Frame:
        out = self.fn(returns)
        if not isinstance(out, pd.DataFrame):
            raise ExpressionError("expression must produce a per-asset signal, not a constant")
        return out.replace([np.inf, -np.inf], np.nan)


def _const(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_const(node.operand)
    raise ExpressionError("expected a numeric constant")


def _build(node) -> Fn | float:
    if isinstance(node, ast.Expression):
        return _build(node.body)
    if isinstance(node, ast.Constant):
        return _const(node)
    if isinstance(node, ast.Name):
        if node.id not in VARIABLES:
            raise ExpressionError(f"unknown variable {node.id!r}; available: {sorted(VARIABLES)}")
        return VARIABLES[node.id]
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            inner = _build(node.operand)
            return -inner if isinstance(inner, (int, float)) else (lambda r, f=inner: -f(r))
        if isinstance(node.op, ast.UAdd):
            return _build(node.operand)
        raise ExpressionError("only unary + and - are allowed")
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}
        op = ops.get(type(node.op))
        if op is None:
            raise ExpressionError("only + - * / are allowed")
        a, b = _build(node.left), _build(node.right)
        if not callable(a) and not callable(b):
            return float(op(a, b))

        def ev(side, r):
            return side(r) if callable(side) else side

        return lambda r, a=a, b=b, op=op: op(ev(a, r), ev(b, r))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only plain function calls are allowed")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")
        name, args = node.func.id, node.args
        if name in TS_OPS or name in LAG_OPS:
            if len(args) != 2:
                raise ExpressionError(f"{name}(x, n) takes 2 arguments")
            x = _build(args[0])
            if not callable(x):
                raise ExpressionError(f"{name}: first argument must be a signal")
            lo = 0 if name in LAG_OPS else 1
            n = _win(_const(args[1]), name, lo)
            op = TS_OPS.get(name) or LAG_OPS[name]
            return lambda r, x=x, n=n, op=op: op(x(r), n)
        if name in XS_OPS:
            if len(args) != 1:
                raise ExpressionError(f"{name}(x) takes 1 argument")
            x = _build(args[0])
            if not callable(x):
                raise ExpressionError(f"{name}: argument must be a signal")
            op = XS_OPS[name]
            return lambda r, x=x, op=op: op(x(r))
        if name == "momentum":
            if len(args) != 2:
                raise ExpressionError("momentum(lookback, skip) takes 2 arguments")
            lb, sk = _win(_const(args[0]), name, 2), _win(_const(args[1]), name, 0)
            if sk >= lb:
                raise ExpressionError("momentum: skip must be smaller than lookback")
            return lambda r, lb=lb, sk=sk: _momentum(r, lb, sk)
        if name == "clip":
            if len(args) != 3:
                raise ExpressionError("clip(x, lo, hi) takes 3 arguments")
            x = _build(args[0])
            lo, hi = _const(args[1]), _const(args[2])
            if lo > hi:
                raise ExpressionError("clip: lo must not exceed hi")
            return lambda r, x=x, lo=lo, hi=hi: x(r).clip(lo, hi)
        raise ExpressionError(f"unknown function {name!r}; available: {FUNCTIONS}")
    raise ExpressionError(f"syntax not allowed: {type(node).__name__}")


def compile_expression(text: str) -> Compiled:
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("expression must be a non-empty string")
    if len(text) > 2000:
        raise ExpressionError("expression too long")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from None
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > MAX_NODES:
        raise ExpressionError(f"expression too complex ({n_nodes} nodes, max {MAX_NODES})")
    fn = _build(tree)
    if not callable(fn):
        raise ExpressionError("expression must depend on the data, not only constants")
    return Compiled(canonical=ast.unparse(tree), fn=fn, n_nodes=n_nodes)
