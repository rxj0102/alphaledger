"""MCP server exposing the research gateway to any MCP client (Claude Desktop,
Claude Code, or another agent).

Run:
    alphaledger-mcp --ledger research.db --synthetic          # demo data
    alphaledger-mcp --ledger research.db --returns-csv r.csv  # your own daily returns

Recommended locked-down deployment (the operator runs this, the agent only
gets the MCP connection):
    export ALPHALEDGER_KEY=...            # shared-secret signing, never given to the agent
    # or, so outside auditors can verify with a public key:
    alphaledger-keygen operator.pem operator.pub
    export ALPHALEDGER_SIGNING_KEY_FILE=/srv/keys/operator.pem
    alphaledger-mcp --ledger /srv/research.db --returns-csv r.csv \
        --registry /srv/registry.jsonl --anchors /srv/anchors/anchors.jsonl

Point-in-time macro data from FRED (needs FRED_API_KEY in the environment):
    alphaledger-mcp ... --fred INDPRO --fred UNRATE
Each series becomes a variable (indpro, unrate) holding, on every date, the
latest value released by then. This path makes network calls and has only
been tested offline against fixtures.

The CSV must have dates in the first column and one column of daily simple
returns per asset.

Claude Desktop config example (claude_desktop_config.json):
    {"mcpServers": {"alphaledger": {"command": "alphaledger-mcp",
                                    "args": ["--ledger", "/path/research.db", "--synthetic"]}}}
"""

from __future__ import annotations

import argparse
import threading

import pandas as pd
from mcp.server.mcpserver import MCPServer

from ..synthetic import make_market
from .expr import __doc__ as EXPR_DOC
from .service import ResearchGateway

INSTRUCTIONS = (
    "You are researching cross-sectional equity signals. Every backtest you run is "
    "permanently logged, including failures, and results are deflated for the number "
    "of trials. Prefer a few well-motivated hypotheses over large blind sweeps. "
    "Give a rationale for each test and set parent_id when refining an earlier idea. "
    "Judge results by deflated_sharpe_ratio_effective, not the naive Sharpe. The holdout is "
    "hidden and can be opened only a few times; use it for your final candidate."
)


def build_server(gateway: ResearchGateway) -> MCPServer:
    mcp = MCPServer(name="alphaledger", instructions=INSTRUCTIONS)
    lock = threading.Lock()  # tools run in worker threads; one ledger write at a time

    @mcp.tool(description="Describe the data, the hidden holdout, the portfolio rules and the trial count so far.")
    def describe_data() -> dict:
        with lock:
            return gateway.describe()

    @mcp.tool(description="Reference for the signal expression language: variables, functions and examples.")
    def expression_language() -> str:
        return EXPR_DOC

    @mcp.tool(description=(
        "Register, leak-audit and backtest one signal expression on the in-sample data. "
        "Always counts as a trial once it compiles. q is the quantile per leg (0 to 0.5), "
        "rebalance_freq is D, W or M. Returns naive stats plus trial-adjusted stats."))
    def test_signal(expression: str, rationale: str, q: float = 0.2, rebalance_freq: str = "M",
                    parent_id: str | None = None) -> dict:
        with lock:
            return gateway.test_signal(expression, q=q, rebalance_freq=rebalance_freq,
                                       rationale=rationale, parent_id=parent_id)

    @mcp.tool(description="Top trials ranked by naive Sharpe with deflated Sharpe, FDR significance, haircut Sharpe and PBO.")
    def leaderboard(top: int = 10) -> dict:
        with lock:
            return gateway.leaderboard(top)

    @mcp.tool(description="Ancestors and children of a hypothesis.")
    def lineage(hypothesis_id: str) -> dict:
        with lock:
            return gateway.lineage(hypothesis_id)

    @mcp.tool(description="Verify the ledger hash chain and return the current head hash.")
    def verify_ledger() -> dict:
        with lock:
            return gateway.verify()

    @mcp.tool(description="Evaluate one tested hypothesis on the hidden holdout. Limited budget; every use is logged.")
    def evaluate_holdout(hypothesis_id: str) -> dict:
        with lock:
            return gateway.evaluate_holdout(hypothesis_id)

    return mcp


def main(argv=None) -> None:  # pragma: no cover - process entry point
    ap = argparse.ArgumentParser(prog="alphaledger-mcp")
    ap.add_argument("--ledger", required=True, help="path of the SQLite ledger file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--returns-csv")
    src.add_argument("--synthetic", action="store_true", help="noise market with a planted momentum edge")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--holdout-budget", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--lag", type=int, default=2, help="2 = trade at the next close (default), 1 = same close")
    ap.add_argument("--registry", help="file binding each dataset to one ledger")
    ap.add_argument("--anchors", help="append-only file the ledger head is anchored to after every write")
    ap.add_argument("--survivor-free", action="store_true",
                    help="declare the universe survivor-free (e.g. industry portfolios); silences the heuristic")
    ap.add_argument("--fred", action="append", default=[], metavar="SERIES",
                    help="add a FRED series as a point-in-time variable (repeatable)")
    args = ap.parse_args(argv)

    if args.synthetic:
        returns, label = make_market(n_assets=150, drift_vol=0.0008, seed=0), "synthetic (planted momentum, seed 0)"
    else:
        returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)
        label = f"csv:{args.returns_csv}"
    pit, fields = None, None
    if args.fred:
        import os

        from ..data import PITStore
        from ..data.sources import fetch_fred_vintages

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise SystemExit("set FRED_API_KEY to use --fred")
        pit = PITStore()
        for series in args.fred:
            pit.add(fetch_fred_vintages(series, api_key))
        fields = {series.lower(): series for series in args.fred}
    import os as _os

    from ..ledger.store import load_key, load_signer

    signer_file = _os.environ.get("ALPHALEDGER_SIGNING_KEY_FILE")
    signer = load_signer(signer_file) if signer_file else None

    gw = ResearchGateway(returns, args.ledger, holdout_frac=args.holdout_frac,
                         holdout_budget=args.holdout_budget, cost_bps=args.cost_bps, data_label=label,
                         execution_lag=args.lag, pit=pit, pit_fields=fields, key=load_key(),
                         anchor_path=args.anchors, registry_path=args.registry,
                         survivor_free=args.survivor_free, signer=signer)
    build_server(gw).run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
