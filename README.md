# alphaledger

A research integrity layer for AI-generated alpha.

AI agents can propose and backtest hundreds of trading signals in an afternoon. The best of those backtests will look great by luck alone. alphaledger sits between the agent and the data, logs every hypothesis it tries (including the failures), and reports performance adjusted for how many things were tested, along with checks for look-ahead and data leakage.

## Status

| Week | Component | Status |
|---|---|---|
| 1 | Point-in-time data layer, vectorized backtester, momentum sanity check | Done |
| 2 | Append-only, hash-chained hypothesis ledger with lineage | Planned |
| 3 | Stats engine: Deflated Sharpe Ratio, PBO via CSCV, FDR control, t-stat haircuts | Planned |
| 4 | Leakage detectors tested against planted leaks | Planned |
| 5 | MCP gateway so an agent can only reach data and backtests through the ledger | Planned |
| 6 | Dashboard and case study on agent-generated signals | Planned |

## Week 1 design

**Point-in-time store** (`alphaledger/data/pit.py`). Every observation has a `period` (what it describes) and an `available_at` (when it could first be known). Queries are made as of a decision time and only see rows with `available_at` at or before it. Revisions are stored as new vintages, so a later revision never leaks into an earlier decision. `panel()` builds a daily dates by entities input with an optional extra lag.

**Backtester** (`alphaledger/backtest/engine.py`). Weights decided at the close of day t are held over day t+1. A zero lag raises `LookAheadError` unless explicitly allowed. Turnover based transaction costs, monthly rebalancing support, and standard metrics (Sharpe, t-stat, max drawdown, hit rate, turnover).

**Free data loaders** (`alphaledger/data/sources.py`):
- Ken French daily factors
- FRED vintages through the official API with the full real-time period, so each value enters at its release date (needs a free FRED API key)
- yfinance adjusted closes. Universes built from yfinance are survivorship-biased because delisted tickers are missing; any result using them says so.

Parsers are tested offline against fixture text. Network fetch functions are thin wrappers around them.

**Synthetic markets** (`alphaledger/synthetic.py`). Assets get a slowly drifting expected return, so momentum has a known, planted edge. Setting the drift to zero gives pure noise. This gives the pipeline a ground truth to be tested against.

## Tests

```
pip install -e .[dev]
pytest
```

The suite checks, among other things:
- values are invisible before release and revisions never leak backwards
- the lag alignment and transaction costs match hand calculations
- a signal built from same-day returns shows huge fake alpha with zero lag and nothing with a one day lag
- momentum recovers the planted edge, finds nothing in noise, and loses money when the sign is flipped
- changing today's and future returns does not change today's momentum signal

Planting bugs confirms the tests catch them: removing the momentum skip, or removing the execution lag, makes the suite fail.

## Run the sanity check

```
python examples/week1_momentum.py            # synthetic market with planted momentum
python examples/week1_momentum.py --noise    # synthetic pure noise
python examples/week1_momentum.py --yfinance AAPL MSFT NVDA JPM XOM   # real prices, needs pip install yfinance
```

Output from the two synthetic runs (200 assets, seed 0, 12-1 momentum, quintile long short, monthly rebalance, 1 day lag, 5 bps costs):

| Market | Sharpe | t-stat | Max drawdown |
|---|---|---|---|
| Planted momentum | 2.83 | 7.47 | -3.4% |
| Pure noise | 0.14 | 0.38 | -5.9% |

These are synthetic markets built to test the pipeline. They say nothing about real momentum returns.

## License

MIT
