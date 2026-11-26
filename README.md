# alphaledger

A research integrity layer for AI-generated alpha.

AI agents can propose and backtest hundreds of trading signals in an afternoon. The best of those backtests will look great by luck alone. alphaledger sits between the agent and the data, logs every hypothesis it tries (including the failures), and reports performance adjusted for how many things were tested, along with checks for look-ahead and data leakage.

## Status

| Week | Component | Status |
|---|---|---|
| 1 | Point-in-time data layer, vectorized backtester, momentum sanity check | Done |
| 2 | Append-only, hash-chained hypothesis ledger with lineage | Done |
| 3 | Stats engine: Deflated Sharpe Ratio, PBO via CSCV, FDR control, t-stat haircuts | Done |
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

## Week 2 design: the hypothesis ledger

`alphaledger/ledger/` stores every hypothesis before it is tested and every outcome after, including runs that fail or raise. It uses SQLite from the Python standard library, so there is no extra dependency.

**Tamper evidence.** Each entry stores the hash of the previous entry, and its own hash covers its content plus that link. Editing, deleting or reordering any past entry breaks the chain, and `verify()` reports the first bad entry. SQLite triggers also reject UPDATE and DELETE through normal SQL. Deleting the newest entries leaves a valid shorter chain, so `head_hash()` can be published externally (a commit message, a report) and checked later with `verify(expected_head=...)`.

**Trial counting.** `n_trials()` counts distinct specs that were actually tested. Failed runs count. Re-registering an identical spec (even with keys in a different order) is logged and linked to the original but does not add a trial. This count is what the Week 3 stats engine deflates by.

**Lineage.** Each hypothesis can name a parent, so the tree of ideas an agent explored can be rebuilt with `lineage()` and `children()`.

**Data fingerprint.** Each result stores a hash of the exact returns data it was tested on.

**One entry point.** `run_logged()` registers, backtests and records in one call, catching exceptions as failed trials. The Week 5 agent gateway will expose only this path.

Demo:

```
python examples/week2_ledger_demo.py
```

It sweeps 14 momentum variants on a pure noise market, then deletes the worst trial from a copy of the ledger. Output:

```
Trials logged: 14
Best naive Sharpe on pure noise: 0.55 (hypothesis h_dd181499ccbd)
Chain verifies: True
Published head hash: e12a09c0eb38d28a...
After deleting entry 24: verifies=False, first bad entry=24, reason: missing entry 24
```

Removing the content hash check or the duplicate detection from the code makes the test suite fail, so the tests are checking the real guarantees.

## Week 3 design: the stats engine

`alphaledger/stats/` turns the ledger into trial-adjusted results. `evaluate(ledger)` scores every recorded trial at once.

| Method | Module | What it answers |
|---|---|---|
| Probabilistic Sharpe Ratio | `sharpe.py` | How confident are we the true Sharpe beats zero, given sample length, skew and fat tails? |
| Deflated Sharpe Ratio | `sharpe.py` | Same, but against the best Sharpe expected from N zero-skill trials |
| Minimum track record length | `sharpe.py` | How many observations are needed for a given confidence |
| Probability of Backtest Overfitting (CSCV) | `pbo.py` | How often does the in-sample winner land below the out-of-sample median? |
| Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli | `multiple.py` | Which trials stay significant after correcting for the number tested |
| Harvey and Liu haircut Sharpe | `multiple.py` | What Sharpe survives once the p-value is adjusted for multiple testing |

Choices made in `evaluate`:
- N is the ledger trial count, which includes failed runs. Hiding failures cannot lower the bar.
- The variance of trial Sharpes is estimated from the trials themselves, as the DSR paper requires.
- Sharpe ratios are per period and kurtosis is Pearson kurtosis (3 for a normal), matching the papers.

**Checked against published numbers:**
- The numerical example in Bailey and Lopez de Prado (2014): annual SR 2.5, T = 1250, skew -3, kurtosis 10, N = 100, trial variance 0.5. The paper reports DSR 0.9004. alphaledger gives 0.9004.
- A monthly worked example (quantt.co.uk DSR guide): hurdle 0.3797 and DSR 0.624 reproduced.
- The 15 p-value example in Benjamini and Hochberg (1995): BH rejects 4, Bonferroni and Holm reject 3, as published.
- All four p-value corrections match statsmodels to 1e-12 on random inputs.

Removing the non-normality term from the PSR, or making PBO pick the worst in-sample trial, makes the suite fail. Building the demo also exposed a real bug: very large t-stats underflowed to p = 0 and crashed the haircut. It is fixed with tail-safe formulas and covered by a regression test.

Demo:

```
python examples/week3_deflation_demo.py
```

It runs the same 60-variant momentum sweep on two synthetic markets (150 assets, 8 years, seed 21, 5 bps costs). Output:

| | Pure noise | Planted momentum |
|---|---|---|
| Best naive annual Sharpe | 0.50 | 3.12 |
| Hurdle Sharpe for 60 trials | 0.47 | 0.70 |
| Deflated Sharpe Ratio of best | 0.540 | 1.000 |
| Significant before correction | 0 | 60 |
| Significant after BH | 0 | 60 |
| Passing DSR 0.95 | 0 | 60 |
| PBO | 0.52 | 0.22 |

On noise, PBO is close to the 0.5 expected when picking the winner is a coin flip, and the best trial barely clears the hurdle Sharpe. With a real edge, every variant survives correction. PBO is still 0.22 because the variants all share the same edge, so which one wins in sample is partly luck. PBO measures how reliable the choice among trials is, not whether the edge is real. Both views are needed.

## License

MIT
