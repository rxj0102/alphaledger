# alphaledger

A research integrity layer for AI-generated alpha.

AI agents can propose and backtest hundreds of trading signals in an afternoon. The best of those backtests will look great by luck alone. alphaledger sits between the agent and the data, logs every hypothesis it tries (including the failures), and reports performance adjusted for how many things were tested, along with checks for look-ahead and data leakage.

## Status

| Week | Component | Status |
|---|---|---|
| 1 | Point-in-time data layer, vectorized backtester, momentum sanity check | Done |
| 2 | Append-only, hash-chained hypothesis ledger with lineage | Done |
| 3 | Stats engine: Deflated Sharpe Ratio, PBO via CSCV, FDR control, t-stat haircuts | Done |
| 4 | Leakage detectors tested against planted leaks | Done |
| 5 | MCP gateway so an agent can only reach data and backtests through the ledger | Done |
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

## Week 4 design: leakage detectors

`alphaledger/leakage.py` audits any function that turns returns into signals or weights. A causal function's row for date t can only depend on data up to t, and the hard checks test exactly that.

| Check | Severity | How it works |
|---|---|---|
| Truncation | fail | Recompute with all data after t removed; row t must not change |
| Future perturbation | fail | Scramble every return after t; no row at or before t may change |
| IC implausibility | warn | Mean daily rank IC above 0.1, far beyond real daily signals |
| Lag fragility | warn | Sharpe above 1 at lag 1 that falls below 30% of that at lag 2 |
| Survivorship | warn | A multi-year universe in which no asset ever stops trading |
| Holdout reuse | warn | More than one trial evaluated on a declared holdout window |

`run_logged(..., audit_leaks=True)` runs the audit before recording. A trial that fails a hard check is stored with status leak: it still counts toward N, so the stats stay honest, but its returns are excluded from evaluation.

Demo:

```
python examples/week4_leak_demo.py
```

Output (80 synthetic assets with a planted momentum edge, 4 years, seed 0):

| Signal | Naive Sharpe | Audit | Hard checks failed | Warnings |
|---|---|---|---|---|
| clean: 6-1 momentum | 2.05 | PASS | none | none |
| clean: 1-day reversal | -0.14 | PASS | none | none |
| leak: tomorrow's return | 198.29 | FAIL | perturbation, truncation | IC, lag fragility |
| leak: centered 21-day window | 28.13 | FAIL | perturbation, truncation | IC |
| leak: full-sample z-score | -1.04 | FAIL | perturbation, truncation | none |
| leak: backward fill | 3.26 | FAIL | truncation | none |
| leak: month-end value all month | 9.86 | FAIL | perturbation, truncation | none |

The backward fill row is the dangerous kind: a believable Sharpe of 3.26 against 2.05 for the clean version of the same signal, with no soft check firing. Only the truncation test sees it. The full-sample z-score shows that leaks do not always help; they make the backtest wrong in either direction.

**Known limitation.** A discrete choice made with hindsight, such as picking a signal's sign from its full-sample correlation with future returns, is invisible to any data-level check when the edge is strong: every truncated sample picks the same sign, so the output never changes. A test documents this. The defence is the ledger: trying both signs is two trials, and the deflated statistics account for it.

## Week 5 design: the agent gateway

`alphaledger/gateway/` is the only door between an agent and the data. It is exposed as an MCP server, so Claude Desktop, Claude Code or any MCP client can use it.

**A safe signal language instead of code.** The agent writes expressions such as `rank(momentum(252, 21)) - rank(ts_std(returns, 63))`. They are parsed with a whitelist of syntax, names and functions: no imports, no attribute access, no indexing, no lambdas, bounded size. Every operator looks only backwards (there is no negative delay), and the test suite verifies each one with the Week 4 truncation and perturbation checks. The leak audit still runs on every trial as a second line of defence.

**No backtest without a ledger entry.** `test_signal` compiles, registers, audits, backtests and records in one step. Malformed expressions are rejected without counting as trials because nothing was measured. A repeat of an earlier expression (even with different spacing) is logged and linked but not counted twice.

**Deflated numbers in every reply.** Each result carries the trial count so far, the hurdle Sharpe for that many trials and the deflated Sharpe ratio next to the naive stats.

**A hidden holdout with a budget.** The last 20% of dates are invisible. `evaluate_holdout` opens them for one hypothesis at a time, at most 3 times by default. Each use is written to the ledger, so the budget survives restarts and cannot be reset.

| MCP tool | Purpose |
|---|---|
| `describe_data` | Universe, in-sample dates, holdout status, portfolio rules, trial count |
| `expression_language` | Reference for the signal language |
| `test_signal` | Register, audit and backtest one expression |
| `leaderboard` | Top trials with deflated Sharpe, FDR significance, haircut Sharpe, PBO |
| `lineage` | Ancestors and children of a hypothesis |
| `verify_ledger` | Check the hash chain and return the head hash |
| `evaluate_holdout` | Open the hidden period for one hypothesis, within budget |

Run it:

```
pip install -e .[mcp]
alphaledger-mcp --ledger research.db --synthetic             # demo market
alphaledger-mcp --ledger research.db --returns-csv returns.csv  # dates in column 1, one column of daily returns per asset
```

Add it to Claude Code:

```
claude mcp add alphaledger -- alphaledger-mcp --ledger /full/path/research.db --synthetic
```

Or to Claude Desktop in `claude_desktop_config.json` under `mcpServers`, with command `alphaledger-mcp` and the same arguments.

**Tests** cover every operator for causality, 20 hostile or malformed expressions, the hidden holdout, budget enforcement across restarts, duplicate handling, and real tool calls both in process and over stdio through the MCP client. Writing them caught two real bugs: MCP runs tools in worker threads, which SQLite rejected (fixed with a cross-thread connection and serialized writes), and repeated expressions crashed the response (fixed).

## License

MIT
