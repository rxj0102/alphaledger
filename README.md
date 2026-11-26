# alphaledger

A research integrity layer for AI-generated alpha.

AI agents can propose and backtest hundreds of trading signals in an afternoon. The best of those backtests will look great by luck alone. alphaledger sits between the agent and the data, logs every hypothesis it tries (including the failures), and reports performance adjusted for how many things were tested, along with checks for look-ahead and data leakage.

## Quickstart

```
pip install -e .[dev]
pytest                                            # 135 tests
python examples/week6_case_study.py               # blind agent vs alphaledger, about 8 minutes
alphaledger-mcp --ledger research.db --synthetic  # point Claude or any MCP client at the gateway
alphaledger-report research.db -o report.html     # one-file report of any ledger
alphaledger-dashboard research.db                 # Streamlit dashboard
```

Headline (Week 6 case study): a blind agent that tried 200 signals on a pure noise market would have shipped one with an in-sample Sharpe of 0.68 that lost money on the hidden holdout. On a market with a real edge, the same 200 trials made 36 look significant, yet after accounting for the search none could be certified. Six prior-driven trials on a comparable market kept enough power for 4 to survive correction.

## Status

| Week | Component | Status |
|---|---|---|
| 1 | Point-in-time data layer, vectorized backtester, momentum sanity check | Done |
| 2 | Append-only, hash-chained hypothesis ledger with lineage | Done |
| 3 | Stats engine: Deflated Sharpe Ratio, PBO via CSCV, FDR control, t-stat haircuts | Done |
| 4 | Leakage detectors tested against planted leaks | Done |
| 5 | MCP gateway so an agent can only reach data and backtests through the ledger | Done |
| 6 | Dashboard, HTML report and case study on agent-generated signals | Done |

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
Published head hash: <16 hex characters>...
After deleting entry 24: verifies=False, first bad entry=24, reason: missing entry 24
```

The head hash is different on every run because each entry's timestamp is part of its hash. Everything else is deterministic.

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
- Significance is one-sided: the question is whether a trial has positive skill. A high-turnover signal eaten by costs can have a hugely significant negative Sharpe, and that is not a discovery. (This was two-sided until the Week 6 case study showed losers being counted as discoveries.)
- Two deflated Sharpe ratios are reported. `dsr` follows the paper and uses the spread of Sharpe ratios across the trials themselves. `dsr_analytic` uses the spread a zero-skill Sharpe estimate has from sampling alone, 1 / (T - 1). For independent noise trials the two hurdles agree (tested), but when trials differ for systematic reasons, such as costs or sign flips of one idea, the empirical spread and the paper's hurdle grow far beyond what luck produces (also tested). The Week 6 case study shows why both are needed.

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

## Week 6: dashboard, report and case study

**Report and dashboard.** `alphaledger-report research.db -o report.html` writes one self-contained HTML file (no external assets, light and dark mode): summary tiles, a scatter of naive Sharpe against deflated Sharpe ratio with hover details, the leaderboard and the ledger verification result. `alphaledger-dashboard research.db` opens the same view in Streamlit with a lineage browser. Both are tested, the dashboard with Streamlit's headless AppTest.

### Case study 1: a blind agent proposing 200 signals

`examples/week6_case_study.py` plays an agent that proposes random combinations of common quant building blocks (momentum, reversal, low volatility, trend z-scores, range position, and weighted blends of two), as an LLM told to find alpha might. Every proposal goes through the gateway: hidden 20% holdout, leak audit, ledger, 5 bps costs. It runs on two synthetic markets of 100 assets and 8 years, so the truth is known: one is pure noise, the other has a real momentum-type edge. A naive researcher then picks the best in-sample Sharpe and checks it on the holdout.

| | Noise market | Planted edge market |
|---|---|---|
| Trials counted (237 proposals, duplicates not counted) | 200 | 200 |
| Naive pick: in-sample Sharpe | 0.68 | 1.27 |
| Raw p < 0.05, one-sided | 1 | 36 |
| Significant after BH | 0 | 0 |
| Hurdle Sharpe: trial spread / analytic | 1.89 / 1.12 | 2.43 / 1.12 |
| Best DSR: trial spread / analytic | 0.001 / 0.15 | 0.002 / 0.62 |
| PBO | 0.33 | 0.17 |
| Holm haircut Sharpe of the naive pick | 0.00 | 0.33 |
| Holdout Sharpe of the naive pick | -1.28 | 0.53 |

What it shows:
- On noise, a naive researcher would ship a signal with an in-sample Sharpe of 0.68 that lost money on the holdout (-1.28). alphaledger certified nothing.
- On the planted market, 36 trials look significant before correction, and the naive pick really does have an edge (holdout 0.53). alphaledger still certified nothing: after 200 trials on 6.4 years of data, an edge of about Sharpe 1 cannot be told apart from the best of 200 lucky draws. That is the real cost of blind search. It does not only produce false discoveries; it destroys the power to confirm true ones.
- The trial-spread hurdle from the original DSR paper (2.43) is more than twice the analytic one (1.12) because these trials differ systematically: high-turnover variants are eaten by costs and blends flip the sign of real signals. That spread is not luck, so it should not raise the bar. Both versions are reported.

### Case study 2: Claude as the agent

`examples/week6_claude_session.py` replays a session in which Claude used the gateway interactively, one decision at a time, on a third synthetic market (seed 303). Claude started from documented anomalies (12-1 momentum, short-term reversal, low volatility), refined the idea that worked with lineage links, stopped after 6 trials, and chose the holdout candidate before opening the holdout. Claude knew the generator plants a continuation edge, so this demonstrates the workflow, not an unbiased discovery.

| Trial | Expression | Naive Sharpe | DSR trial spread | DSR analytic |
|---|---|---|---|---|
| 1 | `rank(momentum(252, 21))` | 0.84 | 0.025 | 0.738 |
| 2 | `rank(-ts_sum(returns, 5))` | -2.15 | 0.000 | 0.000 |
| 3 | `rank(-ts_std(returns, 63))` | -0.43 | 0.000 | 0.009 |
| 4 | `rank(momentum(126, 5))` | 1.11 | 0.082 | 0.917 |
| 5 | `rank(ts_zscore(price, 252))` | 0.99 | 0.055 | 0.839 |
| 6 | blend of trials 4 and 5 | 1.00 | 0.057 | 0.843 |

(DSR values from the final leaderboard, after all 6 trials.) With 6 trials the analytic hurdle is 0.56 instead of 1.12 at 200, so the same kind of edge gets an analytic DSR of 0.92 rather than 0.62, and 4 trials survive BH. Few, well-motivated hypotheses keep the statistical power that a blind sweep spends.

The weekly reversal trial, which lost money mostly to costs at 32% daily turnover, is what drives the trial-spread hurdle to 1.68 after only six trials: one systematic loser is enough to make the paper's version of DSR almost impossible to pass.

**The holdout came back at -1.12**, even though the edge is real and the generator shows it did not weaken. Running the same strategy on 30 other seeds of the same market model gives a holdout Sharpe of 1.06 on average (sd 0.70, lowest -0.25), so this was a 3.1 sd bad draw. The seed was not changed after seeing it. A 1.6 year holdout is a noisy judge too, which is why the holdout budget exists and why one bad holdout should be read alongside the in-sample evidence rather than as a verdict.

### Bugs the case study found (all fixed, with regression tests)
- A signal whose two halves cancel produced a flat book and crashed the gateway. Flat results are now recorded with status degenerate: counted as a trial, excluded from evaluation.
- Two-sided tests counted cost-crushed losers as discoveries. Evaluation is now one-sided.
- With pandas 3, a missing duplicate link was stored as NaN, which is truthy, so returns of different trials collided in one column whenever the ledger held a duplicate. Links are now kept as None.

## Related work

The closest work is Gençay (2026), *What survives honest evaluation? Leakage-safe, search-aware assessment of LLM-driven trading strategy discovery*, arXiv:2608.27734. It builds leakage safety and search-aware deflation into an LLM strategy discovery system, records every evaluation the search performs, and finds that no LLM-discovered strategy is certified at the agent's own trial count. It also reports that a planted look-ahead oracle survives deflation at a Sharpe of 35, the same reason alphaledger treats leakage as a structural check rather than a statistical one (Week 4).

alphaledger is built as a reusable layer rather than a study: a tamper-evident ledger, a causal expression language and an MCP gateway that any agent can be pointed at, with a hidden holdout budget. The case studies above reach a conclusion consistent with that paper, and add the observation that the paper's trial-spread variance can overstate the hurdle when trials differ systematically.

Other related work:
- *Hard to Fool: A Bounded LLM Research Loop for Quantitative Finance, and What Two Dozen Campaigns Found*, arXiv:2611.06607.
- Li et al., *Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents*, arXiv:2510.07920, on leakage through what the language model itself memorized.
- *From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets* (KTD-Fin), arXiv:2605.28359.

The statistical methods follow Bailey and Lopez de Prado (2012, 2014), Bailey, Borwein, Lopez de Prado and Zhu (2017), Benjamini and Hochberg (1995), Benjamini and Yekutieli (2001), Holm (1979) and Harvey and Liu (2015).

## Limitations

- All headline results use synthetic markets, chosen because the truth is known. The real data loaders are tested on fixtures only; the network fetchers have not been run in this repo's test environment.
- yfinance universes are survivorship-biased, and the survivorship check will say so.
- The language model's own memory of market history is a leak no data-level check can see (see Li et al. above). Synthetic or anonymized data avoids it.
- A discrete choice made with hindsight can pass the leak audit when the edge is strong (see Week 4); only the trial count catches it.
- The analytic DSR ignores correlation between trials, so it is conservative when trials are highly correlated; the trial-spread DSR can be far too strict when trials differ systematically. Neither is the single right answer.

## License

MIT
