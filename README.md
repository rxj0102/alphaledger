# alphaledger

A research integrity layer for AI-generated alpha.

AI agents can propose and backtest hundreds of trading signals in an afternoon. The best of those backtests will look great by luck alone. alphaledger sits between the agent and the data, logs every hypothesis it tries (including the failures), and reports performance adjusted for how many things were tested, along with checks for look-ahead and data leakage.

## Quickstart

```
pip install -e .[dev]
pytest                                            # 173 tests, plus 4 live data checks (ALPHALEDGER_NETWORK_TESTS=1)
python examples/controlled_study.py               # blind vs pre-registered agents on 40 markets, about 15 minutes
python examples/week6_case_study.py               # one 200-trial blind agent in detail, about 10 minutes
python examples/real_data_study.py                # both agents on 49 real US industries, about 10 minutes
alphaledger-mcp --ledger research.db --synthetic  # point Claude or any MCP client at the gateway
alphaledger-report research.db -o report.html     # one-file report of any ledger
alphaledger-dashboard research.db                 # Streamlit dashboard
```

Headline (controlled study, 20 synthetic markets with a real edge and 20 of pure noise, both agents on identical markets): an agent that tried 100 random signals got its best pick certified on 50% of the markets that had a real edge; an agent that tested 6 hypotheses written down in advance got it certified on 95%. Neither was certified on any of the 20 noise markets. Blind search does not only risk false discoveries; it spends the statistical power needed to confirm true ones. The 6 pre-registered hypotheses were the obvious literature priors and happen to match the planted edge, so this is the case where priors are right.

On real data (49 US industry portfolios, 1990 to 2026), neither agent's pick could be certified. The pre-registered pick, 12-1 industry momentum, had an in-sample Sharpe of 0.42 and 0.29 on the hidden 2019 to 2026 holdout. The blind agent's pick looked better in sample (0.46) and lost money on the holdout (-0.13). One real history is one sample, so this is an illustration, not evidence.

## Status

| Week | Component | Status |
|---|---|---|
| 1 | Point-in-time data layer, vectorized backtester, momentum sanity check | Done |
| 2 | Append-only, hash-chained hypothesis ledger with lineage | Done |
| 3 | Stats engine: Deflated Sharpe Ratio, PBO via CSCV, FDR control, t-stat haircuts | Done |
| 4 | Leakage detectors tested against planted leaks | Done |
| 5 | MCP gateway so an agent can only reach data and backtests through the ledger | Done |
| 6 | Dashboard, HTML report and case study on agent-generated signals | Done |
| Review 1 | Signed and anchored ledger, locked trial accounting, effective trials, point-in-time variables, next-day execution, delistings, controlled study | Done |
| Review 2 | Public-key signing for auditors, registry that recognises trimmed copies, honest integrity status in reports, post-holdout marking, vendor-lag gaps, survivor-free declaration | Done |

## Week 1 design

**Point-in-time store** (`alphaledger/data/pit.py`). Every observation has a `period` (what it describes) and an `available_at` (when it could first be known). Queries are made as of a decision time and only see rows with `available_at` at or before it. Revisions are stored as new vintages, so a later revision never leaks into an earlier decision. `panel()` builds a daily dates by entities input with an optional extra lag.

**Backtester** (`alphaledger/backtest/engine.py`). Weights decided from the close of day t with `lag=1` earn the return from the close of day t to the close of day t+1, which means trading in the same closing auction the signal was computed from. That is optimistic, especially for short-horizon signals. `lag=2` trades at the next close and is the conservative setting; the research gateway uses it by default. A zero lag raises `LookAheadError` unless explicitly allowed. Turnover based transaction costs, monthly rebalancing support, and standard metrics (Sharpe, t-stat, max drawdown, hit rate, turnover).

**Missing returns.** A gap inside an asset's history (a halt) earns zero. A missing return after an asset's last valid one means it stopped trading: a position held at that point takes a one-off delisting return of -30% (close to the average performance-related delisting return reported by Shumway, 1997) and is then closed. Treating those returns as zero would silently hide the losses. Only a gap of at least 5 trading days before the end of the sample counts as a delisting; a shorter trailing gap is usually a data vendor lag on the latest days and earns zero (this was a bug found in review: one ticker missing yesterday's price took a -30% hit). `delist_return=None` switches delistings off.

**Free data loaders** (`alphaledger/data/sources.py`), all run live on 26 Nov 2026 (`ALPHALEDGER_NETWORK_TESTS=1 pytest tests/test_sources.py`; off by default so CI does not depend on outside services):
- Ken French daily factors. The library is updated monthly with a lag, so a day's factor return is treated as known only at the end of the following month. Checked against the live file: the copy downloaded on 26 Nov 2026 was built from the September 2026 CRSP database and dated 26 Oct 2026.
- Ken French 49 industry portfolios, daily, as a free asset universe. They are rebuilt each June from the stocks alive at the time, so they have no survivorship bias. Missing codes (-99.99, -999) become NaN; the data is complete from 1990.
- FRED vintages through the official API with the full real-time period, so each value enters at its release date. Needs a free FRED API key; this path is tested only against fixtures because no key was available.
- FRED vintages without a key, from monthly ALFRED snapshots. Each value enters at the first snapshot that shows it, so it can be up to a month late but never early. Tested live: in a snapshot from early January 2020, December 2019 industrial production is correctly absent.
- yfinance adjusted closes, tested live. Universes built from yfinance are survivorship-biased because delisted tickers are missing; any result using them says so. Yahoo rate-limits aggressively (a 429 is common); yfinance retries without a session cookie.

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
- a held stock that stops trading takes the delisting return once and is closed, and a mid-history gap does not
- momentum recovers the planted edge, finds nothing in noise, and loses money when the sign is flipped
- changing today's and future returns does not change today's momentum signal

Planting bugs confirms the tests catch them: removing the momentum skip, or removing the execution lag, makes the suite fail.

## Run the sanity check

```
python examples/week1_momentum.py            # synthetic market with planted momentum
python examples/week1_momentum.py --noise    # synthetic pure noise
python examples/week1_momentum.py --yfinance AAPL MSFT NVDA JPM XOM   # real prices, needs pip install yfinance
```

On real prices for 24 large US stocks from 2006 to 25 Nov 2026 (today's survivors, so biased upwards), the same 12-1 momentum strategy had a Sharpe of -0.06 (t-stat -0.27, max drawdown -54%), against 2.83 on the synthetic market built to reward it. With about 5 stocks per leg this is a check that the real-data path works end to end, not a test of momentum.

Output from the two synthetic runs (200 assets, seed 0, 12-1 momentum, quintile long short, monthly rebalance, lag 1, 5 bps costs):

| Market | Sharpe | t-stat | Max drawdown |
|---|---|---|---|
| Planted momentum | 2.83 | 7.47 | -3.4% |
| Pure noise | 0.14 | 0.38 | -5.9% |

These are synthetic markets built to test the pipeline. They say nothing about real momentum returns.

## Week 2 design: the hypothesis ledger

`alphaledger/ledger/` stores every hypothesis before it is tested and every outcome after, including runs that fail or raise. It uses SQLite from the Python standard library, so there is no extra dependency.

**Tamper evidence, and its limits.** Each entry stores the hash of the previous entry, and its own hash covers its content plus that link. Careless edits, deletions or reordering break the chain and `verify()` reports the first bad entry. SQLite triggers reject UPDATE and DELETE, but anyone with the file can drop them. The hash chain alone does not stop a deliberate rewrite either: a test shows that deleting a losing trial, inflating a winner and recomputing every hash still verifies. Two protections close that gap:

- **Signing.** Every entry hash can be signed in one of two ways. A rewrite without the signing secret cannot produce valid signatures and `verify()` rejects it (tested with the same attack).
  - `Ledger(path, key=...)`: a shared secret (HMAC-SHA256), for example the `ALPHALEDGER_KEY` environment variable of the server process. Only someone with the same secret can verify, so it proves integrity to the operator but not to anyone else.
  - `Ledger(path, signer=...)`: an Ed25519 private key (`alphaledger-keygen operator.pem operator.pub`, then `ALPHALEDGER_SIGNING_KEY_FILE` for the server). Anyone with the public key can verify, with `Ledger(path, verify_key=...)` or `alphaledger-report --public-key`, and a ledger opened with only the public key is read-only. This is what lets an investment committee or an auditor check the record without being able to forge it (tested, including over stdio).
  - Either way the secret belongs to the operator, never to the agent.
- **Anchoring.** `anchor(path, name)` appends the current head to an external, append-only file (a git repository that gets pushed, another machine, a write-once bucket). `verify_anchors()` then detects any rewrite or truncation up to an anchored entry, with or without a key (tested). The gateway anchors after every write when given an anchor file.

**Trial counting.** `n_trials()` counts distinct tested portfolios. Failed runs count. Re-running an identical spec (even with keys in a different order) is logged but not counted again, and neither is a different expression that produces exactly the same returns, such as `x`, `x * 1` or `rank(rank(x))`. Correlated but different trials still count separately here; the stats engine estimates the effective number of independent trials.

**Honest status in reports.** A chain that verifies without the key only shows the file is internally consistent. Reports and the dashboard say verified only when signatures were actually checked; otherwise they say not proven, with the reason (unsigned, or signed but no key given). Review found the earlier report printed verified for a ledger whose signatures had been forged, because it was opened without the key.

**Scale.** Queries are served from an in-memory index that is synced incrementally from the database, so a test does not reparse every stored return series. Another process writing to the same file is picked up on the next query (tested).

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

Removing the content hash check or the duplicate detection from the code makes the test suite fail.

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
- Three deflated Sharpe ratios are reported:
  - `dsr` follows the paper and uses the spread of Sharpe ratios across the trials themselves. When trials differ for systematic reasons, such as costs or sign flips of one idea, that spread and the hurdle grow far beyond what luck produces (tested).
  - `dsr_analytic` uses the spread a zero-skill Sharpe estimate has from sampling alone, 1 / (T - 1), and treats every trial as independent. For independent noise trials it agrees with the paper's hurdle (tested); for 40 tweaks of one idea it overstates N.
  - `dsr_effective`, the headline number, is the analytic version with N replaced by the effective number of independent trials, with failed, flat and leaky trials each counted as independent. The effective number is the larger of two estimates: the spectral participation ratio, N squared over the sum of squared correlations (bias-corrected for sampling noise), and the number of clusters from average-linkage clustering at a correlation of 0.5. These follow the spectral and clustering approaches Lopez de Prado suggests. The spectral estimate alone can be gamed: padding a ledger with 100 copies of one trial pushed it from 3 to about 1, which would lower the bar for an unrelated lucky trial. The cluster count is unaffected by copies, so taking the larger blocks that (tested).

**Checked against published numbers:**
- The numerical example in Bailey and Lopez de Prado (2014): annual SR 2.5, T = 1250, skew -3, kurtosis 10, N = 100, annualized trial variance 0.5, reported as DSR 0.9004. alphaledger gives 0.9004. Caveat: these inputs and the target value were taken from memory of the paper; the paper's text could not be retrieved in the build environment to confirm them. An exact four-digit match is strong but not conclusive evidence.
- A monthly worked example from a tutorial site (quantt.co.uk DSR guide, not a primary source): hurdle 0.3797 and DSR 0.624 reproduced.
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
| Hurdle Sharpe for 60 trials (trial-spread version) | 0.47 | 0.70 |
| Deflated Sharpe Ratio of best (trial-spread version) | 0.540 | 1.000 |
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

**No backtest without a ledger entry.** `test_signal` compiles, registers, audits, backtests and records in one step. Malformed expressions are rejected without counting as trials because nothing was measured. A repeat of an earlier expression, or any expression giving the same portfolio, is logged and linked but not counted twice.

**Next-day execution by default.** The gateway trades a signal computed from day t's close at day t+1's close (`--lag 2`). `--lag 1` is available but optimistic.

**Point-in-time variables.** A `PITStore` can be attached with `pit_fields`, for example FRED series with `--fred INDPRO`. Each field becomes an expression variable that holds, on every date, only the latest value released by then, so `rank(momentum(126, 5)) * sign(indpro)` cannot see a revision or a release from the future. This matters because the truncation check only truncates returns: a leak coming in through external data would be invisible to it, so for external data the protection has to be structural. A test confirms each value used was released on or before its date. The FRED path makes network calls and has only been tested against fixtures.

**Deflated numbers in every reply.** Each result carries the trial count, the effective number of independent trials, and the three deflated Sharpe ratios next to the naive stats. Data-level warnings such as survivorship are reported once in `describe_data` instead of on every trial.

**A hidden holdout with a budget.** The last 20% of dates are invisible. `evaluate_holdout` opens them for one hypothesis at a time, at most 3 times by default. Each use is written to the ledger as a structured entry, so the budget survives restarts. Once the holdout has been opened, every later test is flagged `after_holdout_look` in its reply and on the leaderboard, because it may be shaped by what the holdout showed and its own holdout result would no longer be out of sample.

**Survivorship.** The survivorship warning is a heuristic that cannot tell a universe that never delists by construction (industry portfolios, indices) from today's survivors. The operator can declare a universe survivor-free (`--survivor-free`); the declaration is recorded in the ledger binding and shown in `describe_data`.

**Input limits.** Expressions are capped at 2,000 characters and 80 syntax nodes, rationales at 4,000 characters.

**What the gateway can and cannot enforce.** It controls what passes through it. It cannot stop an agent that has its own copy of the data from backtesting elsewhere; that has to come from deployment: the operator runs the server, holds the data and the signing key, and the agent only gets the MCP connection. Within that setup:
- the ledger is bound to the dataset and configuration on first use, so it cannot be reopened with a bigger holdout budget, a different holdout, other costs, another lag or other data fields (tested);
- with `--registry`, a dataset can only ever be served from one ledger, so starting a fresh ledger file does not reset the trial count or the holdout budget. The registry recognises a dataset by its exact hash and by a MinHash sketch of its (date, asset, value) cells, so dropping a day, cutting the history or dropping assets still counts as the same data (tested; review found that trimming one day used to buy a fresh ledger). Values are compared to 6 decimals: a dataset deliberately altered by more than that is treated as new, so the operator must control which data is loaded;
- with `ALPHALEDGER_KEY` set and `--anchors`, history cannot be rewritten without detection (tested over stdio).

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

Locked-down deployment, run by the operator:

```
export ALPHALEDGER_KEY=...   # never given to the agent
alphaledger-mcp --ledger /srv/research.db --returns-csv returns.csv \
    --registry /srv/registry.jsonl --anchors /srv/anchors/anchors.jsonl
```

Add it to Claude Code:

```
claude mcp add alphaledger -- alphaledger-mcp --ledger /full/path/research.db --synthetic
```

Or to Claude Desktop in `claude_desktop_config.json` under `mcpServers`, with command `alphaledger-mcp` and the same arguments.

**Tests** cover every operator for causality, 20 hostile or malformed expressions, the hidden holdout, budget enforcement across restarts and fresh ledgers, configuration binding, duplicate handling, point-in-time variables, and real tool calls both in process and over stdio through the MCP client, including the signed and anchored deployment. Writing them caught two real bugs: MCP runs tools in worker threads, which SQLite rejected (fixed with a cross-thread connection and serialized writes), and repeated expressions crashed the response (fixed).

## Week 6: dashboard, report and case study

**Report and dashboard.** `alphaledger-report research.db -o report.html` writes one self-contained HTML file (no external assets, light and dark mode): summary tiles, a scatter of naive Sharpe against deflated Sharpe ratio with hover details, the leaderboard and the ledger verification result. `alphaledger-dashboard research.db` opens the same view in Streamlit with a lineage browser. Both are tested, the dashboard with Streamlit's headless AppTest.

### Controlled study: blind search against pre-registered hypotheses

`examples/controlled_study.py` puts two agents on exactly the same markets: 20 synthetic markets (100 assets, 8 years) with a planted continuation edge, and 20 of pure noise. Both go through the same gateway (hidden 20% holdout, next-day execution, 5 bps costs) and use the same selection rule: take the best in-sample trial, then open the holdout once.

- Blind agent: 100 random expressions from a grammar of momentum, reversal, low volatility, trend z-scores, range position and blends of two, as an LLM told to find alpha might produce.
- Pre-registered agent: 6 expressions fixed in the script before any data was generated (12-1 momentum, weekly reversal, low volatility, 6-1 momentum, a 1-year trend z-score, and a blend of the last two).

Certified means the pick's deflated Sharpe ratio with effective trials is at least 0.95.

| | Planted, blind | Planted, pre-registered | Noise, blind | Noise, pre-registered |
|---|---|---|---|---|
| Trials (effective) | 100 (37.5) | 6 (3.6) | 100 (37.8) | 6 (3.9) |
| Pick certified | 50% | 95% | 0% | 0% |
| Any trial significant after BH | 75% | 95% | 0% | 5% |
| Raw p < 0.05, one-sided, per run | 18.2 | 3.3 | 0.7 | 0.1 |
| Pick in-sample Sharpe, mean | 1.57 | 1.49 | 0.70 | 0.37 |
| Pick holdout Sharpe, mean | 1.03 | 1.33 | -0.38 | -0.21 |
| Pick holdout positive | 95% | 90% | 25% | 35% |

What it shows:
- **No false certifications** in 40 noise runs, for either agent, even though blind search found raw p < 0.05 trials and an average in-sample pick Sharpe of 0.70 that lost money out of sample.
- **Blind search costs power.** With a real edge on every planted market, the blind agent's pick was certified only half the time, against 95% for the pre-registered agent. Its pick also shrank more out of sample (1.57 to 1.03, against 1.49 to 1.33): selecting the best of many trials selects luck along with the edge.
- **Correlation matters.** 100 blind trials were worth about 38 independent ones; counting them as 100 would have raised the bar further.

Caveats: 20 markets per cell, so rates carry sampling error (95% against 50% is clear; small differences are not). The pre-registered list matches the planted edge; with wrong priors, discipline does not create power. The leak audit was off in this study for speed, which is safe here because every expression comes from the causal language.

### Real data: 49 US industry portfolios

`examples/real_data_study.py` runs the same two agents on Kenneth French's 49 value-weighted industry portfolios, daily, from January 1990. The in-sample period is 1990 to May 2019; the hidden holdout is May 2019 to September 2026 (1,851 days). Same gateway rules: next-day execution, 5 bps costs, leak audit on, best in-sample trial opens the holdout once. HTML reports for both agents are in `case_study/`.

| | Pre-registered (6 trials) | Blind (100 trials) |
|---|---|---|
| Effective independent trials | 3.0 | 20.0 |
| Raw p < 0.05, one-sided | 3 | 7 |
| Significant after BH | 0 | 0 |
| Pick | `rank(momentum(252, 21))` | `0.7 * rank(ts_zscore(price, 252)) - 0.3 * rank(range position over 21 days)`, q 0.3 |
| Pick in-sample Sharpe | 0.42 | 0.46 |
| Hurdle Sharpe (effective trials) | 0.16 | 0.35 |
| Pick DSR: effective / analytic / trial spread | 0.91 / 0.82 / 0.79 | 0.71 / 0.47 / 0.00 |
| Certified (DSR effective at least 0.95) | no | no |
| PBO | 0.24 | 0.04 |
| Pick holdout Sharpe, 2019 to 2026 | 0.29 | -0.13 |

The blind agent found a pick that looked slightly better in sample and lost money out of sample. The pre-registered pick is classic 12-1 industry momentum (Moskowitz and Grinblatt, 1999); it stayed positive out of sample but was too weak over 29 years to clear the hurdle with confidence, so alphaledger declined to certify it. The blind agent's low PBO is a reminder that PBO measures how stable the ranking of trials is, not whether the winner is real. This is one history of one market: it illustrates the workflow on real, survivor-free data and should not be read as evidence about either agent.

### Case study 1: one blind agent in detail

`examples/week6_case_study.py` runs the blind agent for 200 trials on one noise market and one planted market, with the leak audit on, and writes an HTML report for each to `case_study/`.

| | Noise market | Planted edge market |
|---|---|---|
| Trials counted (240 proposals; repeats and identical portfolios not counted) | 200 | 200 |
| Effective independent trials | 60.0 | 62.0 |
| Naive pick: in-sample Sharpe | 0.78 | 1.39 |
| Raw p < 0.05, one-sided | 2 | 41 |
| Significant after BH | 0 | 0 |
| Hurdle Sharpe: effective / analytic / trial spread | 0.95 / 1.12 / 1.86 | 0.95 / 1.12 / 2.41 |
| Best DSR: effective / analytic / trial spread | 0.35 / 0.21 / 0.003 | 0.84 / 0.72 / 0.007 |
| PBO | 0.48 | 0.14 |
| Holdout Sharpe of the naive pick | -1.37 | 0.74 |

On noise the naive pick (0.78 in sample) lost money on the holdout and nothing was certified. On the planted market 41 trials looked significant before correction and the pick is real (holdout 0.74), but at 200 trials even the effective-trials DSR stops at 0.84. The paper's trial-spread hurdle (2.41) is far above the others because high-turnover variants eaten by costs and sign-flipped blends widen the spread of trial Sharpes for reasons that are not luck.

### Case study 2: Claude as the agent

`examples/week6_claude_session.py` replays a session in which Claude chose each test after seeing the previous results, on a third synthetic market (seed 303). Claude drove the gateway from Python rather than over MCP (same code path). The session ran when the gateway traded at the same close, so the replay keeps lag 1 to reproduce the decisions as they were made. Claude started from documented anomalies, refined the idea that worked with lineage links, stopped after 6 trials and chose the holdout candidate before opening the holdout. Claude knew the generator plants a continuation edge, so this demonstrates the workflow, not an unbiased discovery; the controlled study above is the fair comparison.

| Trial | Expression | Naive Sharpe | DSR effective | DSR analytic | DSR trial spread |
|---|---|---|---|---|---|
| 1 | `rank(momentum(252, 21))` | 0.84 | 0.860 | 0.738 | 0.025 |
| 2 | `rank(-ts_sum(returns, 5))` | -2.15 | 0.000 | 0.000 | 0.000 |
| 3 | `rank(-ts_std(returns, 63))` | -0.43 | 0.028 | 0.009 | 0.000 |
| 4 | `rank(momentum(126, 5))` | 1.11 | 0.967 | 0.917 | 0.082 |
| 5 | `rank(ts_zscore(price, 252))` | 0.99 | 0.925 | 0.839 | 0.055 |
| 6 | blend of trials 4 and 5 | 1.00 | 0.927 | 0.843 | 0.057 |

(Values from the final leaderboard: 6 trials, 3.2 effective.) The weekly reversal trial lost money mostly to costs at 32% daily turnover, and that one systematic loser is enough to push the paper's trial-spread hurdle to 1.68 after six trials.

**The holdout came back at -1.12**, even though the edge is real and the generator shows it did not weaken. The same strategy on 30 other seeds of the same market model gives a holdout Sharpe of 1.06 on average (sd 0.70, lowest -0.25), so this was a 3.1 sd bad draw. The seed was not changed after seeing it. A 1.6 year holdout is a noisy judge too.

### Bugs found by the case studies and the review (all fixed, with regression tests)
- A signal whose two halves cancel produced a flat book and crashed the gateway. Flat results are now recorded with status degenerate: counted as a trial, excluded from evaluation.
- Two-sided tests counted cost-crushed losers as discoveries. Evaluation is now one-sided.
- With pandas 3, a missing duplicate link was stored as NaN, which is truthy, so returns of different trials collided in one column whenever the ledger held a duplicate. Links are now kept as None.
- The hash chain could be rewritten end to end without detection; signing and anchoring added.
- A fresh ledger file reset the trial count and holdout budget; configuration binding and the registry added.
- Algebraically identical expressions counted as separate trials; portfolio-level deduplication added.
- The spectral effective-trials estimate could be lowered by padding the ledger with copies; it is now combined with a cluster count.
- Execution was same-close by default in the gateway; it is now next-day.
- Missing returns after a delisting counted as zero; a delisting return is now applied.
- Ken French factors were marked available the same day; now the end of the following month.

## Related work

The closest work is Gençay (2026), *What survives honest evaluation? Leakage-safe, search-aware assessment of LLM-driven trading strategy discovery*, arXiv:2608.27734. It builds leakage safety and search-aware deflation into an LLM strategy discovery system, records every evaluation the search performs, and finds that no LLM-discovered strategy is certified at the agent's own trial count. It also reports that a planted look-ahead oracle survives deflation at a Sharpe of 35, the same reason alphaledger treats leakage as a structural check rather than a statistical one (Week 4).

alphaledger is built as a reusable layer rather than a study: a tamper-evident ledger, a causal expression language and an MCP gateway that any agent can be pointed at, with a hidden holdout budget. The case studies above reach a conclusion consistent with that paper, and add the observation that the paper's trial-spread variance can overstate the hurdle when trials differ systematically.

Other related work:
- *Hard to Fool: A Bounded LLM Research Loop for Quantitative Finance, and What Two Dozen Campaigns Found*, arXiv:2611.06607.
- Li et al., *Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents*, arXiv:2510.07920, on leakage through what the language model itself memorized.
- *From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets* (KTD-Fin), arXiv:2605.28359.

The statistical methods follow Bailey and Lopez de Prado (2012, 2014), Bailey, Borwein, Lopez de Prado and Zhu (2017), Benjamini and Hochberg (1995), Benjamini and Yekutieli (2001), Holm (1979) and Harvey and Liu (2015).

## Limitations

- The controlled results use synthetic markets, chosen because the truth is known, and the same author wrote the generator, the agents' grammar and the studies. The one real-data study is a single history of one market. The FRED API path is untested live because no API key was available; the keyless ALFRED path, Ken French and yfinance were tested live.
- The gateway can only account for what passes through it. Its guarantees depend on deployment: the operator holds the data and the key, and the agent only gets the MCP connection. An agent with its own copy of the data can always backtest elsewhere.
- The ledger's timestamps come from the server clock and are not independently attested; anchoring to an external system fixes order and content, not wall-clock time.
- The backtester models linear costs only: no borrow costs, market impact or capacity limits. The -30% delisting return is a single assumption, not security-specific.
- The leak audit samples cut dates rather than proving causality, so a leak confined to a few dates can slip through. The causal expression language is the main defence; the audit is a second line.
- yfinance universes are survivorship-biased, and the survivorship check will say so.
- Ken French data is not point-in-time. Each monthly release recomputes the full history from the latest CRSP database and current industry definitions, so past returns can change slightly between releases. The real-data study uses the release of 26 Oct 2026 throughout.
- The real-data study assumes industry portfolios can be traded at 5 bps. In practice that means trading sector funds or all constituent stocks, which costs more for small industries.
- Shared-secret (HMAC) signing only proves integrity to whoever holds the secret; use Ed25519 signing when someone else must verify.
- The language model's own memory of market history is a leak no data-level check can see (see Li et al. above). Synthetic or anonymized data avoids it.
- A discrete choice made with hindsight can pass the leak audit when the edge is strong (see Week 4); only the trial count catches it.
- The effective number of trials is an estimate: the cluster threshold (0.5) is a choice, and both estimators are built from noisy sample correlations. The three DSR versions bracket the answer rather than settle it.
- The GitHub Actions workflow has not run yet because the repository is not on GitHub. Its steps (`pip install -e .[dev]`, `pytest -q`) were reproduced locally on every version in its matrix: Python 3.10 (pandas 2.3), 3.11 and 3.12 (pandas 3.1), and 3.13 (pandas 3.0 and 2.2).

## License

MIT
