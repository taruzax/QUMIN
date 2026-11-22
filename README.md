# `QUMIN` — Quantitative Modeling in Non-Stationary Systems

**Financial markets have a very low signal-to-noise ratio, so standard quantitative backtests do not discover alpha, they mistake random historical patterns with real predictive power. They overfit noise via temporal leakage, lookahead bias, and unpenalized multiple testing.  This problem exist for decades, and now where AI coding agents are taking over the experiments, we must pay much closer attention to it.**

**`QUMIN`** (*Quantitative Modeling in Non-Stationary Systems*) is a local-first quantitative research and MCP (Model Context Protocol) execution harness implementing core methods from Marcos López de Prado's *Advances in Financial Machine Learning*. Built on Fractional Differentiation, Purged Walk-Forward & Cross-Validation, Ledoit-Wolf Hierarchical Risk Parity, and the Deflated Sharpe Ratio, `QUMIN` enforces strict rules on what data and resutls agent can see across every stage of time-series modeling:

* **Fold isolation regression checks:** Each training split processes and fits only on data up to its cutoff date and implements a repeatable test that catches any future code change that accidentally breaks that rule.
* **Ledger-Gated Holdout Quarantine:** A final holdout test to evaluate winning model once is locked under strict rules. Local sqlite database logs every experiment attempt which helps in tracking every failed run which prevents hiding the number of experiments and reporting only the luckiest seed. 
To unlock the holdout researcher (or agent) must pass three checks:
  1. Beat baseline on out-of-fold validation splits
  2. Achieve a positive net Sharpe
  3. Score at least 0.95 on the Deflated Sharpe Ratio (DSR)

* **Headless CLI & Agent-Ready MCP Server:** CLI and MCP designed to manage agent context and give executable commands so coding agents (Claude Code, Cursor, Codex) can execute structured research tools over stdio, and multi-seed campaigns without dumping raw price series into context or accidentally burning holdout data.

> **[Run Demo (Offline Benchmark)](#2-quick-proof-the-null-control-benchmark):** Execute `uv run qumin --json demo` to watch `QUMIN` plan and run a 20-seed tournament on synthetic geometric random-walk data, apply the multiple-testing penalty across $K=20$ trials, and reject holdout promotion (`REJECTED_BY_DSR`, reason: `NONPOSITIVE_NET_SHARPE`, score: `0.0305` vs. threshold `0.95`, with **0** holdout seals written).
>
> Every run automatically generates an interactive diagnostic HTML report at `artifacts/phase1-demo/reports/decision-*.html` showing equity curves, drawdowns, candidate loss distributions, and gate evidence. Each report is paired with a SHA-256 JSON locator metadata.

---

## Why Phase 1 Exists: A Statistical Negative Control for Agentic Quant Research

If you give an autonomous coding agent unconstrained Python and daily price data, it will quickly produce a fake high-Sharpe backtest. Common mistakes include normalizing across the entire timeline, leaking overlapping future labels, assuming trades fill at the closing price, and cherry-picking 20 random seeds.

Phase 1 of `QUMIN` serves as a statistical negative control. It runs the pipeline to prove that the safeguards reject false discoveries before you trust the system with real data.

| Common Failure | How `QUMIN` Prevents It |
|---|---|
| **Global preprocessing lookahead** | Fits Fractional Differentiation (minimum $d \in [0.1, 1.0]$ passing ADF $p < 0.05$), quantile clipping, and z-score scaling strictly inside each training fold (`FoldPreprocessor`), with a separate inner fit for early stopping. |
| **Overlapping label leakage** | Purges any training row whose forward label end time $t_1$ reaches or crosses the validation start ($t_1 \ge t_{\text{val\_start}}$). |
| **Unrealistic same-bar fills** | Computes signals at bar $t$ close and executes trades at bar $t+1$ open, net of `fees_bps` and `slippage_bps`. |
| **Unpenalized multiple testing** | Logs every run in `trials.sqlite` and gates holdout access via PSR ($K \le 1$) or DSR ($K > 1$, threshold $\ge 0.95$). |
| **Agent holdout contamination** | Omits holdout execution from the MCP server (`src/lab/mcp_server.py`), exposing only discovery, run, campaign, and read-only `trial_evidence` tools alongside [`AGENTS.md`](AGENTS.md) rules. |
---

## 1. Setup (Local-First)

`QUMIN` runs entirely on local Parquet snapshots (`data/snapshots/`) and SQLite (`artifacts/trials.sqlite`)—Docker and external databases are optional. The locked dependencies include `polars-talib`; wheel availability depends on the Python version and platform.

* **Prerequisites:** Python $\ge 3.10$ and [`uv`](https://docs.astral.sh/uv/)

```bash
cd qumin
uv sync --locked --extra dev
uv run pytest   # Enforces >= 80% test coverage across src/
```

*(Background services like Dagster UI on `:3000` and MLflow on `:5001`—can be launched anytime via `make up` / `make dagster` but not necessary for local research; see [`docs/cli-and-operations.md`](docs/cli-and-operations.md)).*

---

## 2. Quick Proof: The Null-Control Benchmark

Run the built-in null-control benchmark and read its JSON output. This command tests whether safeguards catch false discoveries. Below is an example of how slightly lower MSE score beats the baseline, but negative sharpe and low DSR rejecting the trial as non-real predictive skill.

```bash
uv run qumin --json demo
```

```json
{
  "K": 20,
  "completed_variations": 20,
  "candidate_selection": {
    "verdict": "REJECTED_BY_DSR",
    "reason": "NONPOSITIVE_NET_SHARPE",
    "score": 0.030467730217260874
  },
  "decision": {
    "verdict": "REJECTED_BY_DSR",
    "evidence": {
      "metric": "mse",
      "selected_loss": 0.00014281332334817247,
      "baseline_loss": 0.00014284509836057682,
      "raw_sharpe": -0.03609978142348286,
      "score": 0.030467730217260874,
      "statistic": "dsr",
      "threshold": 0.95,
      "trial_count": 20
    }
  }
}
```

```bash
sqlite3 artifacts/phase1-demo/trials.sqlite "SELECT count(*) FROM holdout_seals;"
# 0
open artifacts/phase1-demo/reports/*.html
```

---

## 3. Architecture & Workflow Overview

### Minimal Real-Data & MCP Workflow

1. **Ingest & Pin Snapshot:** `uv run qumin ingest equities-daily` $\rightarrow$ inspect with `uv run qumin --json snapshot-list` and pin `snapshot_id` in `config/pipeline.yaml`.
2. **Run with CLI or MCP:**
   * **CLI:** `uv run qumin --json experiment --config config/pipeline.yaml`
   * **MCP (`stdio`):** Connect Claude Code / Cursor via `.mcp.example.json` (`uv run python -m lab.mcp_server`) to call `list_snapshots`, `run_experiment`, `run_campaign`, and read-only `trial_evidence`.
3. **Inspect Evidence & Report:**
   ```bash
   uv run qumin --json evidence --run-id <run_id>
   uv run qumin --json report <run_id> --output artifacts/reports/run_report.html
   ```
4. **Human-Only Terminal Holdout Check:**
   ```bash
   uv run qumin --json holdout <run_id> --acknowledge-holdout
   ```

---

## 4. Known Limitations & Roadmap

* **Survivorship Bias in Public Data:** Phase 1 includes `yfinance` ingestion so anyone can test how the harness catches high-Sharpe hallucinations on public data without paid vendor keys. Yet cannot prove a real trading strategy. They omit delisted stocks and can change on re-fetch when splits or dividends are restated. Setting data.source to "snapshot" freezes your local copy for repeatable experiments, but can't see which stocks belonged to an index on each past date.

* **Simplified Execution Costs:** Daily OHLCV backtests fill at bar $t+1$ open using flat `fees_bps` and `slippage_bps`. Market impact, order-book queue priority, and adverse selection are not modeled.

* **Soft vs. Hard Agent Isolation:** SHA-256 content hashes (`manifest.json`, `snapshot_hash`), SQLite immutability triggers on `trials` and `attempts`, and MCP tool restrictions prevent accidental leakage. However, if a coding agent also has unrestricted shell or file access, [`AGENTS.md`](AGENTS.md) acts as policy only, so untrusted agents still require OS or container isolation around `data/` and `artifacts/`.

## Roadmap: What's Next
`qumin` stays lean by focusing on local verification and governance. Future work adds new data, execution, and scaling interfaces.

### Order-Book Data Support
Add support for Order-Book and Event-Bar data: input schemas for L2 order-book depth and volume or trade-count bars, track quote freshness, bid-ask spread, and order-flow imbalance
### Smarter Trial Counts
- Upgrade TrialLedger to calculate pairwise trial return correlations to compare every run and separate distinct strategies
### External Order-Book Simulator
- A future project to build a realistic simulator of a stock exchange on a GPU. Connect `qumin`'s interface to a standalone JAX limit order book simulator driven by a sequence model trained on order-book messages. This will let strategies be tested against realistic queue priority, market impact, and simulated order-flow responses.
---

## 5. Documentation & Workbenches

* **Architecture and Methodology:** [`docs/architecture-and-methodology.md`](docs/architecture-and-methodology.md)
* **MCP Server Setup:** [`docs/mcp-and-agents.md`](docs/mcp-and-agents.md)
* **First Experiment Guide:** [`docs/cli-and-operations.md`](docs/cli-and-operations.md)
* **Agent Operating Rules:** [`AGENTS.md`](AGENTS.md)

---

## 6. Repository Map

```text
qumin/
├── config/                  # Declarative YAMLs (tickers, ingestion, pipeline, campaign, platform)
├── docs/                    # Architecture, MCP/agent workflow, and CLI/operations references
├── src/lab/
│   ├── core/                # Pydantic schemas, config loaders, deterministic hashing
│   ├── quant/               # Pure AFML core (FFD, features, triple-barrier, purged CV, HRP, DSR)
│   ├── models/              # Baseline, XGBoost, PyTorch GRU/LSTM, and device resolution
│   ├── research/            # Experiment runner, campaigns, fold preprocessors, TrialLedger, gate
│   ├── platform/            # Parquet snapshot/dataset stores, Dagster assets, MLflow adapter
│   ├── mcp_server.py        # Stdio MCP server (list_snapshots, run_experiment, run_campaign, trial_evidence)
│   └── cli.py               # Headless CLI entrypoint (`uv run qumin ...`)
└── tests/                   # Regression & invariance suite (including test_future_perturbation.py)
```
