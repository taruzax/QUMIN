# CLI Reference, Experiment Walkthrough & Operations

All commands below assume your working directory is `qumin/`.

---

## 1. First Real Experiment (End-to-End Walkthrough)

> **Epistemic Note on `yfinance`:** Free `yfinance` daily bars are ideal for testing pipeline mechanics, walk-forward leakage guards, and multi-trial DSR rejection, but static ticker lists carry survivorship bias and provider corporate-action restatements. Always pin `data.source: "snapshot"` for reproducible comparisons; see [`architecture-and-methodology.md`](./architecture-and-methodology.md#4-phase-1-known-epistemic-boundaries-statistical-negative-control).

### Step 1: Define Instrument Universe (`config/tickers.yaml`)
```yaml
- ticker: AAPL
  name: Apple Inc.
  sector: Technology
  exchange: NASDAQ
- ticker: MSFT
  name: Microsoft Corporation
  sector: Technology
  exchange: NASDAQ
- ticker: NVDA
  name: NVIDIA Corporation
  sector: Technology
  exchange: NASDAQ
```

### Step 2: Configure & Run Stream Ingestion (`config/ingestion.yaml`)
```yaml
streams:
  - name: tech-daily
    provider: yfinance
    tickers: [AAPL, MSFT, NVDA]
    calendar: XNYS
    timeframe: 1d
    start_boundary: "2020-01-01"
    refresh_overlap_bars: 2
    gap_policy: reject
    enabled: true
```

Ingest raw batches (`data/market_batches/`) and publish a content-addressed snapshot (`data/snapshots/`):

```bash
uv run qumin ingest tech-daily
```

### Step 3: Inspect Published Snapshots
```bash
uv run qumin --json snapshot-list
uv run qumin --json snapshot-inspect <snapshot_id>
```

### Step 4: Pin Snapshot in `config/pipeline.yaml`
```yaml
data:
  source: "snapshot"
  snapshot_id: "<snapshot_id>"
  timeframe: "1d"
  calendar: "XNYS"

features:
  return_lags: [1, 5, 10, 21]
  target_horizons: [1, 5]

task:
  kind: "regression"

splits:
  n_folds: 3
  holdout_fraction: 0.20
  development_train_fraction: 0.50
  embargo_bars: 0

models:
  baseline:
    enabled: true
  xgboost:
    enabled: true
    params:
      n_estimators: 100
      max_depth: 3
      learning_rate: 0.05
  gru:
    enabled: false
  lstm:
    enabled: false

allocation:
  method: "hrp"
  max_position_size: 0.40
  gross_limit: 1.0

backtest:
  initial_capital: 100000.0
  fees_bps: 5.0
  slippage_bps: 2.0
```

### Step 5: Execute Development Run & Preview Gate Evidence
```bash
# Run development experiment (holdout remains untouched)
uv run qumin --json experiment --config config/pipeline.yaml

# Preview out-of-fold baseline comparison and PSR/DSR score (read-only)
uv run qumin --json evidence --run-id <run_id>
```

### Step 6: Compile Diagnostic HTML Report
```bash
uv run qumin --json report <run_id> --output artifacts/reports/tech_run_report.html
open artifacts/reports/tech_run_report.html
```

### Step 7: Terminal Holdout Evaluation (Human Researcher Only)
Only after inspecting `evidence` and confirming a genuine candidate ready for one-shot validation:

```bash
uv run qumin --json holdout <run_id> --acknowledge-holdout
```

---

## 2. Complete CLI Reference (`src/lab/cli.py`)

> **Global Flag Placement Rule:** `--json` belongs on the top-level `qumin` parser and **must precede the subcommand**:
> - Correct: `uv run qumin --json experiment --config config/pipeline.yaml`
> - Incorrect: `uv run qumin experiment --config config/pipeline.yaml --json`

| Subcommand | Syntax & Options | Purpose |
|---|---|---|
| `demo` | `qumin [--json] demo [--store-root artifacts/phase1-demo]` | Runs the offline 20-seed synthetic null-control tournament in an isolated store. |
| `ingest` | `qumin [--json] ingest <stream> [--config config/ingestion.yaml]` | Fetches provider bars, validates calendar/gaps, and publishes a snapshot. |
| `snapshot-list` | `qumin [--json] snapshot-list [--root data/snapshots]` | Lists all snapshots registered in `SnapshotCatalog`. |
| `snapshot-inspect` | `qumin [--json] snapshot-inspect <snapshot_id> [--root data/snapshots]` | Prints row counts, columns, tickers, and date boundaries for a snapshot. |
| `prepare` | `qumin [--json] prepare [--config config/pipeline.yaml] [--store-root artifacts]` | Builds and persists an unfitted `PreparedDataset` (`prepared-dataset.v1`). |
| `dataset-inspect` | `qumin [--json] dataset-inspect <dataset_id> [--store-root artifacts]` | Inspects metadata and schema for a stored `PreparedDataset`. |
| `experiment` | `qumin [--json] experiment [--config config/pipeline.yaml] [--dataset-id <id>]` | Executes a development tournament across purged folds and writes run artifacts. |
| `development` | `qumin [--json] development [--config config/pipeline.yaml] [--store-dataset]` | Runs development workflow, optionally persisting the `PreparedDataset` first. |
| `campaign-plan` | `qumin [--json] campaign-plan --campaign <campaign.yaml> [--config <pipeline.yaml>]` | Expands a Cartesian parameter/seed matrix and prints the planned variations. |
| `campaign-run` | `qumin [--json] campaign-run --campaign <campaign.yaml> [--config <pipeline.yaml>]` | Plans and executes a multi-variation campaign across configured workers. |
| `campaign-status` | `qumin [--json] campaign-status <campaign_id>` | Queries completed/failed variation status and campaign summary from `CampaignStore`. |
| `campaign-resume` | `qumin [--json] campaign-resume --campaign <campaign.yaml> [--config <pipeline.yaml>]` | Resumes incomplete variations for an existing campaign plan. |
| `backtest` | `qumin [--json] backtest <run_id> [--model <name>] [--allocation <yaml>] [--backtest-config <yaml>]` | Re-runs portfolio allocation and next-open backtest on saved OOF predictions. |
| `evidence` | `qumin [--json] evidence --run-id <run_id> [--model <name>]` | Read-only holdout gate preview (baseline loss check, net Sharpe, PSR/DSR). |
| `holdout` | `qumin [--json] holdout <run_id> --acknowledge-holdout [--model <name>] [--config <pipeline.yaml>]` | Evaluates the statistical gate; if `PASSED`, seals the snapshot and scores holdout. |
| `report` | `qumin [--json] report [<run_id>] --output <path.html> [--config <pipeline.yaml>] [--decision-id <id>]` | Generates a self-contained HTML diagnostic report and JSON locator metadata. |
| `report-inspect` | `qumin [--json] report-inspect <report_id> [--store-root artifacts]` | Loads the JSON locator record linking a report checksum to its run and decision. |

The replacement for `run-pipeline` is `uv run qumin prepare --config config/pipeline.yaml`, which writes a dataset bundle and returns its ID. For in-memory preparation, call `lab.research.prepare_dataset(config)` from Python.

### Makefile Shortcuts (`Makefile`)

- `make demo` — `uv run qumin demo`
- `make test` — `uv run pytest` (enforces `--cov=src --cov-fail-under=80`)
- `make notebook` — `uv run jupyter lab notebooks/research_walkthrough.ipynb`
- `make dagster` — `uv run dagster dev -w config/dagster-workspace.yaml` (local Dagster UI on `:3000`)
- `make up` / `make down` — starts/stops optional Docker Compose services (Dagster webserver `:3000`, daemon, and MLflow `:5001`)

---

## 3. Hardware Selection & Interactive Interfaces

### Hardware Acceleration (`MODELS__DEVICE`)
Configure `models.device` in `config/pipeline.yaml` or override via `MODELS__DEVICE`:

```bash
export MODELS__DEVICE=cpu   # Deterministic CPU baseline
export MODELS__DEVICE=mps   # Apple Silicon Metal (PyTorch models only)
export MODELS__DEVICE=cuda  # NVIDIA GPU
```

| Model | `cpu` | `mps` | `cuda` | Notes |
|---|:---:|:---:|:---:|---|
| `baseline` | Yes | No | No | Non-parametric; requesting `mps` or `cuda` raises `ValueError`. |
| `xgboost` | Yes | No | Yes | `mps` unsupported by XGBoost adapter; raises `ValueError`. |
| `gru` / `lstm` | Yes | Yes | Yes | PyTorch backend. Cross-device floating-point reductions are not bit-identical. |

### Interactive Notebooks
1. **Algorithm Review Workbench (`notebooks/quant_algorithms_walkthrough.ipynb`):** Read-only visual walkthrough of `lab.quant` (calendars, indicators, FFD $d$-curves, triple-barrier labeling, purged CV, Ledoit-Wolf HRP) on in-memory data without mutating disk stores.
2. **Manual Research Workbench (`notebooks/research_walkthrough.ipynb`):** Interactive runner for configuring snapshots, running fold tournaments, viewing inline equity curves, and inspecting gate evidence (`make notebook`).

---

## 4. Artifact Layout & SQLite Trial Ledger

### Run Directory (`artifacts/runs/<run_id>/`)
```text
artifacts/runs/<run_id>/
├── manifest.json                  # Config provenance, git SHA/dirty state, file SHA-256 checksums
├── split.json                     # Chronological fold and holdout boundaries
├── metrics.json                   # Aggregated out-of-fold candidate metrics
├── predictions.json               # Selected winner out-of-fold predictions
├── snapshot/
│   ├── bars.parquet               # Frozen market snapshot bars
│   └── metadata.json              # Snapshot provenance & calendar metadata
├── preprocessors/
│   └── fold-<k>.json              # Fold-fitted state (per-ticker FFD min d, clip bounds, scalers)
├── candidates/                    # Per-candidate prediction frames (when multiple models run)
├── models/                        # Serialized fold model bundles (manifest.json + model binary)
└── backtest/                      # Next-open execution traces (orders, trade_events, equity, exposure)
```

### Trial Ledger (`artifacts/trials.sqlite`)
Managed by `TrialLedger` (`src/lab/research/trial_ledger.py`):
- `snapshot_reservations`: Binds `snapshot_hash` to `holdout_start`, `holdout_end`, and boundary policy version prior to training.
- `trials`: Canonical trial configurations (`logical_trial_id`, `config_hash`, `snapshot_hash`, `evaluation_hash`); protected by `trials_are_immutable` SQLite trigger.
- `attempts`: Individual execution attempts; protected by `attempt_identity_is_immutable` and `attempt_terminal_is_immutable` triggers once an attempt leaves `running`.
- `holdout_decisions`: Gate evaluation records (`verdict`, `evidence_json`, `source_run_id`).
- `holdout_seals`: Active snapshot seals tracking holdout execution state (`reserved` $\rightarrow$ `running` $\rightarrow$ `completed`/`failed`/`interrupted`). Any row for `snapshot_hash` blocks further development registrations on that snapshot.
- `schema_migrations`: Internal ledger schema version tracking.

---

## 5. Troubleshooting & Quality Checks

- **`ValueError: embargo_bars>0 is unsupported`:** Walk-forward training strictly precedes validation ($t_{\text{train\_end}} = t_{\text{val\_start}}$), leaving zero post-validation training rows to embargo while purging still removes overlapping forward horizons. Set `splits.embargo_bars: 0` in `config/pipeline.yaml` (or use `PurgedKFold` in `lab.quant.cv` directly for non-walk-forward CV).
- **Snapshot is sealed (`Snapshot is sealed against additional development registrations` or `HOLDOUT_ALREADY_SEALED`):** A passed holdout evaluation already sealed that `snapshot_hash` in `artifacts/trials.sqlite`. Ingest a new time window/snapshot or use an isolated `--store-root` for sandbox testing—never delete `trials.sqlite` to re-mine a holdout you have already observed.
- **Demo store vs. research store:** `qumin demo` writes exclusively to `artifacts/phase1-demo/` and never touches `artifacts/trials.sqlite`.
- **MCP call appears quiet during `run_experiment` / `run_campaign`:** Multi-fold training (especially `gru`/`lstm` or 20-variation campaigns) runs synchronously before returning the handle payload; check CPU activity rather than re-submitting duplicate calls.
- **Missing model folder under `models/`:** Inspect `candidates` in `artifacts/runs/<run_id>/manifest.json`—disabled (`enabled: false`) or failed candidates record their status in the manifest without writing a model bundle.

### Developer Verification Suite
```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest
```
