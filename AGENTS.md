# Qumin — Agent Operating Instructions

These rules apply to all autonomous coding and quantitative research agents operating inside `qumin/`.

---

## 1. Non-Negotiable Epistemic & Containment Rules

1. **Use only the managed workflow:** Execute experiments via the `qumin` CLI (`uv run qumin ...`), the stdio MCP server (`lab.mcp_server`), or the public `lab.research` APIs against the configured research store (`config/platform.yaml`).
2. **Keep the holdout fixed and untouched:** Never alter a snapshot's reserved holdout boundary, and never read trailing holdout rows directly from `data/snapshots/` to tune features, models, or hyperparameters.
3. **Never trigger holdout evaluation autonomously:** Do **not** run `uv run qumin holdout <run_id> --acknowledge-holdout` on your own initiative. Inspect readiness strictly via read-only `uv run qumin --json evidence --run-id <run_id>` (or MCP `trial_evidence`), present the exact output to the human researcher, and run `holdout` only if the user explicitly instructs you to execute it for that specific `run_id`.
4. **Report verdicts verbatim:** Always report gate `verdict`, `reason`, `score`, `raw_sharpe`, and `trial_count` ($K$) exactly as returned by Qumin—never round up failed DSR scores or summarize a rejected run as passing.
5. **Treat artifacts and ledgers as immutable:** Never manually edit, truncate, or delete `artifacts/trials.sqlite`, `artifacts/runs/<run_id>/`, or `data/snapshots/`. Never point production research commands at the isolated demo store (`artifacts/phase1-demo/`).

---

## 2. Canonical Commands (Run from `qumin/`)

> **Critical CLI Flag Rule:** Global `--json` **must** come *before* the subcommand (`uv run qumin --json <subcommand> ...`), never after.

```bash
# Offline 20-seed negative-control benchmark (isolated in artifacts/phase1-demo/)
uv run qumin --json demo

# Discover and inspect published market snapshots
uv run qumin --json snapshot-list
uv run qumin --json snapshot-inspect <snapshot_id>

# Run a single development tournament or multi-variation campaign
uv run qumin --json experiment --config config/pipeline.yaml
uv run qumin --json campaign-plan --campaign config/campaign.example.yaml
uv run qumin --json campaign-run --campaign config/campaign.example.yaml

# Read-only statistical gate preview (safe: writes zero seals)
uv run qumin --json evidence --run-id <run_id>

# Generate standalone HTML diagnostic report
uv run qumin --json report <run_id> --output artifacts/reports/<name>.html

# Code quality & regression verification (>= 80% coverage + perturbation invariance)
uv run ruff check src tests
uv run mypy src
uv run pytest
```

### Configuration Guardrails
* **Pin snapshots for comparisons:** Set `data.source: "snapshot"` and `data.snapshot_id` in `config/pipeline.yaml` so multi-run comparisons share an identical `snapshot_hash`.
* **Walk-forward embargo:** Keep `splits.embargo_bars: 0` in `config/pipeline.yaml` (walk-forward splits have zero post-validation training rows; overlapping forward labels are already purged automatically).
* **Hardware device (`MODELS__DEVICE`):** `baseline` is CPU-only and `xgboost` supports `cpu`/`cuda` (`mps` raises `ValueError`). Default to `MODELS__DEVICE=cpu` unless running `gru`/`lstm` or CUDA-backed `xgboost`.

---

## 3. Documentation Map (Read Before Acting)

* **Project overview & Phase 1 Negative-Control scope:** [`README.md`](README.md)
* **CLI subcommands, YAML walkthrough, artifact layout, `trials.sqlite` schema & troubleshooting:** [`docs/cli-and-operations.md`](docs/cli-and-operations.md)
* **MCP server setup, 4-tool contracts (`list_snapshots`, `run_experiment`, `run_campaign`, `trial_evidence`) & payload hygiene:** [`docs/mcp-and-agents.md`](docs/mcp-and-agents.md)
* **Econometric stages (FFD, triple-barrier, purged walk-forward, HRP), Holdout Gate state machine & `yfinance`/cost boundaries:** [`docs/architecture-and-methodology.md`](docs/architecture-and-methodology.md)
