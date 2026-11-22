# MCP Server & Autonomous Agent Workflow

Qumin ships a local `stdio` Model Context Protocol (MCP) server (`src/lab/mcp_server.py`) built on the official `mcp` Python SDK (`MCPServer`). It allows AI coding and research agents (Claude Code, Claude Desktop, Cursor, Codex) to discover immutable snapshots, run single experiments or multi-seed campaigns, and inspect out-of-fold gate evidence without blowing up context windows.

---

## 1. Guardrail & Containment Model (What MCP Prevents vs. What Requires `AGENTS.md`)

Qumin separates **exploration** from **terminal holdout evaluation** using two complementary layers:

1. **Structural MCP Tool Exclusion:** The MCP server exposes **4 tools** and deliberately **omits** any holdout evaluation tool. Through MCP alone, an agent can run experiments and preview out-of-fold PSR/DSR evidence (`trial_evidence`), but cannot unseal or evaluate the holdout partition.
2. **Cooperative Local Workspace Rules (`AGENTS.md`):** In local coding-agent hosts (Claude Code, Cursor, Codex) that also grant general Bash and filesystem tools in the same user account, MCP exclusion is a **workflow guardrail (*poka-yoke*), not OS process sandboxing**. Agents operating inside `qumin/` must follow [`AGENTS.md`](../AGENTS.md):
   - Never invoke `uv run qumin holdout` autonomously unless explicitly instructed by the human researcher after reviewing `evidence` / `trial_evidence`.
   - Never manually read holdout slices in `data/snapshots/` or mutate `artifacts/trials.sqlite` and `artifacts/runs/`.

*(If you require hard untrusted-agent isolation, run the MCP server or CLI store under a separate OS user/container without giving the agent direct shell/filesystem access to `data/` and `artifacts/`.)*

---

## 2. Host Configuration

### Workspace Clients (`.mcp.json` — Claude Code / Cursor)
Copy `.mcp.example.json` to `.mcp.json` in the `qumin/` root (or add an optional `env` override for hardware selection):

```json
{
  "mcpServers": {
    "qumin": {
      "command": "uv",
      "args": [
        "--directory",
        "${workspaceFolder}",
        "run",
        "python",
        "-m",
        "lab.mcp_server"
      ],
      "env": {
        "MODELS__DEVICE": "cpu"
      }
    }
  }
}
```

### Claude Desktop (`claude_desktop_config.json`)
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "qumin": {
      "command": "/usr/local/bin/uv",
      "args": [
        "--directory",
        "/absolute/path/to/qumin",
        "run",
        "python",
        "-m",
        "lab.mcp_server"
      ],
      "env": {
        "MODELS__DEVICE": "cpu"
      }
    }
  }
}
```
*(Replace `/usr/local/bin/uv` with `which uv` and `/absolute/path/to/qumin` with your local checkout path.)*

---

## 3. Advertised MCP Tool Contracts

`src/lab/mcp_server.py` registers exactly four tools over `stdio`:

| Tool Name | Kind | Arguments | Return Payload | Behavior |
|---|:---:|---|---|---|
| `list_snapshots` | Read | *None* | `list[dict[str, Any]]` | Lists published snapshots in `data/snapshots/` (ID, SHA-256 hash, stream, date bounds, ticker count). Opens catalog read-only. |
| `run_experiment` | Write | `config_path: str` | `{"run_id": str, "snapshot_hash": str}` | Runs one development experiment from a `PipelineConfig` YAML and registers the attempt in `trials.sqlite`. |
| `run_campaign` | Write | `campaign_path: str`, `config_path: str \| None = None` | `{"campaign_id": str, "results": list[dict], "campaign_result": dict \| None, "completed_runs": list[dict]}` | Plans and executes a Cartesian parameter/seed matrix (`CampaignMatrixConfig`) and returns completed `(variation_id, run_id)` handles. |
| `trial_evidence` | Read | `run_id: str`, `model: str \| None = None` | `dict[str, Any]` | **Read-only gate preview (`preview_holdout`):** Computes OOF baseline comparison, net Sharpe, and PSR/DSR verdict **without** writing a decision or holdout seal. |

---

## 4. Payload Budgeting & Context Hygiene

LLM tool hosts truncate or spill oversized tool responses (e.g., ~25k tokens in Claude Code), which breaks structured reasoning when tools dump raw time series. Qumin keeps MCP payloads compact by design:
- **Handle-based mutations:** `run_experiment` and `run_campaign` return lightweight identifiers (`run_id`, `campaign_id`, `snapshot_hash`), never raw Parquet bars, equity arrays, or serialized weights.
- **Disk-backed artifacts:** Detailed summaries (`manifest.json`, `metrics.json`) and time series (`backtest/*.parquet`) live under `artifacts/runs/<run_id>/`. Agents query statistical readiness first via `trial_evidence`.

---

## 5. Interactive Verification with MCP Inspector

Verify tool registration and JSON-RPC responses locally without an LLM client:

```bash
npx @modelcontextprotocol/inspector uv run python -m lab.mcp_server
```

1. Open the local Inspector UI printed in your terminal (default `http://localhost:5173`) and click **Connect**.
2. Inspect the **Tools** tab to confirm `list_snapshots`, `run_experiment`, `run_campaign`, and `trial_evidence` appear.
3. Run `list_snapshots` and `trial_evidence` against an existing `run_id` to validate round-trip serialization.

---

## 6. Reference Agent-Human Operating Loop

1. Call `list_snapshots`, then `run_experiment` or `run_campaign`. A campaign returns `completed_runs`, each containing `variation_id` and `run_id`; this includes persisted completions when a campaign is resumed.
2. After the campaign finishes, call `trial_evidence(run_id)` for the candidate run IDs in `completed_runs`. The preview is advisory and read-only. Its shape includes a top-level verdict and nested evidence, for example (excerpt from the null-control demo):

   ```json
   {
     "verdict": "REJECTED_BY_DSR",
     "evidence": {
       "reason": "NONPOSITIVE_NET_SHARPE",
       "score": 0.030467730217260874,
       "raw_sharpe": -0.03609978142348286,
       "trial_count": 20,
       "statistic": "dsr",
       "threshold": 0.95
     },
     "decision_id": null,
     "child_run_id": null,
     "advisory": true
   }
   ```

3. Compare eligible candidates using the same snapshot, split, and current evidence context; inspect the returned evidence, including comparable trials and exclusions, before selecting a run. Preview again if the ledger changes, since the evidence can change as trials are added. If no candidate passes, stop.
4. Give the researcher the selected `run_id` and exact preview evidence for review. The researcher may then explicitly run `uv run qumin holdout <run_id> --acknowledge-holdout` in the terminal. The gate is evaluated again at that point; a preview does not guarantee a seal or holdout evaluation.
