"""Stdio tools for the managed research workflow."""

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from lab.core.config import CampaignMatrixConfig, PipelineConfig, get_platform_config
from lab.platform.campaign_store import CampaignStore
from lab.platform.market_store import SnapshotCatalog
from lab.research.campaign import execute_campaign, plan_campaign
from lab.research.experiment import preview_holdout, run_experiment

mcp = MCPServer("qumin")


@mcp.tool(name="list_snapshots")
def list_snapshots() -> list[dict[str, Any]]:
    """List the configured local snapshots."""
    catalog = SnapshotCatalog(get_platform_config().paths.snapshot_dir, read_only=True)
    try:
        return [entry.model_dump(mode="json") for entry in catalog.list()]
    finally:
        catalog.connection.close()


@mcp.tool(name="run_experiment")
def run_development(config_path: str) -> dict[str, str]:
    """Run a development experiment in the configured artifact store."""
    result = run_experiment(PipelineConfig.from_yaml(Path(config_path)))
    return {"run_id": result.ref.run_id, "snapshot_hash": result.ref.snapshot_hash}


@mcp.tool(name="run_campaign")
def run_campaign(campaign_path: str, config_path: str | None = None) -> dict[str, Any]:
    """Run a campaign using the configured catalog and artifact store."""
    matrix_path = Path(campaign_path)
    matrix = CampaignMatrixConfig.from_yaml(matrix_path)
    selected = Path(config_path) if config_path is not None else matrix.base_pipeline_config
    if not selected.is_absolute() and not selected.exists():
        selected = matrix_path.parent / selected
    config = PipelineConfig.from_yaml(selected)
    plan = plan_campaign(config, matrix)
    results = execute_campaign(config, plan)
    store = CampaignStore(get_platform_config().paths.artifacts_dir)
    try:
        outcome = store.latest_result(plan.campaign_id)
        completed_runs = [
            {"variation_id": entry.variation_id, "run_id": entry.run_id}
            for entry in store.entries(plan.campaign_id)
            if entry.status == "completed"
        ]
    finally:
        store.close()
    return {
        "campaign_id": plan.campaign_id,
        "results": results,
        "campaign_result": outcome.model_dump(mode="json") if outcome else None,
        "completed_runs": completed_runs,
    }


@mcp.tool(name="trial_evidence")
def trial_evidence(run_id: str, model: str | None = None) -> dict[str, Any]:
    """Preview the current holdout gate without writing a decision."""
    return preview_holdout(run_id, model_name=model)


if __name__ == "__main__":
    mcp.run(transport="stdio")
