import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lab.mcp_server as mcp_server_module
from lab.core.config import PipelineConfig
from lab.core.contracts import CampaignPlan, ExperimentVariation
from lab.platform.campaign_store import CampaignStore


@pytest.fixture
def mcp_server():
    return mcp_server_module


def _campaign_plan(campaign_id: str) -> CampaignPlan:
    variations = tuple(
        ExperimentVariation(
            variation_id=f"variation-{index}",
            campaign="adapter-test",
            snapshot_id="snapshot-1",
            snapshot_hash="snapshot-hash",
            effective_config_hash=f"config-{index}",
            feature_specification=("close",),
            seed=index,
        )
        for index in (1, 2)
    )
    return CampaignPlan(
        campaign_id=campaign_id,
        campaign="adapter-test",
        base_config_hash="base-hash",
        variations=variations,
        max_planned_experiments=2,
        worker_count=1,
    )


def test_mcp_server_advertises_exactly_expected_tools(mcp_server):
    tools = asyncio.run(mcp_server.mcp.list_tools())
    tool_names = [tool.name for tool in tools]
    assert len(tool_names) == 4
    assert set(tool_names) == {
        "list_snapshots",
        "run_experiment",
        "run_campaign",
        "trial_evidence",
    }
    assert "request_holdout" not in tool_names
    assert not hasattr(mcp_server, "request_holdout")


def test_campaign_adapter_returns_persisted_results_on_fresh_and_resumed_calls(tmp_path, monkeypatch, mcp_server):
    from lab.research import campaign as campaign_module

    plan = _campaign_plan("campaign-adapter-test")
    base_config = PipelineConfig()
    base_config.models.device = "cpu"
    platform = SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path))
    monkeypatch.setattr(mcp_server, "get_platform_config", lambda: platform)
    monkeypatch.setattr(campaign_module, "get_platform_config", lambda: platform)
    monkeypatch.setattr(
        mcp_server,
        "CampaignMatrixConfig",
        SimpleNamespace(from_yaml=lambda _path: SimpleNamespace(base_pipeline_config=Path("base.yaml"))),
    )
    monkeypatch.setattr(mcp_server, "PipelineConfig", SimpleNamespace(from_yaml=lambda _path: base_config))
    monkeypatch.setattr(mcp_server, "plan_campaign", lambda _config, _matrix: plan)
    monkeypatch.setattr(mcp_server, "execute_campaign", campaign_module.execute_campaign)
    attempts = {"variation-2": 0}

    def execute_one(_config, variation, *_claim):
        if variation.variation_id == "variation-2":
            attempts[variation.variation_id] += 1
            if attempts[variation.variation_id] == 1:
                return {"variation_id": variation.variation_id, "status": "failed", "error": "expected fixture failure"}
        return {"variation_id": variation.variation_id, "status": "completed", "run_id": f"run-{variation.variation_id}"}

    monkeypatch.setattr(campaign_module, "_execute_one", execute_one)
    monkeypatch.setattr(campaign_module, "index_deferred_run", lambda run_id: {"status": "disabled", "run_id": run_id})

    class ReadOnlyCampaignStore(CampaignStore):
        def finalize(self, *_args, **_kwargs):
            raise AssertionError("MCP adapter must not finalize a campaign")

    monkeypatch.setattr(mcp_server, "CampaignStore", ReadOnlyCampaignStore)

    # 1. Partial failure: variation-1 completes, variation-2 fails
    first = mcp_server.run_campaign("campaign.yaml")
    assert first["campaign_id"] == plan.campaign_id
    assert [item["status"] for item in first["results"]] == ["completed", "failed"]
    first_payload = json.loads(Path(first["campaign_result"]["bundle_path"]).read_text())
    assert [entry["status"] for entry in first_payload["entries"]] == ["completed", "failed"]
    assert first["completed_runs"] == [{"variation_id": "variation-1", "run_id": "run-variation-1"}]

    # 2. Resumed success: variation-2 is retried and completes
    second = mcp_server.run_campaign("campaign.yaml")
    assert second["campaign_id"] == plan.campaign_id
    assert [item["status"] for item in second["results"]] == ["completed"]
    assert all(
        entry["status"] == "completed"
        for entry in json.loads(Path(second["campaign_result"]["bundle_path"]).read_text())["entries"]
    )
    assert second["completed_runs"] == [
        {"variation_id": "variation-1", "run_id": "run-variation-1"},
        {"variation_id": "variation-2", "run_id": "run-variation-2"},
    ]

    # 3. Fully completed rerun: all variations completed, empty results, completed_runs still full
    resumed = mcp_server.run_campaign("campaign.yaml")
    assert resumed["campaign_id"] == plan.campaign_id
    assert resumed["results"] == []
    assert resumed["campaign_result"] == second["campaign_result"]
    assert resumed["completed_runs"] == [
        {"variation_id": "variation-1", "run_id": "run-variation-1"},
        {"variation_id": "variation-2", "run_id": "run-variation-2"},
    ]


def test_mcp_tool_wrappers_keep_read_only_preview(tmp_path, monkeypatch, mcp_server):
    platform = SimpleNamespace(paths=SimpleNamespace(snapshot_dir=tmp_path, artifacts_dir=tmp_path))
    monkeypatch.setattr(mcp_server, "get_platform_config", lambda: platform)
    closed = []

    class Catalog:
        def __init__(self, root, *, read_only):
            assert root == tmp_path
            assert read_only is True
            self.connection = SimpleNamespace(close=lambda: closed.append(True))

        def list(self):
            return [SimpleNamespace(model_dump=lambda **_kwargs: {"snapshot_id": "snapshot-1"})]

    monkeypatch.setattr(mcp_server, "SnapshotCatalog", Catalog)
    assert mcp_server.list_snapshots() == [{"snapshot_id": "snapshot-1"}]
    assert closed == [True]

    base_config = PipelineConfig()
    monkeypatch.setattr(mcp_server, "PipelineConfig", SimpleNamespace(from_yaml=lambda _path: base_config))
    monkeypatch.setattr(
        mcp_server,
        "run_experiment",
        lambda config: SimpleNamespace(ref=SimpleNamespace(run_id="run-1", snapshot_hash="snapshot-hash")),
    )
    assert mcp_server.run_development("config.yaml") == {"run_id": "run-1", "snapshot_hash": "snapshot-hash"}

    previews = []
    monkeypatch.setattr(
        mcp_server, "preview_holdout", lambda run_id, *, model_name: previews.append((run_id, model_name)) or {"advisory": True}
    )
    assert mcp_server.trial_evidence("run-1", "xgboost") == {"advisory": True}
    assert previews == [("run-1", "xgboost")]
