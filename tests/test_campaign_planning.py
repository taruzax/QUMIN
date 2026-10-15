from types import SimpleNamespace

import polars as pl
import pytest

import lab.research.campaign as campaign
from lab.core.config import CampaignMatrixConfig, PipelineConfig, Timeframe
from lab.platform.data_access import build_market_snapshot
from lab.platform.market_store import SnapshotCatalog
from lab.quant.timing import bar_times_for_session


def test_campaign_plan_is_stable_for_saved_snapshot_and_full_config(tmp_path, monkeypatch):
    opened, closed, session_id = bar_times_for_session("XNYS", "2024-01-02", Timeframe.D1)[0]
    bars = pl.DataFrame({
        "ticker": ["AAA"], "raw_bar_index": [0], "bar_open_time": [opened],
        "bar_close_time": [closed], "timestamp": [closed], "open": [100.0],
        "high": [102.0], "low": [99.0], "close": [101.0], "volume": [500.0],
        "session_id": [session_id], "return_1b": [0.01],
    })
    snapshot = build_market_snapshot([bars], calendar="XNYS", timeframe=Timeframe.D1, provenance={"source": "test"})
    root = tmp_path / "snapshots"
    SnapshotCatalog(root).publish(snapshot, stream_id="daily")
    monkeypatch.setattr(campaign, "get_platform_config", lambda: SimpleNamespace(paths=SimpleNamespace(snapshot_dir=root)))
    matrix = CampaignMatrixConfig(
        name="stable", snapshot_ids=(snapshot.snapshot_id,), feature_column_sets=(("return_1b",),),
        seeds=(7,), max_planned_experiments=1, worker_count=1,
    )

    base_config = PipelineConfig(data={"timeframe": "1d"})
    first = campaign.plan_campaign(base_config, matrix)
    second = campaign.plan_campaign(base_config, matrix)

    assert first.campaign_id == second.campaign_id
    assert first.variations[0].effective_config_hash == PipelineConfig(
        data={"source": "snapshot", "snapshot_id": snapshot.snapshot_id, "timeframe": "1d"},
        features={"feature_columns": ["return_1b"]}, models={"seed": 7}, campaign={"name": "stable"},
    ).resolved_config_hash
    changed_config = base_config.model_copy(
        update={"backtest": base_config.backtest.model_copy(update={"initial_capital": 200_000.0})}
    )
    changed = campaign.plan_campaign(changed_config, matrix)
    assert changed.variations[0].effective_config_hash != first.variations[0].effective_config_hash
    assert changed.campaign_id != first.campaign_id


def test_campaign_planning_rejects_unknown_feature_without_calculating_indicators(tmp_path, monkeypatch):
    opened, closed, session_id = bar_times_for_session("XNYS", "2024-01-02", Timeframe.D1)[0]
    bars = pl.DataFrame({
        "ticker": ["AAA"], "raw_bar_index": [0], "bar_open_time": [opened],
        "bar_close_time": [closed], "timestamp": [closed], "open": [100.0],
        "high": [102.0], "low": [99.0], "close": [101.0], "volume": [500.0],
        "session_id": [session_id],
    })
    snapshot = build_market_snapshot([bars], calendar="XNYS", timeframe=Timeframe.D1, provenance={"source": "test"})
    root = tmp_path / "snapshots"
    SnapshotCatalog(root).publish(snapshot, stream_id="daily")
    monkeypatch.setattr(campaign, "get_platform_config", lambda: SimpleNamespace(paths=SimpleNamespace(snapshot_dir=root)))
    matrix = CampaignMatrixConfig(
        name="invalid-feature", snapshot_ids=(snapshot.snapshot_id,), feature_column_sets=(("not_a_feature",),),
        max_planned_experiments=1,
    )

    with pytest.raises(ValueError, match="Unknown feature columns.*not_a_feature"):
        campaign.plan_campaign(PipelineConfig(data={"timeframe": "1d"}), matrix)
