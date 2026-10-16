import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import lab.demo as demo_module
import lab.platform.dataset_store as dataset_store_module
import lab.research.experiment as experiment_module
import lab.research.preprocessing as preprocessing_module
from lab.core.config import PipelineConfig, Timeframe
from lab.quant.metrics import return_convention, statistical_returns
from lab.research.experiment import evaluate_saved_predictions, load_candidate_predictions, load_run
from lab.research.trial_ledger import TrialLedger


@pytest.mark.parametrize("timeframe", [Timeframe.D1, Timeframe.H1])
def test_offline_demo_runs_reload_and_saved_prediction_workflow(tmp_path, monkeypatch, capsys, timeframe):
    artifacts = tmp_path / "artifacts"
    snapshots = tmp_path / "snapshots"
    platform = SimpleNamespace(
        paths=SimpleNamespace(artifacts_dir=artifacts, snapshot_dir=snapshots),
        mlflow=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(demo_module, "get_platform_config", lambda: platform)
    monkeypatch.setattr(dataset_store_module, "get_platform_config", lambda: platform)
    monkeypatch.setattr(experiment_module, "get_platform_config", lambda: platform)
    monkeypatch.setattr(preprocessing_module, "get_platform_config", lambda: platform)

    # Exercise the inner chronological selector and outer refit in the daily
    # workflow. The hourly integration stays smaller; its standalone smoke uses defaults.
    overrides = (
        {"models": {"early_stopping": True, "early_stopping_fraction": 0.20, "early_stopping_patience": 2}}
        if timeframe == Timeframe.D1
        else {"models": {"xgboost": {"enabled": False}}}
    )
    config = PipelineConfig.from_yaml("config/demo.yaml", overrides=overrides).model_dump(mode="json")
    config["data"]["input_path"] = str(Path("data/demo/ohlcv.csv").resolve())
    config["data"]["metadata_path"] = str(Path("data/demo/metadata.yaml").resolve())
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    report = demo_module.run_demo(
        config_path,
        timeframe=timeframe.value if timeframe == Timeframe.H1 else None,
    )

    output = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines() if "=" in line)
    development_id = output["development_run_id"]
    holdout_id = output["holdout_run_id"]
    development = load_run(development_id, artifacts_dir=artifacts)
    holdout = load_run(holdout_id, artifacts_dir=artifacts)
    baseline = next(item for item in development.artifact_manifest["candidates"] if item.get("model") == "baseline")
    candidate_predictions = load_candidate_predictions(development_id, "baseline", artifacts_dir=artifacts)
    saved_variant = evaluate_saved_predictions(
        development_id,
        allocation_overrides={"max_position_size": 0.10},
        model_name="baseline",
    )
    locator_path = Path(output["locator_path"])
    locator = json.loads(locator_path.read_text())

    assert baseline["status"] == "completed"
    assert baseline["selection_metric"] in {"mse", "log_loss"}
    if timeframe == Timeframe.D1:
        xgboost = next(item for item in development.artifact_manifest["candidates"] if item.get("model") == "xgboost")
        assert xgboost["status"] == "completed"
        assert xgboost["selection_metric"] in {"mse", "log_loss"}
        assert any(metric["name"] in {"mse", "log_loss"} for metric in xgboost["metrics"])
        assert len(xgboost["selected_durations"]) == 3
        assert all(1 <= duration <= 8 for duration in xgboost["selected_durations"])
        assert len(xgboost["stopping_diagnostics"]) == 3
        assert all(item["selected_duration"] in xgboost["selected_durations"] for item in xgboost["stopping_diagnostics"])
    frequency = development.artifact_manifest["config"]["statistics"]["frequency"]
    convention = return_convention(frequency)
    assert development.artifact_manifest["return_convention"] == convention
    assert development.artifact_manifest["statistics_calendar"] == development.snapshot.calendar
    evidence_metrics = [metric for metric in development.metrics if metric.name in {"sharpe", "psr", "dsr"}]
    assert evidence_metrics
    assert all(metric.conventions["return_convention"] == convention for metric in evidence_metrics)
    native_count = len(development.backtest.net_returns)
    statistical_count = len(
        statistical_returns(
            development.backtest.equity.get_column("timestamp").to_list(),
            development.backtest.net_returns.to_numpy(),
            frequency=frequency,
            calendar=development.snapshot.calendar,
        )
    )
    if timeframe == Timeframe.H1 and frequency == "daily":
        assert statistical_count < native_count
    ledger = TrialLedger(artifacts / "trials.sqlite")
    try:
        trial = ledger.trials()[0]
        attempt = ledger.attempts(trial["logical_trial_id"])[0]
    finally:
        ledger.connection.close()
    assert trial["settings"]["return_convention"] == convention
    assert trial["settings"]["calendar"] == development.snapshot.calendar
    assert attempt["result"]["return_convention"] == convention
    assert len(attempt["result"]["development_net_returns"]) == statistical_count
    assert (
        development.artifact_manifest["predicted_coverage"]["total"]
        >= development.artifact_manifest["scored_coverage"]["total"]
        > 0
    )
    assert sum(len(frame.keys) for frame in candidate_predictions) == development.artifact_manifest["predicted_coverage"]["total"]
    assert development.snapshot is not None
    assert holdout.ref.parent_run_id == development_id
    holdout_selection = next(
        item["selected_model"] for item in holdout.artifact_manifest["candidates"] if item.get("selected_model")
    )
    assert holdout_selection == output["selected_model"]
    assert saved_variant.ref.parent_run_id == development_id
    assert report.is_file() and report.stat().st_size > 0
    assert locator_path.is_file()
    assert locator["development"]["run_id"] == development_id
    assert locator["holdout"]["run_id"] == holdout_id
    assert output["selected_model"] in {
        item["model"] for item in development.artifact_manifest["candidates"] if item.get("status") == "completed"
    }
