from pathlib import Path
from types import SimpleNamespace

import pytest

import lab.demo as demo
from lab.core.config import PipelineConfig, Timeframe
from lab.demo import run_demo
from lab.platform.synthetic_data import HOURLY_DEMO_GENERATOR_VERSION, demo_metadata, generate_demo_bars
from lab.quant.timing import bar_times_for_session, get_exchange_calendar


def test_demo_configuration_is_bundled_and_offline():
    assert Path("config/demo.yaml").exists()
    assert Path("data/demo/ohlcv.csv").exists()
    assert Path("data/demo/metadata.yaml").exists()


def test_demo_explicitly_sets_task_and_stopping_policy():
    config = PipelineConfig.from_yaml("config/demo.yaml")

    assert config.data.source == "bundled_demo"
    assert config.task.kind == "regression"
    assert config.models.early_stopping is False
    assert config.models.early_stopping_fraction == 0.20
    assert config.models.early_stopping_patience == 5


def test_hourly_demo_emits_every_exchange_calendar_slot_including_partial_bars():
    calendar = get_exchange_calendar("XNYS")
    sessions = calendar.sessions[calendar.sessions <= "2024-12-31"][-300:]
    expected = [
        item
        for session in sessions
        for item in bar_times_for_session("XNYS", session, Timeframe.H1)
    ]
    bars = generate_demo_bars(n_sessions=300, tickers=("TEST",), timeframe=Timeframe.H1)
    actual = list(bars.select("bar_open_time", "bar_close_time").iter_rows())

    assert actual == [(opened, closed) for opened, closed, _session_id in expected]
    assert any((closed - opened).total_seconds() < 3600 for opened, closed in actual)
    assert demo_metadata(timeframe=Timeframe.H1)["generator_version"] == HOURLY_DEMO_GENERATOR_VERSION


def test_demo_entrypoint_is_callable():
    assert callable(run_demo)


def test_demo_selects_completed_candidate_and_prints_both_runs(tmp_path, monkeypatch, capsys):
    artifact_root = tmp_path / "artifacts"
    platform = SimpleNamespace(paths=SimpleNamespace(artifacts_dir=artifact_root))
    development = SimpleNamespace(
        ref=SimpleNamespace(run_id="dev-1", snapshot_hash="snapshot"),
        backtest=SimpleNamespace(net_returns=object()),
        artifact_manifest={
            "candidates": [
                {"model": "baseline", "status": "completed"},
                {"selected_model": "baseline"},
            ]
        },
    )
    holdout = SimpleNamespace(ref=SimpleNamespace(run_id="dev-1-holdout"), backtest=SimpleNamespace(net_returns=object()))
    calls = {}

    monkeypatch.setattr(demo, "get_platform_config", lambda: platform)
    monkeypatch.setattr(demo, "run_experiment", lambda config: development)

    def fake_holdout(result, *, config, model_name):
        calls["model_name"] = model_name
        return holdout

    def fake_report(path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>demo</html>")
        return path

    def fake_locator(path, **kwargs):
        path.write_text("{}")
        return path

    monkeypatch.setattr(demo, "evaluate_holdout", fake_holdout)
    monkeypatch.setattr(demo, "generate_demo_report", fake_report)
    monkeypatch.setattr(demo, "write_demo_locator", fake_locator)

    config_path = tmp_path / "demo.yaml"
    config_path.write_text("data:\n  source: bundled_demo\n")
    report = run_demo(config_path)

    output = capsys.readouterr().out
    assert calls["model_name"] == "baseline"
    assert "development_run_id=dev-1" in output
    assert "holdout_run_id=dev-1-holdout" in output
    assert report.exists()
    assert "report_path=" in output
    assert "locator_path=" in output


def test_demo_does_not_announce_success_when_report_creation_fails(tmp_path, monkeypatch, capsys):
    platform = SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"))
    development = SimpleNamespace(
        ref=SimpleNamespace(run_id="dev-1"),
        backtest=SimpleNamespace(net_returns=object()),
        artifact_manifest={"candidates": [{"model": "baseline", "status": "completed"}, {"selected_model": "baseline"}]},
    )
    holdout = SimpleNamespace(ref=SimpleNamespace(run_id="holdout-1"), backtest=SimpleNamespace(net_returns=object()))
    monkeypatch.setattr(demo, "get_platform_config", lambda: platform)
    monkeypatch.setattr(demo, "run_experiment", lambda config: development)
    monkeypatch.setattr(demo, "evaluate_holdout", lambda *args, **kwargs: holdout)
    monkeypatch.setattr(demo, "generate_demo_report", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("report failed")))
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("data:\n  source: bundled_demo\n")

    with pytest.raises(RuntimeError, match="report failed"):
        run_demo(config_path)

    assert capsys.readouterr().out == ""
