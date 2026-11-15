from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import lab.demo as demo
from lab.core.config import PipelineConfig, Timeframe
from lab.demo import run_demo
from lab.demo_phase1 import run_phase1_demo
from lab.platform.synthetic_data import HOURLY_DEMO_GENERATOR_VERSION, demo_metadata, generate_demo_bars
from lab.quant.timing import bar_times_for_session, get_exchange_calendar
from lab.research.trial_ledger import TrialLedger


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
    expected = [item for session in sessions for item in bar_times_for_session("XNYS", session, Timeframe.H1)]
    bars = generate_demo_bars(n_sessions=300, tickers=("TEST",), timeframe=Timeframe.H1)
    actual = list(bars.select("bar_open_time", "bar_close_time").iter_rows())

    assert actual == [(opened, closed) for opened, closed, _session_id in expected]
    assert any((closed - opened).total_seconds() < 3600 for opened, closed in actual)
    assert demo_metadata(timeframe=Timeframe.H1)["generator_version"] == HOURLY_DEMO_GENERATOR_VERSION


def test_demo_entrypoint_is_callable():
    assert callable(run_demo)


def test_default_entrypoint_uses_phase1_workflow(monkeypatch, capsys, tmp_path):
    calls = []
    from lab import demo_phase1

    monkeypatch.setattr(
        demo_phase1,
        "run_phase1_demo",
        lambda **kwargs: (
            calls.append(kwargs)
            or {
                "report_path": "report.html",
                "selected_run_id": "selected-run",
                "K": 20,
                "candidate_selection": {"verdict": "REJECTED_BY_DSR", "score": 0.8, "reason": "SCORE_BELOW_THRESHOLD"},
                "decision": {"verdict": "REJECTED_BY_BASELINE"},
            }
        ),
    )
    assert demo.main(["--store-root", str(tmp_path)]) == 0
    assert calls == [{"store_root": tmp_path}]
    output = capsys.readouterr().out
    assert "candidate_selection_run_id=selected-run" in output
    assert "candidate_selection_verdict=REJECTED_BY_DSR" in output
    assert "candidate_selection_score=0.8" in output
    assert "candidate_selection_reason=SCORE_BELOW_THRESHOLD" in output
    assert "gate_K=20" in output
    assert "holdout_verdict=REJECTED_BY_BASELINE" in output


@pytest.mark.parametrize(
    ("arguments", "expected_config", "expected_timeframe"),
    [
        (["--config", "demo.yaml"], Path("demo.yaml"), None),
        (["--timeframe", "1h"], Path("config/demo.yaml"), "1h"),
    ],
)
def test_explicit_legacy_arguments_run_development_demo_only(monkeypatch, arguments, expected_config, expected_timeframe):
    calls = []
    monkeypatch.setattr(demo, "run_demo", lambda config, *, timeframe: calls.append((config, timeframe)))

    assert demo.main(arguments) == 0
    assert calls == [(expected_config, expected_timeframe)]


def test_explicit_daily_timeframe_rejects_hourly_config(tmp_path, monkeypatch):
    config = PipelineConfig.from_yaml("config/demo.yaml").model_dump(mode="json")
    config["data"]["timeframe"] = "1h"
    config_path = tmp_path / "hourly-demo.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(demo, "run_experiment", lambda *_args: (_ for _ in ()).throw(AssertionError("experiment started")))

    with pytest.raises(ValueError, match="Cannot override a 1h demo config with --timeframe 1d"):
        run_demo(config_path, timeframe="1d")


def test_legacy_demo_writes_development_report_without_requesting_holdout(tmp_path, monkeypatch, capsys):
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
    calls = {}

    monkeypatch.setattr(demo, "get_platform_config", lambda: platform)
    monkeypatch.setattr(demo, "run_experiment", lambda config: development)

    def fake_report(path, **kwargs):
        calls["report_type"] = kwargs["report_type"]
        calls["gate_decision"] = kwargs.get("gate_decision")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>development</html>")
        return path

    def fake_locator(path, **kwargs):
        path.write_text("{}")
        return path

    monkeypatch.setattr(
        demo,
        "evaluate_holdout",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("holdout requested")),
        raising=False,
    )
    monkeypatch.setattr(demo, "generate_run_report", fake_report)
    monkeypatch.setattr(demo, "write_demo_locator", fake_locator)

    config_path = tmp_path / "demo.yaml"
    config_path.write_text("data:\n  source: bundled_demo\n")
    report = run_demo(config_path)

    output = capsys.readouterr().out
    assert calls["report_type"] == "development"
    assert calls["gate_decision"] is None
    assert "development_run_id=dev-1" in output
    assert "holdout_run_id=" not in output
    assert "holdout_verdict=" not in output
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
    monkeypatch.setattr(demo, "get_platform_config", lambda: platform)
    monkeypatch.setattr(demo, "run_experiment", lambda config: development)
    monkeypatch.setattr(
        demo,
        "evaluate_holdout",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("holdout requested")),
        raising=False,
    )
    monkeypatch.setattr(demo, "generate_run_report", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("report failed")))
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("data:\n  source: bundled_demo\n")

    with pytest.raises(RuntimeError, match="report failed"):
        run_demo(config_path)

    assert capsys.readouterr().out == ""


def test_null_demo_selects_highest_finite_score_even_when_it_rejects_and_breaks_ties_stably():
    from lab.demo_phase1 import _highest_scored_candidate

    selected = _highest_scored_candidate(
        [
            ("run-z", {"verdict": "REJECTED_BY_DSR", "evidence": {"score": 0.81}}),
            ("run-b", {"verdict": "PASSED", "evidence": {"score": 0.72}}),
            ("run-a", {"verdict": "REJECTED_BY_BASELINE", "evidence": {"score": None}}),
            ("run-c", {"verdict": "REJECTED_BY_DSR", "evidence": {"score": 0.61}}),
            ("run-sealed", {"verdict": "REJECTED_HOLDOUT_SEALED", "evidence": {"score": 0.99}}),
        ]
    )

    assert selected == ("run-z", {"verdict": "REJECTED_BY_DSR", "evidence": {"score": 0.81}})
    assert (
        _highest_scored_candidate(
            [
                ("run-z", {"verdict": "PASSED", "evidence": {"score": 0.72}}),
                ("run-b", {"verdict": "PASSED", "evidence": {"score": 0.72}}),
            ]
        )[0]
        == "run-b"
    )


def test_null_demo_ignores_nonfinite_scores():
    from lab.demo_phase1 import _highest_scored_candidate

    assert _highest_scored_candidate([("run-a", {"verdict": "REJECTED_BY_DSR", "evidence": {"score": float("nan")}})]) is None


def test_null_demo_requires_canonical_gate_k_of_twenty():
    from lab.demo_phase1 import _require_demo_trial_count

    assert _require_demo_trial_count({"evidence": {"trial_count": 20}}) == 20
    with pytest.raises(RuntimeError, match="Expected holdout gate K = 20; got 19"):
        _require_demo_trial_count({"evidence": {"trial_count": 19}})


def test_null_demo_end_to_end_asserts_canonical_k_and_leaves_no_seal(tmp_path):
    outcome = run_phase1_demo(store_root=tmp_path)

    assert outcome["completed_variations"] == 20
    assert outcome["K"] == outcome["decision"]["evidence"]["trial_count"] == 20
    assert outcome["decision"]["verdict"] in {"REJECTED_BY_BASELINE", "REJECTED_BY_DSR"}
    assert Path(outcome["report_path"]).is_file()
    assert Path(outcome["locator_path"]).is_file()
    ledger = TrialLedger.read_only(tmp_path / "trials.sqlite")
    try:
        assert ledger.seal(outcome["snapshot_hash"]) is None
    finally:
        ledger.close()
