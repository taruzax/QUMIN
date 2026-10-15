from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import nbformat
import polars as pl
import pytest

from lab.core.config import AllocationConfig, BacktestConfig, TaskConfig
from lab.core.contracts import AllocationFrame, FoldBundle, PredictionFrame, SampleKey, SampleSet, ScoringTargets, WindowSet
from lab.models.baseline import EmpiricalPriorBaseline, ZeroForecastBaseline
from lab.platform.artifacts import ArtifactStore
from lab.quant.constraints import apply_all_constraints
from lab.quant.covariance import estimate_ledoit_wolf
from lab.quant.hrp import hrp_pipe
from lab.research.experiment import _write_scoring_targets, evaluate_predictions, load_scoring_targets
from lab.research.portfolio import predictions_to_signals
from lab.research.trial_ledger import TrialLedger


def test_baselines_follow_local_model_contract(tmp_path):
    X = np.ones((4, 3, 2))
    regression = ZeroForecastBaseline().fit(X, np.ones(4))
    assert np.array_equal(regression.predict(X), np.zeros(4))
    regression.save(tmp_path / "regression")
    assert np.array_equal(ZeroForecastBaseline.load(tmp_path / "regression").predict(X), np.zeros(4))

    classification = EmpiricalPriorBaseline().fit(X, np.array([-1, 0, 0, 1]))
    assert classification.predict_proba(X).shape == (4, 3)
    assert np.allclose(classification.predict_proba(X).sum(axis=1), 1.0)


def test_baselines_validate_and_predict_lazy_windows_without_materializing(monkeypatch):
    windows = WindowSet(
        [np.ones((5, 2), dtype=np.float32)], np.zeros(3, dtype=np.int64), np.arange(3), 3,
    )
    monkeypatch.setattr(WindowSet, "to_numpy", lambda self: (_ for _ in ()).throw(AssertionError("unexpected materialization")))

    regression = ZeroForecastBaseline().fit(windows, np.ones(3))
    classification = EmpiricalPriorBaseline().fit(windows, np.asarray([-1, 0, 1]))

    assert regression.predict(windows).shape == (3,)
    assert classification.predict_proba(windows).shape == (3, 3)


def test_artifacts_are_write_once_and_ledger_records_retry(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    run_id = store.create_run("run-fixed")
    record = store.write_json(run_id, "input.json", {"x": 1})
    assert store.verify_artifact(run_id, record).exists()
    with pytest.raises(FileExistsError):
        store.write_json(run_id, "input.json", {"x": 2})
    ledger = TrialLedger(tmp_path / "trials.sqlite")
    args = {"campaign": "campaign", "config_hash": "config", "snapshot_hash": "snapshot", "evaluation_hash": "split", "settings": {}}
    logical, first, retry = ledger.register_attempt(**args)
    _, second, retry_again = ledger.register_attempt(**args)
    assert logical and first != second and not retry and retry_again
    ledger.close()


def test_ledger_comparable_evidence_keeps_failed_and_completed_attempts(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite")
    args = {
        "campaign": "campaign",
        "config_hash": "config",
        "snapshot_hash": "snapshot",
        "evaluation_hash": "split",
        "settings": {"frequency": "daily", "cost_treatment": "fees+slippage"},
    }
    logical, attempt, _ = ledger.register_attempt(**args)
    ledger.update_attempt(attempt, status="failed", error="candidate failed")
    _, second, _ = ledger.register_attempt(**args)
    ledger.update_attempt(second, status="completed", result={"development_net_returns": [0.01, -0.01]})

    evidence = ledger.comparable_evidence(snapshot_hash="snapshot", evaluation_hash="split", frequency="daily")

    assert evidence[0]["logical_trial_id"] == logical
    assert {item["status"] for item in evidence[0]["attempts"]} == {"failed", "completed"}
    ledger.close()


def test_constraints_do_not_reinvest_dropped_cash():
    config = AllocationConfig(max_position_size=0.6, min_position_size=0.2, gross_limit=1.0)
    assert apply_all_constraints({"AAA": 0.8, "BBB": 0.1}, config) == {"AAA": 0.6}


def test_signal_policy_handles_dead_zone_and_classification_ties():
    assert predictions_to_signals(np.array([-0.1, 0.0, 0.1]), dead_zone=0.05).tolist() == [-1, 0, 1]
    probabilities = np.array([[0.2, 0.6, 0.2], [0.5, 0.5, 0.0]])
    assert predictions_to_signals(probabilities, task="classification", confidence=None).tolist() == [0, 0]


def test_covariance_and_hrp_use_labeled_raw_returns():
    rng = np.random.default_rng(42)
    returns = pl.DataFrame({"timestamp": list(range(300)), "AAA": rng.normal(0, 0.01, 300), "BBB": rng.normal(0, 0.02, 300)})
    covariance, diagnostics = estimate_ledoit_wolf(returns)
    assert diagnostics.observations == 300
    assert covariance.columns == ["ticker", "AAA", "BBB"]
    weights = hrp_pipe(returns)
    assert set(weights) == {"AAA", "BBB"}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_notebook_checkpoint_source_and_cell_state_are_well_formed():
    notebook = json.loads(open("notebooks/research_walkthrough.ipynb").read())
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None or isinstance(cell.get("execution_count"), int) for cell in code_cells)
    assert all(isinstance(cell.get("outputs", []), list) for cell in code_cells)
    assert all(
        isinstance(output, dict)
        and output.get("output_type") in {"stream", "display_data", "execute_result", "error"}
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "sharpe_evidence_metrics" in source
    assert "SNAPSHOT_ID" in source
    assert "plan_campaign(config, campaign_spec)" in source
    assert "current_evidence['included_logical_trial_ids']" in source


@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_contract_inputs_are_explicit(kind):
    config = TaskConfig(kind=kind)
    assert config.kind == kind
    key = SampleKey(ticker="AAA", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))
    allocation = AllocationFrame(
        decision_time=key.timestamp,
        valid_from=key.timestamp,
        ticker_budgets={},
        config_hash="config",
    )
    assert allocation.ticker_budgets == {}


def test_vectorbt_execution_dependency_is_optional_at_import_time():
    from lab.quant.backtest import simulate_strategy

    assert callable(simulate_strategy)


def _scoring_fixture(task="regression", scoring_values=None, *, fold_ref="0", start_day=1):
    timestamps = [datetime(2024, 1, day, tzinfo=timezone.utc) for day in range(start_day, start_day + 3)]
    keys = tuple(SampleKey(ticker="AAA", timestamp=value) for value in timestamps)
    samples = SampleSet(
        X=WindowSet([np.ones((4, 1))], np.zeros(3, dtype=np.int64), np.arange(3), 2),
        keys=keys,
        window_start=tuple(value - timedelta(days=1) for value in timestamps),
        window_end=tuple(timestamps),
        feature_order=("close",),
    )
    values = np.asarray(scoring_values if scoring_values is not None else [0.3, -0.2], dtype=np.float32)
    scoring_keys = (keys[0], keys[2]) if len(values) else ()
    bundle = FoldBundle(
        fold_ref=fold_ref, train_samples=samples, evaluation_samples=samples,
        scoring_keys=scoring_keys,
        scoring_targets=ScoringTargets(keys=scoring_keys, values=values),
        preprocessor=None, boundaries={},
    )
    return bundle, keys


def test_prediction_metrics_match_by_key_when_frame_order_changes():
    bundle, keys = _scoring_fixture()
    predictions = np.asarray([0.5, 99.0, -0.1])
    ordered = PredictionFrame(keys=keys, model_ref="model", fold_ref="0", predictions=predictions)
    reverse = PredictionFrame(
        keys=tuple(reversed(keys)), model_ref="model", fold_ref="0",
        predictions=predictions[::-1],
    )

    left = evaluate_predictions((ordered,), (bundle,), task="regression")
    right = evaluate_predictions((reverse,), (bundle,), task="regression")

    assert [(metric.name, metric.value) for metric in left] == [(metric.name, metric.value) for metric in right]
    assert next(metric for metric in left if metric.name == "mse").value == pytest.approx(0.025)


def test_classification_metrics_match_by_key_when_frame_order_changes():
    bundle, keys = _scoring_fixture(task="classification", scoring_values=[-1, 1])
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
    ordered = PredictionFrame(keys=keys, model_ref="model", fold_ref="0", predictions=probabilities)
    reverse = PredictionFrame(
        keys=tuple(reversed(keys)), model_ref="model", fold_ref="0", predictions=probabilities[::-1],
    )

    left = evaluate_predictions((ordered,), (bundle,), task="classification")
    right = evaluate_predictions((reverse,), (bundle,), task="classification")

    assert [(metric.name, metric.value) for metric in left] == [(metric.name, metric.value) for metric in right]


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_prediction_metrics_aggregate_folds_with_distinct_model_bundle_refs(task):
    scoring_values = [-1, 1] if task == "classification" else [0.3, -0.2]
    first, first_keys = _scoring_fixture(task=task, scoring_values=scoring_values, fold_ref="0", start_day=1)
    second, second_keys = _scoring_fixture(task=task, scoring_values=scoring_values, fold_ref="1", start_day=4)
    if task == "regression":
        first_values, second_values = np.asarray([0.2, -0.1, 0.4]), np.asarray([0.1, -0.2, 0.3])
    else:
        first_values = np.full((3, 3), 1 / 3)
        second_values = np.full((3, 3), 1 / 3)
    frames = (
        PredictionFrame(keys=first_keys, model_ref="models/xgboost-fold-0", fold_ref="0", predictions=first_values),
        PredictionFrame(keys=second_keys, model_ref="models/xgboost-fold-1", fold_ref="1", predictions=second_values),
    )

    metrics = evaluate_predictions(frames, (first, second), task=task)

    primary = next(metric for metric in metrics if metric.name == ("mse" if task == "regression" else "log_loss"))
    assert primary.status == "available"
    assert primary.inputs["n"] == 4
    assert primary.value == pytest.approx(0.165 if task == "regression" else np.log(3.0))


def test_prediction_metrics_report_empty_scoring_without_requiring_empty_predictions():
    bundle, keys = _scoring_fixture(scoring_values=[])
    frame = PredictionFrame(keys=keys, model_ref="model", fold_ref="0", predictions=np.asarray([0.2, 0.3, 0.4]))

    metrics = evaluate_predictions((frame,), (bundle,), task="regression")

    assert metrics
    assert all(metric.status == "unavailable" for metric in metrics)
    assert all(metric.reason == "no scoreable samples" for metric in metrics)


def test_prediction_metrics_reject_missing_evaluation_keys():
    bundle, keys = _scoring_fixture()
    frame = PredictionFrame(keys=keys[:2], model_ref="model", fold_ref="0", predictions=np.asarray([0.1, 0.2]))

    with pytest.raises(ValueError, match="cover every evaluation key"):
        evaluate_predictions((frame,), (bundle,), task="regression")


def test_prediction_metrics_reject_wrong_fold_identity_and_task_shape():
    bundle, keys = _scoring_fixture()
    wrong_fold = PredictionFrame(keys=keys, model_ref="model", fold_ref="other", predictions=np.zeros(3))
    wrong_shape = PredictionFrame(
        keys=keys, model_ref="model", fold_ref="0", predictions=np.full((3, 3), 1 / 3),
    )

    with pytest.raises(ValueError, match="fold identities"):
        evaluate_predictions((wrong_fold,), (bundle,), task="regression")
    with pytest.raises(ValueError, match="Regression predictions must be one-dimensional"):
        evaluate_predictions((wrong_shape,), (bundle,), task="regression")


def test_prediction_metrics_reject_frame_bundle_count_mismatch():
    bundle, _ = _scoring_fixture()

    with pytest.raises(ValueError, match="frame count"):
        evaluate_predictions((), (bundle,), task="regression")


def test_scoring_target_artifact_round_trips_by_fold_and_key(tmp_path):
    bundle, _ = _scoring_fixture()
    store = ArtifactStore(tmp_path / "artifacts")
    run_id = store.create_run("scoring-fixture")
    record = _write_scoring_targets(store, run_id, bundle)
    store.finalize(
        run_id,
        {"scored_coverage": {"artifacts": [record["path"]]}, "artifacts": [record]},
        required_artifacts=[record["path"]],
    )

    restored = load_scoring_targets(run_id, artifacts_dir=tmp_path / "artifacts")

    assert restored["0"].keys == bundle.scoring_keys
    np.testing.assert_array_equal(restored["0"].values, bundle.scoring_targets.values)
