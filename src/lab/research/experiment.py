"""Reusable saved baseline experiment and explicit holdout APIs."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lab.core.config import PIPELINE_CONFIG_SCHEMA_VERSION, PipelineConfig, get_platform_config
from lab.core.contracts import (
    AllocationFrame,
    BacktestResult,
    CampaignEvidenceReport,
    FoldBundle,
    FoldSpec,
    MarketSnapshot,
    MetricResult,
    PredictionFrame,
    PreparedDataset,
    PreparedDatasetRef,
    RunRef,
    RunResult,
    SampleKey,
    ScoringTargets,
)
from lab.models.registry import create_model
from lab.platform.artifacts import ArtifactStore, sha256_file
from lab.platform.evidence_store import EvidenceStore
from lab.platform.mlflow_adapter import log_local_model
from lab.quant.backtest import simulate_strategy
from lab.quant.metrics import (
    classification_metrics,
    regression_metrics,
    return_convention,
    sharpe_evidence_metrics,
    sharpe_statistics,
    statistical_returns,
    unavailable_metric,
)
from lab.research.holdout_gate import HoldoutRejected, evaluate_gate
from lab.research.portfolio import build_allocations
from lab.research.preprocessing import prepare_dataset, prepare_fold
from lab.research.training import train_candidate
from lab.research.trial_ledger import TrialLedger


@dataclass(frozen=True)
class FoldRun:
    fold_id: int
    bundle: FoldBundle
    model_ref: str
    predictions: PredictionFrame
    metrics: tuple[MetricResult, ...]


def _split_hash(dataset: PreparedDataset) -> str:
    payload = json.dumps(dataset.split_plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _code_identity() -> tuple[str, str]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or status.returncode != 0:
        return "unknown", "unknown"
    return revision.stdout.strip(), hashlib.sha256(status.stdout.encode()).hexdigest()


def train_fold(
    bundle: FoldBundle,
    config: PipelineConfig,
    *,
    model_name: str = "baseline",
    dataset: PreparedDataset | None = None,
    fold: FoldSpec | None = None,
    duration: int | None = None,
):
    return train_candidate(bundle, config, model_name=model_name, dataset=dataset, fold=fold, duration=duration)


def predict_fold(model: Any, bundle: FoldBundle, *, model_ref: str, fold_ref: str, task: str) -> PredictionFrame:
    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")
    if fold_ref != bundle.fold_ref:
        raise ValueError(f"Prediction fold identity {fold_ref!r} does not match bundle {bundle.fold_ref!r}")
    if task == "classification":
        predictions = model.predict_proba(bundle.evaluation_samples.X)
    else:
        predictions = model.predict(bundle.evaluation_samples.X)
    return PredictionFrame(keys=bundle.evaluation_samples.keys, model_ref=model_ref, fold_ref=fold_ref, predictions=predictions)


def evaluate_predictions(
    prediction_frames: tuple[PredictionFrame, ...], bundles: tuple[FoldBundle, ...], *, task: str
) -> tuple[MetricResult, ...]:
    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")
    if len(prediction_frames) != len(bundles):
        raise ValueError("Prediction frame count must match fold bundle count")
    bundle_by_fold = {bundle.fold_ref: bundle for bundle in bundles}
    if len(bundle_by_fold) != len(bundles):
        raise ValueError("Fold bundle identities must be unique")
    frame_by_fold = {frame.fold_ref: frame for frame in prediction_frames}
    if len(frame_by_fold) != len(prediction_frames):
        raise ValueError("Prediction fold identities must be unique")
    if set(frame_by_fold) != set(bundle_by_fold):
        raise ValueError("Prediction fold identities must match fold bundles")

    actual_by_key: dict[tuple[str, datetime], float] = {}
    predicted_by_key: dict[tuple[str, datetime], Any] = {}
    for fold_ref, bundle in bundle_by_fold.items():
        prediction = frame_by_fold[fold_ref]
        expected_keys = {_sample_key_id(key) for key in bundle.evaluation_samples.keys}
        frame_keys = [_sample_key_id(key) for key in prediction.keys]
        if len(set(frame_keys)) != len(frame_keys):
            raise ValueError(f"Prediction keys must be unique in fold {fold_ref}")
        if set(frame_keys) != expected_keys:
            raise ValueError(f"Predictions must cover every evaluation key exactly in fold {fold_ref}")
        if task == "regression" and prediction.predictions.ndim != 1:
            raise ValueError("Regression predictions must be one-dimensional")
        if task == "classification" and prediction.predictions.ndim != 2:
            raise ValueError("Classification predictions must have shape [N, 3]")
        if any(key in predicted_by_key for key in frame_keys):
            raise ValueError("Prediction keys must be unique across fold bundles")
        predicted_by_key.update(zip(frame_keys, prediction.predictions))
        target_values = bundle.scoring_targets.values
        target_keys = [_sample_key_id(key) for key in bundle.scoring_keys]
        if len(target_keys) != len(target_values):
            raise ValueError(f"Scoring targets are misaligned in fold {fold_ref}")
        if any(key in actual_by_key for key in target_keys):
            raise ValueError("Scoring keys must be unique across fold bundles")
        actual_by_key.update(zip(target_keys, target_values.tolist()))

    scoring_order = [_sample_key_id(key) for bundle in bundles for key in bundle.scoring_keys]
    if len(set(scoring_order)) != len(scoring_order):
        raise ValueError("Scoring keys must be unique across fold bundles")
    if any(key not in predicted_by_key for key in scoring_order):
        raise ValueError("A scoring key is missing its required prediction")
    actual = [actual_by_key[key] for key in scoring_order]
    predicted = [predicted_by_key[key] for key in scoring_order]
    if not actual:
        names = ("log_loss", "accuracy") if task == "classification" else ("mse", "rmse", "mae", "improvement_over_zero")
        return tuple(unavailable_metric(name, "no scoreable samples", n=0) for name in names)
    if task == "classification":
        labels = np.asarray(actual, dtype=float)
        if (
            not np.isfinite(labels).all()
            or not np.equal(labels, labels.astype(int)).all()
            or not np.isin(labels, (-1, 0, 1)).all()
        ):
            raise ValueError("Classification scoring targets must be integer labels in {-1, 0, +1}")
        return tuple(classification_metrics(actual, np.asarray(predicted)).values())
    return tuple(regression_metrics(actual, predicted).values())


def _sample_key_id(key: SampleKey) -> tuple[str, datetime]:
    return key.ticker, key.timestamp.astimezone(timezone.utc)


def _write_prediction(
    store: ArtifactStore,
    run_id: str,
    predictions: tuple[PredictionFrame, ...],
    relative_path: str = "predictions.json",
) -> dict[str, Any]:
    records = []
    for frame in predictions:
        for key, value in zip(frame.keys, frame.predictions):
            records.append(
                {
                    "ticker": key.ticker,
                    "timestamp": key.timestamp.isoformat(),
                    "model_ref": frame.model_ref,
                    "fold_ref": frame.fold_ref,
                    "prediction": np.asarray(value).tolist(),
                }
            )
    return store.write_json(run_id, relative_path, records)


def _write_scoring_targets(store: ArtifactStore, run_id: str, bundle: FoldBundle) -> dict[str, Any]:
    """Persist keyed score labels and their fold coverage diagnostics."""
    payload = {
        "schema_version": "scoring-target-artifact.v1",
        "fold_ref": bundle.fold_ref,
        "targets": [
            {"ticker": key.ticker, "timestamp": key.timestamp.isoformat(), "target": float(value)}
            for key, value in zip(bundle.scoring_targets.keys, bundle.scoring_targets.values)
        ],
        "diagnostics": bundle.diagnostics.get("scoring", {}),
    }
    return store.write_json(run_id, f"scoring/fold-{bundle.fold_ref}.json", payload)


def _combined_predictions(frames: tuple[PredictionFrame, ...]) -> list[tuple[SampleKey, np.ndarray]]:
    return [(key, np.asarray(value)) for frame in frames for key, value in zip(frame.keys, frame.predictions)]


def _strategy_result(
    dataset: PreparedDataset,
    predictions: tuple[PredictionFrame, ...],
    config: PipelineConfig,
    *,
    period_name: str,
) -> tuple[tuple[AllocationFrame, ...], BacktestResult]:
    rows = _combined_predictions(predictions)
    period = next(item for item in dataset.split_plan.execution_periods if item.name == period_name)
    # Build allocations only from bars inside the scored execution interval.
    # Keep pre-period history available for causal allocation warm-up.
    if period_name == "development":
        allowed_bars = dataset.snapshot.bars.filter(pl.col("timestamp") < dataset.split_plan.holdout_start)
    else:
        allowed_bars = dataset.snapshot.bars.filter(pl.col("timestamp") < dataset.split_plan.holdout_end)
    allocation_snapshot = dataset.snapshot.model_copy(update={"bars": allowed_bars})
    allocations = build_allocations(
        allocation_snapshot,
        rows,
        config.allocation,
        task=config.task.kind,
        backtest_config=config.backtest,
        config_hash=config.resolved_config_hash,
    )
    if period.final_liquidation_open is not None:
        allocations = tuple(item for item in allocations if item.valid_from < period.final_liquidation_open)
    backtest = simulate_strategy(dataset.snapshot, rows, allocations, config.task, config.backtest, period)
    return allocations, backtest


def _statistical_backtest_returns(backtest: BacktestResult, *, frequency: str, calendar: str, gross: bool = False):
    """Convert immutable native backtest returns to the configured evidence frequency."""
    series = backtest.gross_returns if gross else backtest.net_returns
    if series is None:
        return None
    timestamps = backtest.equity.get_column("timestamp").to_list()
    return statistical_returns(timestamps, series.to_numpy(), frequency=frequency, calendar=calendar)


def persist_run(
    *,
    store: ArtifactStore,
    run_id: str,
    config: PipelineConfig,
    dataset: PreparedDataset,
    predictions: tuple[PredictionFrame, ...],
    metrics: tuple[MetricResult, ...],
    model_dirs: list[str],
    allocations: tuple[AllocationFrame, ...] = (),
    backtest: BacktestResult | None = None,
    bundles: tuple[FoldBundle, ...] = (),
    candidate_status: list[dict[str, Any]] | None = None,
    candidate_predictions: dict[str, tuple[PredictionFrame, ...]] | None = None,
    inherited_scoring_coverage: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
    parent_model_ref: str | None = None,
    reservation: dict[str, Any] | None = None,
    decision_id: str | None = None,
) -> RunResult:
    prediction_record = _write_prediction(store, run_id, predictions)
    candidate_prediction_records = []
    for candidate_name, candidate_frames in (candidate_predictions or {}).items():
        candidate_prediction_records.append(
            _write_prediction(store, run_id, candidate_frames, f"candidates/{candidate_name}/predictions.json")
        )
    metrics_record = store.write_json(run_id, "metrics.json", [metric.model_dump(mode="json") for metric in metrics])
    snapshot_record = store.write_parquet(run_id, "snapshot/bars.parquet", dataset.snapshot.bars)
    snapshot_metadata = store.write_json(
        run_id, "snapshot/metadata.json", dataset.snapshot.model_dump(mode="json", exclude={"bars"})
    )
    split_record = store.write_json(run_id, "split.json", dataset.split_plan.model_dump(mode="json"))
    allocation_record = store.write_json(run_id, "allocations.json", [item.model_dump(mode="json") for item in allocations])
    preprocessor_records = [
        store.write_json(run_id, f"preprocessors/fold-{index}.json", bundle.preprocessor.state.model_dump(mode="json"))
        for index, bundle in enumerate(bundles)
    ]
    scoring_records = [_write_scoring_targets(store, run_id, bundle) for bundle in bundles]
    records = [
        prediction_record,
        metrics_record,
        snapshot_record,
        snapshot_metadata,
        split_record,
        allocation_record,
        *preprocessor_records,
        *scoring_records,
        *candidate_prediction_records,
    ]
    for model_dir in model_dirs:
        for name in ("manifest.json", "model.pkl"):
            path = store.run_dir(run_id) / model_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Incomplete local model bundle: {path}")
            records.append(
                {"path": str(path.relative_to(store.run_dir(run_id))), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    backtest_paths: list[str] = []
    if backtest is not None:
        for name in ("orders", "trade_events", "equity", "exposure"):
            frame = getattr(backtest, name)
            if frame is not None:
                record = store.write_parquet(run_id, f"backtest/{name}.parquet", frame)
                records.append(record)
                backtest_paths.append(record["path"])
        series_record = store.write_json(
            run_id,
            "backtest/series.json",
            {
                "gross_returns": backtest.gross_returns.to_list() if backtest.gross_returns is not None else None,
                "net_returns": backtest.net_returns.to_list() if backtest.net_returns is not None else None,
                "turnover": backtest.turnover.to_list() if backtest.turnover is not None else None,
                "period": backtest.period,
                "diagnostics": backtest.diagnostics,
            },
        )
        records.append(series_record)
        backtest_paths.append(series_record["path"])
    revision, dirty_identity = _code_identity()
    manifest = {
        "config": config.model_dump(mode="json"),
        "resolved_config": config.model_dump(mode="json"),
        "config_schema_version": PIPELINE_CONFIG_SCHEMA_VERSION,
        "config_source_provenance": config.source_provenance,
        "config_hash": config.resolved_config_hash,
        "snapshot_hash": dataset.snapshot.snapshot_hash,
        "split_hash": _split_hash(dataset),
        "code_revision": revision,
        "dirty_state_identity": dirty_identity,
        "dependency_lock_hash": sha256_file("uv.lock") if Path("uv.lock").exists() else None,
        "seed": config.models.seed,
        "task": config.task.kind,
        "return_convention": return_convention(config.statistics.frequency),
        "execution_evidence_version": "period-bounded-next-open.v2",
        "snapshot_reservation": reservation,
        "decision_id": decision_id,
        "statistics_frequency": config.statistics.frequency,
        "statistics_calendar": dataset.snapshot.calendar,
        "model_bundles": model_dirs,
        "parent_model_ref": parent_model_ref,
        "preprocessor_bundles": [item["path"] for item in preprocessor_records],
        "prediction_keys": [
            {"ticker": key.ticker, "timestamp": key.timestamp.isoformat()} for frame in predictions for key in frame.keys
        ],
        "predicted_coverage": {
            "status": "available",
            "folds": {frame.fold_ref: len(frame.keys) for frame in predictions},
            "total": sum(len(frame.keys) for frame in predictions),
        },
        "scored_coverage": (
            {
                "status": "available",
                "folds": {bundle.fold_ref: len(bundle.scoring_keys) for bundle in bundles},
                "total": sum(len(bundle.scoring_keys) for bundle in bundles),
                "artifacts": [record["path"] for record in scoring_records],
            }
            if bundles
            else {
                "status": "inherited" if inherited_scoring_coverage else "not_recomputed",
                "source_run_id": parent_run_id,
                "coverage": inherited_scoring_coverage,
            }
        ),
        "candidates": candidate_status or [],
        "candidate_prediction_paths": {
            name: record["path"] for name, record in zip((candidate_predictions or {}).keys(), candidate_prediction_records)
        },
        "allocation_paths": [allocation_record["path"]],
        "backtest_paths": backtest_paths,
        "report_paths": [],
        "artifacts": records,
        "parent_run_id": parent_run_id,
    }
    store.finalize(run_id, manifest, required_artifacts=[item["path"] for item in records])
    ref = RunRef(
        run_id=run_id,
        snapshot_hash=dataset.snapshot.snapshot_hash,
        config_hash=config.resolved_config_hash,
        split_hash=_split_hash(dataset),
        model_ref=model_dirs[0] if model_dirs else parent_model_ref or "baseline",
        parent_run_id=parent_run_id,
    )
    return RunResult(
        ref=ref,
        predictions=predictions,
        allocations=allocations,
        metrics=metrics,
        backtest=backtest,
        snapshot=dataset.snapshot,
        artifact_manifest=manifest,
    )


def run_experiment(
    config: PipelineConfig,
    dataset: PreparedDataset | None = None,
    *,
    tracking_mode: str = "immediate",
    dataset_ref: PreparedDatasetRef | None = None,
) -> RunResult:
    """Evaluate every enabled candidate, persist failures, and select one result."""
    if tracking_mode not in {"immediate", "deferred", "disabled"}:
        raise ValueError("tracking_mode must be immediate, deferred, or disabled")
    if dataset is not None and dataset_ref is not None:
        raise ValueError("Pass either dataset or dataset_ref, not both")
    prepared = dataset or prepare_dataset(config, dataset_ref=dataset_ref)
    if dataset is not None or dataset_ref is not None:
        trusted = prepare_dataset(config)
        if (
            prepared.snapshot.snapshot_hash != trusted.snapshot.snapshot_hash
            or _split_hash(prepared) != _split_hash(trusted)
            or prepared.feature_specification != trusted.feature_specification
            or not prepared.features.equals(trusted.features)
            or not prepared.labels.equals(trusted.labels)
            or prepared.config.resolved_config_hash != config.resolved_config_hash
        ):
            raise ValueError("Supplied prepared dataset does not match rebuilt trusted source and effective configuration")
        prepared = trusted
    store = ArtifactStore(get_platform_config().paths.artifacts_dir)
    ledger = TrialLedger(get_platform_config().paths.artifacts_dir / "trials.sqlite")
    reservation = ledger.reserve_snapshot(
        snapshot_hash=prepared.snapshot.snapshot_hash,
        holdout_start=prepared.split_plan.holdout_start.isoformat(),
        holdout_end=prepared.split_plan.holdout_end.isoformat(),
    )
    logical_trial, attempt_id, _ = ledger.register_attempt(
        campaign=config.campaign.name,
        config_hash=config.resolved_config_hash,
        snapshot_hash=prepared.snapshot.snapshot_hash,
        evaluation_hash=_split_hash(prepared),
        settings={
            "task": config.task.kind,
            "models": config.models.model_dump(mode="json"),
            "candidate_context": {
                key: value for key, value in config.model_dump(mode="json").items() if key not in {"campaign", "data", "models"}
            },
            "frequency": config.statistics.frequency,
            "return_convention": return_convention(config.statistics.frequency),
            "execution_evidence_version": "period-bounded-next-open.v2",
            "calendar": prepared.snapshot.calendar,
            "cost_treatment": "fees+slippage",
        },
    )
    run_id = store.create_run()
    bundles: list[FoldBundle] = []
    candidate_status: list[dict[str, Any]] = []
    try:
        for fold in prepared.split_plan.folds:
            bundle = prepare_fold(prepared, fold)
            bundles.append(bundle)
        candidates: list[tuple[str, tuple[PredictionFrame, ...], tuple[MetricResult, ...], list[str]]] = []
        candidate_prediction_sets: dict[str, tuple[PredictionFrame, ...]] = {}
        for model_name in ("baseline", "xgboost", "gru", "lstm"):
            spec = getattr(config.models, model_name)
            if not spec.enabled:
                candidate_status.append({"model": model_name, "status": "disabled"})
                continue
            candidate_predictions: list[PredictionFrame] = []
            candidate_dirs: list[str] = []
            tracking_refs: list[dict[str, str]] = []
            selected_durations: list[int] = []
            stopping_diagnostics: list[dict[str, Any]] = []
            try:
                for fold_index, bundle in enumerate(bundles):
                    model = train_fold(
                        bundle,
                        config,
                        model_name=model_name,
                        dataset=prepared,
                        fold=prepared.split_plan.folds[fold_index],
                    )
                    if model.selected_duration is not None:
                        selected_durations.append(int(model.selected_duration))
                    if model.stopping_diagnostics:
                        stopping_diagnostics.append(model.stopping_diagnostics)
                    model_dir = store.run_dir(run_id) / "models" / f"{model_name}-fold-{len(candidate_dirs)}"
                    model.save(model_dir)
                    model_ref = str(model_dir.relative_to(store.run_dir(run_id)))
                    candidate_dirs.append(model_ref)
                    prediction = predict_fold(model, bundle, model_ref=model_ref, fold_ref=bundle.fold_ref, task=config.task.kind)
                    candidate_predictions.append(prediction)
                    platform = get_platform_config()
                    if platform.mlflow.enabled and tracking_mode == "immediate":
                        fold_metrics = evaluate_predictions((prediction,), (bundle,), task=config.task.kind)
                        tracking_refs.append(
                            log_local_model(
                                model,
                                artifact_path=f"{model_name}-fold-{len(candidate_dirs) - 1}",
                                model_dir=model_dir,
                                tracking_uri=platform.paths.mlflow_tracking_uri,
                                experiment_name=platform.mlflow.experiment_name,
                                params={
                                    "config_hash": config.resolved_config_hash,
                                    "snapshot_hash": prepared.snapshot.snapshot_hash,
                                    "split_hash": _split_hash(prepared),
                                    "task": config.task.kind,
                                    "model": model_name,
                                    "seed": config.models.seed,
                                    "fold": len(candidate_dirs) - 1,
                                    "model_bundle": model_ref,
                                },
                                metrics={
                                    item.name: item.value
                                    for item in fold_metrics
                                    if item.status == "available" and item.value is not None
                                },
                            )
                        )
                candidate_metrics = evaluate_predictions(tuple(candidate_predictions), tuple(bundles), task=config.task.kind)
                selection_metric = next(
                    (
                        metric
                        for metric in candidate_metrics
                        if metric.name in {"mse", "log_loss"} and metric.status == "available"
                    ),
                    None,
                )
                candidate_status.append(
                    {
                        "model": model_name,
                        "status": "completed",
                        "metrics": [metric.model_dump(mode="json") for metric in candidate_metrics],
                        "selection_metric": selection_metric.name if selection_metric else None,
                        "selection_value": selection_metric.value if selection_metric else None,
                        "selected_durations": selected_durations,
                        "stopping_diagnostics": stopping_diagnostics,
                        "stopping_policy": {
                            "enabled": config.models.early_stopping,
                            "fraction": config.models.early_stopping_fraction,
                            "patience": config.models.early_stopping_patience,
                        },
                        "model_bundles": candidate_dirs,
                        "tracking_mode": tracking_mode if get_platform_config().mlflow.enabled else "disabled",
                        "tracking_refs": tracking_refs,
                    }
                )
                candidates.append((model_name, tuple(candidate_predictions), candidate_metrics, candidate_dirs))
                candidate_prediction_sets[model_name] = tuple(candidate_predictions)
            except Exception as exc:
                candidate_status.append({"model": model_name, "status": "failed", "error": repr(exc)})
        if not candidates:
            raise RuntimeError("No enabled model candidate completed successfully")

        def candidate_score(candidate: tuple[str, tuple[PredictionFrame, ...], tuple[MetricResult, ...], list[str]]) -> float:
            metric = next(
                (item for item in candidate[2] if item.name in {"mse", "log_loss"} and item.status == "available"), None
            )
            return float(metric.value) if metric is not None and metric.value is not None else float("inf")

        selected_name, predictions, metrics, model_dirs = min(
            candidates, key=lambda candidate: (candidate_score(candidate), candidate[0])
        )
        candidate_status.append({"selected_model": selected_name})
        allocations, backtest = _strategy_result(prepared, predictions, config, period_name="development")
        challenger_returns: dict[str, dict[str, Any]] = {}
        for name, candidate_predictions, _, _ in candidates:
            if name == "baseline":
                continue
            try:
                candidate_backtest = (
                    backtest
                    if name == selected_name
                    else _strategy_result(prepared, candidate_predictions, config, period_name="development")[1]
                )
                series = _statistical_backtest_returns(
                    candidate_backtest, frequency=config.statistics.frequency, calendar=prepared.snapshot.calendar
                )
                if series is not None:
                    challenger_returns[name] = {
                        "returns": series.to_list(),
                        "timestamps": [timestamp.isoformat() for timestamp in series.index],
                    }
            except Exception as exc:
                challenger_returns[name] = {"error": repr(exc)}
        evidence_metrics = ()
        if backtest.net_returns is not None:
            statistical_net = _statistical_backtest_returns(
                backtest,
                frequency=config.statistics.frequency,
                calendar=prepared.snapshot.calendar,
            )
            evidence_metrics = tuple(
                sharpe_evidence_metrics(
                    statistical_net.to_numpy(),
                    frequency=config.statistics.frequency,
                    min_observations=config.statistics.min_observations,
                    trial_sharpes=[],
                ).values()
            )
            convention = return_convention(config.statistics.frequency)
            for metric in evidence_metrics:
                metric.inputs.update({"return_convention": convention, "calendar": prepared.snapshot.calendar})
                metric.conventions.update({"return_convention": convention, "calendar": prepared.snapshot.calendar})
            metrics = tuple(metrics) + evidence_metrics
        result = persist_run(
            store=store,
            run_id=run_id,
            config=config,
            dataset=prepared,
            predictions=tuple(predictions),
            metrics=metrics,
            model_dirs=model_dirs,
            allocations=allocations,
            backtest=backtest,
            bundles=tuple(bundles),
            candidate_status=candidate_status,
            candidate_predictions=candidate_prediction_sets,
            reservation=reservation,
        )
        ledger.update_attempt(
            attempt_id,
            status="completed",
            result={
                "run_id": run_id,
                "logical_trial_id": logical_trial,
                "return_convention": return_convention(config.statistics.frequency),
                "execution_evidence_version": "period-bounded-next-open.v2",
                "calendar": prepared.snapshot.calendar,
                "challenger_returns": challenger_returns,
                "development_net_returns": (
                    _statistical_backtest_returns(
                        backtest, frequency=config.statistics.frequency, calendar=prepared.snapshot.calendar
                    ).to_list()
                    if backtest.net_returns is not None
                    else None
                ),
                "development_gross_returns": (
                    _statistical_backtest_returns(
                        backtest, frequency=config.statistics.frequency, calendar=prepared.snapshot.calendar, gross=True
                    ).to_list()
                    if backtest.gross_returns is not None
                    else None
                ),
            },
        )
        return result
    except Exception as exc:
        store.mark_failed(
            run_id,
            repr(exc),
            {"logical_trial_id": logical_trial, "config_hash": config.resolved_config_hash, "candidates": candidate_status},
        )
        ledger.update_attempt(attempt_id, status="failed", error=repr(exc))
        raise
    finally:
        ledger.close()


def load_run(run_id: str, *, artifacts_dir: str | Path | None = None) -> RunResult:
    """Load a completed run and its verified local snapshot and strategy results."""
    store = ArtifactStore(artifacts_dir or get_platform_config().paths.artifacts_dir)
    manifest = store.load_manifest(run_id)
    for record in manifest.get("artifacts", []):
        store.verify_artifact(run_id, record)
    load_scoring_targets(run_id, artifacts_dir=artifacts_dir)
    records = json.loads((store.run_dir(run_id) / "predictions.json").read_text())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model_ref"], record["fold_ref"]), []).append(record)
    predictions = []
    for (model_ref, fold_ref), items in grouped.items():
        keys = tuple(SampleKey(ticker=item["ticker"], timestamp=item["timestamp"]) for item in items)
        values = np.asarray([item["prediction"] for item in items])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        predictions.append(PredictionFrame(keys=keys, model_ref=model_ref, fold_ref=fold_ref, predictions=values))
    metrics = tuple(
        MetricResult.model_validate(item) for item in json.loads((store.run_dir(run_id) / "metrics.json").read_text())
    )
    snapshot_metadata = json.loads((store.run_dir(run_id) / "snapshot/metadata.json").read_text())
    snapshot_metadata["bars"] = pl.read_parquet(store.run_dir(run_id) / "snapshot/bars.parquet")
    snapshot = MarketSnapshot.model_validate(snapshot_metadata)
    if snapshot.snapshot_hash != manifest["snapshot_hash"]:
        raise ValueError("Saved snapshot identity does not match the run manifest")
    allocations = tuple(
        AllocationFrame.model_validate(item) for item in json.loads((store.run_dir(run_id) / "allocations.json").read_text())
    )
    backtest = None
    if manifest.get("backtest_paths"):
        series = json.loads((store.run_dir(run_id) / "backtest/series.json").read_text())
        backtest = BacktestResult(
            orders=pl.read_parquet(store.run_dir(run_id) / "backtest/orders.parquet"),
            trade_events=pl.read_parquet(store.run_dir(run_id) / "backtest/trade_events.parquet"),
            equity=pl.read_parquet(store.run_dir(run_id) / "backtest/equity.parquet"),
            exposure=pl.read_parquet(store.run_dir(run_id) / "backtest/exposure.parquet"),
            gross_returns=pl.Series("gross_returns", series["gross_returns"]) if series["gross_returns"] is not None else None,
            net_returns=pl.Series("net_returns", series["net_returns"]) if series["net_returns"] is not None else None,
            turnover=pl.Series("turnover", series["turnover"]) if series["turnover"] is not None else None,
            period=series["period"],
            diagnostics=series["diagnostics"],
        )
    ref = RunRef(
        run_id=run_id,
        snapshot_hash=manifest["snapshot_hash"],
        config_hash=manifest["config_hash"],
        split_hash=manifest["split_hash"],
        model_ref=manifest.get("model_bundles", ["baseline"])[0]
        if manifest.get("model_bundles")
        else manifest.get("parent_model_ref") or "baseline",
        parent_run_id=manifest.get("parent_run_id"),
    )
    return RunResult(
        ref=ref,
        predictions=tuple(predictions),
        allocations=allocations,
        metrics=metrics,
        backtest=backtest,
        snapshot=snapshot,
        artifact_manifest=manifest,
    )


def _load_prediction_frames(store: ArtifactStore, run_id: str, relative_path: str) -> tuple[PredictionFrame, ...]:
    records = json.loads((store.run_dir(run_id) / relative_path).read_text())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model_ref"], record["fold_ref"]), []).append(record)
    frames = []
    for (model_ref, fold_ref), items in grouped.items():
        keys = tuple(SampleKey(ticker=item["ticker"], timestamp=item["timestamp"]) for item in items)
        values = np.asarray([item["prediction"] for item in items])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        frames.append(PredictionFrame(keys=keys, model_ref=model_ref, fold_ref=fold_ref, predictions=values))
    return tuple(frames)


def load_scoring_targets(run_id: str, *, artifacts_dir: str | Path | None = None) -> dict[str, ScoringTargets]:
    """Reload checksummed scoring labels by fold, preserving legacy runs without them."""
    store = ArtifactStore(artifacts_dir or get_platform_config().paths.artifacts_dir)
    manifest = store.load_manifest(run_id)
    records = {record["path"]: record for record in manifest.get("artifacts", [])}
    coverage = manifest.get("scored_coverage", {})
    targets_by_fold: dict[str, ScoringTargets] = {}
    for relative_path in coverage.get("artifacts", []):
        record = records.get(relative_path)
        if record is None:
            raise ValueError(f"Scoring target artifact is not checksummed in the run manifest: {relative_path}")
        path = store.verify_artifact(run_id, record)
        artifact = json.loads(path.read_text())
        fold_ref = artifact["fold_ref"]
        if fold_ref in targets_by_fold:
            raise ValueError(f"Duplicate scoring target artifact for fold {fold_ref}")
        rows = artifact["targets"]
        keys = tuple(SampleKey(ticker=row["ticker"], timestamp=row["timestamp"]) for row in rows)
        values = np.asarray([row["target"] for row in rows], dtype=np.float32)
        targets_by_fold[fold_ref] = ScoringTargets(keys=keys, values=values)
    return targets_by_fold


def load_candidate_predictions(
    run_id: str, model_name: str, *, artifacts_dir: str | Path | None = None
) -> tuple[PredictionFrame, ...]:
    """Load a completed candidate's saved development predictions without retraining."""
    store = ArtifactStore(artifacts_dir or get_platform_config().paths.artifacts_dir)
    manifest = store.load_manifest(run_id)
    for record in manifest.get("artifacts", []):
        store.verify_artifact(run_id, record)
    load_scoring_targets(run_id, artifacts_dir=artifacts_dir)
    candidate = next(
        (
            item
            for item in manifest.get("candidates", [])
            if item.get("model") == model_name and item.get("status") == "completed"
        ),
        None,
    )
    if candidate is None:
        raise ValueError(f"Candidate '{model_name}' is not a completed development candidate")
    relative_path = manifest.get("candidate_prediction_paths", {}).get(model_name)
    if relative_path is None:
        raise FileNotFoundError(f"Saved predictions are missing for candidate '{model_name}'")
    return _load_prediction_frames(store, run_id, relative_path)


def campaign_evidence(run_id: str, *, artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Compute comparable campaign evidence from the frozen local ledger snapshot."""
    result = load_run(run_id, artifacts_dir=artifacts_dir)
    ledger = TrialLedger(Path(artifacts_dir or get_platform_config().paths.artifacts_dir) / "trials.sqlite")
    try:
        frequency = result.artifact_manifest["config"]["statistics"]["frequency"]
        calendar = (
            result.snapshot.calendar if result.snapshot is not None else result.artifact_manifest["config"]["data"]["calendar"]
        )
        convention = return_convention(frequency)
        series = ledger.comparable_return_series(
            campaign=result.artifact_manifest["config"]["campaign"]["name"],
            snapshot_hash=result.ref.snapshot_hash,
            evaluation_hash=result.ref.split_hash,
            frequency=frequency,
            cost_treatment="fees+slippage",
            task=result.artifact_manifest.get("task"),
            return_convention=convention,
            execution_evidence_version=result.artifact_manifest.get("execution_evidence_version"),
        )
        trial_sharpes = []
        for item in series["complete"]:
            try:
                trial_sharpes.append(sharpe_statistics(item["returns"], frequency=frequency, min_observations=2)["sharpe"])
            except ValueError:
                continue
        current_returns = (
            _statistical_backtest_returns(result.backtest, frequency=frequency, calendar=calendar).to_numpy()
            if result.backtest is not None and result.backtest.net_returns is not None
            else []
        )
        metrics = sharpe_evidence_metrics(
            current_returns,
            trial_sharpes=trial_sharpes,
            frequency=frequency,
            min_observations=result.artifact_manifest["config"]["statistics"]["min_observations"],
        )
        for metric in metrics.values():
            metric.inputs.update({"return_convention": convention, "calendar": calendar})
            metric.conventions.update({"return_convention": convention, "calendar": calendar})
        report = CampaignEvidenceReport(
            source_run_id=run_id,
            campaign=result.artifact_manifest["config"]["campaign"]["name"],
            comparison_dimensions={
                "snapshot_hash": result.ref.snapshot_hash,
                "evaluation_hash": result.ref.split_hash,
                "task": result.artifact_manifest.get("task"),
                "frequency": frequency,
                "return_convention": convention,
                "calendar": calendar,
                "cost_treatment": "fees+slippage",
            },
            included_logical_trial_ids=tuple(item["logical_trial_id"] for item in series["complete"]),
            selected_attempt_ids=tuple(item["attempt_id"] for item in series["complete"]),
            exclusions=tuple(series["exclusions"]),
            population_counts={
                "logical_trials": series["population"],
                "selected_attempts": len(series["complete"]),
                "excluded_attempts": len(series["exclusions"]),
            },
            metric_values={name: metric.value for name, metric in metrics.items()},
            metric_details={
                name: {"status": metric.status, "reason": metric.reason, "conventions": metric.conventions}
                for name, metric in metrics.items()
            },
            calculated_at=datetime.now(timezone.utc),
        )
        sidecar = EvidenceStore(artifacts_dir or get_platform_config().paths.artifacts_dir).write(report.model_dump(mode="json"))
        return {**report.model_dump(mode="json"), "evidence_sidecar": sidecar, "metrics": metrics}
    finally:
        ledger.close()


def _holdout_decision_inputs(result: RunResult, model_name: str, ledger: TrialLedger) -> dict[str, Any]:
    manifest = result.artifact_manifest
    config = PipelineConfig.from_artifact_config(
        manifest["config"], schema_version=manifest.get("config_schema_version"), expected_hash=result.ref.config_hash
    )
    frequency = config.statistics.frequency
    convention = return_convention(frequency)
    evidence = ledger.candidate_search_evidence(
        snapshot_hash=result.ref.snapshot_hash,
        evaluation_hash=result.ref.split_hash,
        frequency=frequency,
        task=config.task.kind,
        return_convention=convention,
        min_observations=max(30, config.statistics.min_observations),
        calendar=result.snapshot.calendar if result.snapshot is not None else None,
        backtest_policy=config.backtest.model_dump(mode="json"),
        target_policy=config.task.model_dump(mode="json"),
    )
    trials = ledger.trials(snapshot_hash=result.ref.snapshot_hash)
    run_attempt = next(
        (
            attempt
            for trial in trials
            if trial["config_hash"] == result.ref.config_hash
            and trial["snapshot_hash"] == result.ref.snapshot_hash
            and trial["evaluation_hash"] == result.ref.split_hash
            and not trial["settings"].get("parent_prediction_run")
            for attempt in ledger.attempts(trial["logical_trial_id"])
            if attempt["status"] == "completed" and (attempt.get("result") or {}).get("run_id") == result.ref.run_id
        ),
        None,
    )
    current_returns = (run_attempt or {}).get("result", {}).get("development_net_returns")
    selected_candidate = next(
        (item for item in evidence["complete"] if item["run_id"] == result.ref.run_id and item["model"] == model_name),
        None,
    )
    saved_series = (
        _statistical_backtest_returns(result.backtest, frequency=frequency, calendar=result.snapshot.calendar)
        if result.backtest is not None and result.snapshot is not None and result.backtest.net_returns is not None
        else None
    )
    if (
        selected_candidate is None
        or selected_candidate["returns"] != current_returns
        or saved_series is None
        or selected_candidate["returns"] != saved_series.to_list()
        or selected_candidate["timestamps"] != [timestamp.isoformat() for timestamp in saved_series.index]
    ):
        current_returns = None
    gate = evaluate_gate(
        candidates=manifest.get("candidates", []),
        selected_model=model_name,
        task=config.task.kind,
        returns=current_returns,
        comparable=evidence["complete"],
        exclusions=evidence["exclusions"],
        trial_count=evidence["K"],
        frequency=frequency,
        min_observations=config.statistics.min_observations,
    )
    gate["evidence"].update(
        {
            "source_run_id": result.ref.run_id,
            "snapshot_hash": result.ref.snapshot_hash,
            "split_hash": result.ref.split_hash,
            "holdout_start": manifest.get("snapshot_reservation", {}).get("holdout_start"),
            "holdout_end": manifest.get("snapshot_reservation", {}).get("holdout_end"),
            "candidate_keys": evidence["candidate_keys"],
            "requested_attempt_id": (run_attempt or {}).get("attempt_id"),
        }
    )
    audit_inputs = {
        "policy_version": gate["evidence"]["policy_version"],
        "source_run_id": result.ref.run_id,
        "source_config_hash": result.ref.config_hash,
        "snapshot_hash": result.ref.snapshot_hash,
        "split_hash": result.ref.split_hash,
        "selected_model": model_name,
        "trial_count": evidence["K"],
        "trials": [
            {
                "logical_trial_id": item["logical_trial_id"],
                "settings": item["settings"],
                "attempts": [
                    {"attempt_id": attempt["attempt_id"], "status": attempt["status"], "result": attempt["result"]}
                    for attempt in ledger.attempts(item["logical_trial_id"])
                ],
            }
            for item in trials
        ],
        "comparable": evidence["complete"],
        "exclusions": evidence["exclusions"],
    }

    def finite_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return f"nonfinite:{value}"
        return value

    gate["evidence"]["evidence_digest"] = hashlib.sha256(
        json.dumps(finite_json(audit_inputs), sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()
    ).hexdigest()
    return gate


def preview_holdout(run_id: str, *, model_name: str | None = None) -> dict[str, Any]:
    """Read-only advisory evaluation of the current holdout gate inputs."""
    result = load_run(run_id)
    if result.ref.parent_run_id is not None or result.snapshot is None:
        raise ValueError("Holdout preview requires a completed development run")
    selected = model_name or next(
        (item.get("selected_model") for item in result.artifact_manifest.get("candidates", []) if item.get("selected_model")),
        None,
    )
    if selected is None:
        raise ValueError("Saved development winner is missing")
    ledger_path = get_platform_config().paths.artifacts_dir / "trials.sqlite"
    if not ledger_path.is_file():
        raise ValueError("Holdout ledger is not initialized for this store")
    ledger = TrialLedger.read_only(ledger_path)
    try:
        reservation = ledger.snapshot_reservation(result.ref.snapshot_hash)
        saved_reservation = result.artifact_manifest.get("snapshot_reservation") or {}
        if (
            reservation is None
            or reservation["eligibility"] != "eligible"
            or reservation["holdout_start"] != saved_reservation.get("holdout_start")
            or reservation["holdout_end"] != saved_reservation.get("holdout_end")
        ):
            gate = {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": {"reason": "LEGACY_BOUNDARY_UNVERIFIED"}}
        else:
            winner = next(
                (
                    item.get("selected_model")
                    for item in result.artifact_manifest.get("candidates", [])
                    if item.get("selected_model")
                ),
                None,
            )
            if model_name is not None and model_name != winner:
                gate = {"verdict": "REJECTED_MODEL_MISMATCH", "evidence": {"reason": "MODEL_IS_NOT_SAVED_WINNER"}}
            else:
                gate = _holdout_decision_inputs(result, selected, ledger)
        if ledger.seal(result.ref.snapshot_hash):
            gate = {"verdict": "REJECTED_HOLDOUT_SEALED", "evidence": {**gate["evidence"], "reason": "HOLDOUT_ALREADY_SEALED"}}
        return {
            "verdict": gate["verdict"],
            "evidence": gate["evidence"],
            "decision_id": None,
            "child_run_id": None,
            "advisory": True,
        }
    finally:
        ledger.close()


def index_deferred_run(run_id: str, *, artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Index deferred campaign fold bundles serially without changing the run manifest."""
    store = ArtifactStore(artifacts_dir or get_platform_config().paths.artifacts_dir)
    run_dir = store.run_dir(run_id)
    sidecar_path = run_dir / "tracking" / "index.json"
    if sidecar_path.exists():
        return json.loads(sidecar_path.read_text())
    result = load_run(run_id, artifacts_dir=artifacts_dir)
    platform = get_platform_config()
    if not platform.mlflow.enabled:
        return {"status": "disabled", "run_id": run_id, "references": []}
    references: list[dict[str, str]] = []
    errors: list[str] = []
    config = PipelineConfig.from_artifact_config(
        result.artifact_manifest["config"],
        schema_version=result.artifact_manifest.get("config_schema_version"),
        expected_hash=result.artifact_manifest.get("config_hash"),
    )
    for candidate in result.artifact_manifest.get("candidates", []):
        model_name = candidate.get("model")
        if candidate.get("status") != "completed" or model_name not in {"baseline", "xgboost", "gru", "lstm"}:
            continue
        model_bundles = candidate.get("model_bundles", [])
        metrics = {
            metric["name"]: float(metric["value"])
            for metric in candidate.get("metrics", [])
            if metric.get("status") == "available" and metric.get("value") is not None
        }
        for fold_index, model_ref in enumerate(model_bundles):
            model_dir = run_dir / model_ref
            try:
                model = create_model(model_name, task=config.task.kind)
                model = type(model).load(model_dir)
                references.append(
                    log_local_model(
                        model,
                        artifact_path=f"{run_id}-{model_name}-fold-{fold_index}",
                        model_dir=model_dir,
                        tracking_uri=platform.paths.mlflow_tracking_uri,
                        experiment_name=platform.mlflow.experiment_name,
                        params={
                            "run_id": run_id,
                            "config_hash": result.ref.config_hash,
                            "snapshot_hash": result.ref.snapshot_hash,
                            "split_hash": result.ref.split_hash,
                            "task": config.task.kind,
                            "model": model_name,
                            "seed": config.models.seed,
                            "fold": fold_index,
                            "model_bundle": model_ref,
                        },
                        metrics=metrics,
                        idempotency_key=f"{run_id}:{model_name}:{fold_index}",
                    )
                )
            except Exception as exc:
                errors.append(f"{model_name} fold {fold_index}: {exc!r}")
    sidecar = {
        "schema_version": "tracking-index.v1",
        "run_id": run_id,
        "status": "indexed" if not errors else "partial_failure",
        "references": references,
        "errors": errors,
    }
    if not errors:
        store.write_json(run_id, "tracking/index.json", sidecar)
    return sidecar


def evaluate_holdout(
    selected_run: RunResult | str,
    *,
    config: PipelineConfig | None = None,
    model_name: str | None = None,
    allocation_overrides: dict[str, Any] | None = None,
    backtest_overrides: dict[str, Any] | None = None,
) -> RunResult:
    """Evaluate an explicitly selected development run against its saved snapshot."""
    run_id = selected_run if isinstance(selected_run, str) else selected_run.ref.run_id
    result = load_run(run_id)
    if result.ref.parent_run_id is not None or not result.artifact_manifest.get("candidates"):
        raise ValueError("Holdout requires a completed, explicitly selected development run")
    original = PipelineConfig.from_artifact_config(
        result.artifact_manifest["config"],
        schema_version=result.artifact_manifest.get("config_schema_version"),
        expected_hash=result.ref.config_hash,
    )
    config = config or original
    if allocation_overrides is not None or backtest_overrides is not None:
        raise ValueError("Holdout allocation and backtest overrides are not supported")
    if config.resolved_config_hash != result.ref.config_hash:
        raise ValueError("Holdout configuration differs from the selected development run")
    if result.snapshot is None:
        raise ValueError("Selected run has no saved snapshot")
    winner = next(
        (item.get("selected_model") for item in result.artifact_manifest.get("candidates", []) if item.get("selected_model")),
        None,
    )
    if winner is None:
        raise ValueError("Selected development run has no saved tournament winner")
    store = ArtifactStore(get_platform_config().paths.artifacts_dir)
    ledger_path = get_platform_config().paths.artifacts_dir / "trials.sqlite"
    if not ledger_path.is_file():
        raise ValueError("Holdout ledger is not initialized for this store")
    ledger = TrialLedger(ledger_path)
    reservation = ledger.snapshot_reservation(result.ref.snapshot_hash)
    if reservation is None or reservation["eligibility"] != "eligible":
        decision = ledger.record_decision(
            decision_id=f"decision-{uuid.uuid4().hex}",
            snapshot_hash=result.ref.snapshot_hash,
            source_run_id=run_id,
            model_name=model_name or winner,
            verdict="REJECTED_INSUFFICIENT_EVIDENCE",
            evidence={"reason": "LEGACY_BOUNDARY_UNVERIFIED", "source_run_id": run_id, "snapshot_hash": result.ref.snapshot_hash},
        )
        ledger.close()
        raise HoldoutRejected(decision)
    if reservation["holdout_start"] != result.artifact_manifest.get("snapshot_reservation", {}).get(
        "holdout_start"
    ) or reservation["holdout_end"] != result.artifact_manifest.get("snapshot_reservation", {}).get("holdout_end"):
        decision = ledger.record_decision(
            decision_id=f"decision-{uuid.uuid4().hex}",
            snapshot_hash=result.ref.snapshot_hash,
            source_run_id=run_id,
            model_name=model_name or winner,
            verdict="REJECTED_INSUFFICIENT_EVIDENCE",
            evidence={"reason": "RESERVATION_MISMATCH", "source_run_id": run_id, "snapshot_hash": result.ref.snapshot_hash},
        )
        ledger.close()
        raise HoldoutRejected(decision)
    existing_seal = ledger.seal(result.ref.snapshot_hash)
    resume_seal = None
    if existing_seal:
        prior_decision = ledger.decision(existing_seal["decision_id"]) or {}
        if (
            existing_seal["execution_status"] == "reserved"
            and prior_decision.get("source_run_id") == run_id
            and prior_decision.get("model_name") == winner
            and (model_name is None or model_name == winner)
        ):
            resume_seal = existing_seal
        else:
            if existing_seal["execution_status"] in {"completed", "running"}:
                try:
                    sealed_result = load_run(existing_seal["child_run_id"])
                    if (
                        sealed_result.ref.parent_run_id == run_id
                        and sealed_result.artifact_manifest.get("decision_id") == existing_seal["decision_id"]
                        and sealed_result.ref.snapshot_hash == result.ref.snapshot_hash
                        and sealed_result.ref.config_hash == result.ref.config_hash
                        and sealed_result.ref.split_hash == result.ref.split_hash
                        and (model_name is None or model_name == winner)
                    ):
                        if existing_seal["execution_status"] == "running":
                            for trial in ledger.trials(snapshot_hash=result.ref.snapshot_hash):
                                if (
                                    trial["settings"].get("evaluation") != "holdout"
                                    or trial["settings"].get("selected_run") != run_id
                                ):
                                    continue
                                for prior_attempt in ledger.attempts(trial["logical_trial_id"]):
                                    if prior_attempt["status"] == "running":
                                        ledger.update_attempt(
                                            prior_attempt["attempt_id"],
                                            status="completed",
                                            result={
                                                "run_id": sealed_result.ref.run_id,
                                                "logical_trial_id": trial["logical_trial_id"],
                                            },
                                        )
                            ledger.update_seal(result.ref.snapshot_hash, status="completed", owner_id=existing_seal["owner_id"])
                        ledger.close()
                        return sealed_result
                except (FileNotFoundError, ValueError):
                    pass
            ledger.close()
            raise HoldoutRejected(
                {
                    "verdict": "REJECTED_HOLDOUT_SEALED",
                    "evidence": {
                        **prior_decision.get("evidence", {}),
                        "reason": f"HOLDOUT_{existing_seal['execution_status'].upper()}",
                    },
                    "decision_id": existing_seal["decision_id"],
                    "execution_status": existing_seal["execution_status"],
                }
            )
    if model_name is not None and model_name != winner:
        decision = ledger.record_decision(
            decision_id=f"decision-{uuid.uuid4().hex}",
            snapshot_hash=result.ref.snapshot_hash,
            source_run_id=run_id,
            model_name=model_name,
            verdict="REJECTED_MODEL_MISMATCH",
            evidence={"reason": "MODEL_IS_NOT_SAVED_WINNER", "selected_model": model_name, "saved_winner": winner},
        )
        ledger.close()
        raise HoldoutRejected(decision)
    dataset = prepare_dataset(config, snapshot=result.snapshot)
    if dataset.snapshot.snapshot_hash != result.ref.snapshot_hash or _split_hash(dataset) != result.ref.split_hash:
        ledger.close()
        raise ValueError("Saved snapshot or evaluation schedule changed")
    selected_candidate = next(
        (
            item
            for item in result.artifact_manifest.get("candidates", [])
            if item.get("model") == winner and item.get("status") == "completed"
        ),
        None,
    )
    if selected_candidate is None:
        ledger.close()
        raise ValueError(f"Candidate '{winner}' is not a completed development candidate")
    selection_suffix = hashlib.sha256(
        json.dumps([winner, result.ref.config_hash, result.ref.split_hash], sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    child_id = f"{run_id}-holdout-{selection_suffix}"
    decision_id = resume_seal["decision_id"] if resume_seal else f"decision-{uuid.uuid4().hex}"
    if resume_seal and resume_seal["child_run_id"] != child_id:
        ledger.close()
        raise ValueError("Reserved holdout child does not match the verified source identity")
    recorded = (
        ledger.decision(decision_id)
        if resume_seal
        else ledger.record_decision(
            decision_id=decision_id,
            snapshot_hash=result.ref.snapshot_hash,
            source_run_id=run_id,
            model_name=winner,
            verdict="REJECTED_INSUFFICIENT_EVIDENCE",
            evidence={},
            child_run_id=child_id,
            evidence_factory=lambda: _holdout_decision_inputs(result, winner, ledger),
        )
    )
    if recorded.get("verdict") != "PASSED":
        ledger.close()
        raise HoldoutRejected(recorded)
    execution_owner = uuid.uuid4().hex
    ledger.update_seal(result.ref.snapshot_hash, status="running", owner_id=execution_owner)
    if (store.run_dir(child_id) / "manifest.json").exists():
        ledger.update_seal(result.ref.snapshot_hash, status="interrupted", owner_id=execution_owner)
        ledger.close()
        raise HoldoutRejected({"verdict": "REJECTED_HOLDOUT_SEALED", "evidence": {"reason": "UNVERIFIED_CHILD_ARTIFACT"}})
    existing_holdout_variants = list((store.root / "runs").glob(f"{run_id}-holdout-*"))
    exploratory_reuse = bool(existing_holdout_variants)
    ledger.close()
    selected = winner
    model_name = selected
    durations = [int(value) for value in selected_candidate.get("selected_durations", []) if int(value) > 0]
    holdout_duration = int(np.floor(np.median(durations) + 0.5)) if durations else None
    ledger = TrialLedger(ledger_path)
    attempt = None
    holdout_spec = FoldSpec(
        fold_id=-1,
        train_start=dataset.split_plan.development_start,
        train_end=dataset.split_plan.holdout_start,
        validation_start=dataset.split_plan.holdout_start,
        validation_end=dataset.split_plan.holdout_end,
    )
    try:
        logical, attempt, _ = ledger.register_attempt(
            campaign=config.campaign.name,
            config_hash=config.resolved_config_hash,
            snapshot_hash=result.ref.snapshot_hash,
            evaluation_hash=result.ref.split_hash,
            settings={
                "evaluation": "holdout",
                "selected_run": run_id,
                "model": selected,
                "allocation": config.allocation.model_dump(mode="json"),
                "backtest": config.backtest.model_dump(mode="json"),
            },
        )
        store.create_run(child_id)
        bundle = prepare_fold(dataset, holdout_spec)
        model = train_fold(bundle, config, model_name=selected, duration=holdout_duration)
        model_dir = store.run_dir(child_id) / "models" / f"{model_name}-holdout"
        model.save(model_dir)
        model_ref = str(model_dir.relative_to(store.run_dir(child_id)))
        prediction = predict_fold(model, bundle, model_ref=model_ref, fold_ref="holdout", task=config.task.kind)
        metrics = evaluate_predictions((prediction,), (bundle,), task=config.task.kind)
        allocations, backtest = _strategy_result(dataset, (prediction,), config, period_name="holdout")
        holdout_result = persist_run(
            store=store,
            run_id=child_id,
            config=config,
            dataset=dataset,
            predictions=(prediction,),
            metrics=metrics,
            model_dirs=[model_ref],
            allocations=allocations,
            backtest=backtest,
            bundles=(bundle,),
            candidate_status=[
                {
                    "selected_model": selected,
                    "evaluation": "holdout",
                    "selection_source": "explicit" if model_name else "development_default",
                    "exploratory_holdout_reuse": exploratory_reuse,
                    "allocation": config.allocation.model_dump(mode="json"),
                    "backtest": config.backtest.model_dump(mode="json"),
                }
            ],
            parent_run_id=run_id,
            reservation=reservation,
            decision_id=decision_id,
        )
        ledger.update_attempt(attempt, status="completed", result={"run_id": child_id, "logical_trial_id": logical})
        ledger.update_seal(result.ref.snapshot_hash, status="completed", owner_id=execution_owner)
        return holdout_result
    except Exception as exc:
        if store.run_dir(child_id).exists():
            store.mark_failed(child_id, repr(exc), {"parent_run_id": run_id})
        if attempt is not None:
            ledger.update_attempt(attempt, status="failed", error=repr(exc))
        ledger.update_seal(result.ref.snapshot_hash, status="failed", owner_id=execution_owner)
        raise
    finally:
        ledger.close()


def evaluate_saved_predictions(
    run_id: str,
    allocation_overrides: dict[str, Any] | None = None,
    backtest_overrides: dict[str, Any] | None = None,
    *,
    model_name: str | None = None,
) -> RunResult:
    """Evaluate an immutable development strategy variant from saved predictions."""
    parent = load_run(run_id)
    if parent.ref.parent_run_id is not None or parent.snapshot is None:
        raise ValueError("Strategy comparison requires a saved development run")
    saved_config = PipelineConfig.from_artifact_config(
        parent.artifact_manifest["config"],
        schema_version=parent.artifact_manifest.get("config_schema_version"),
        expected_hash=parent.ref.config_hash,
    )
    values = saved_config.model_dump(mode="python")
    values["allocation"] = {**values["allocation"], **(allocation_overrides or {})}
    values["backtest"] = {**values["backtest"], **(backtest_overrides or {})}
    config = PipelineConfig(**values)
    store = ArtifactStore(get_platform_config().paths.artifacts_dir)
    selected_name = model_name or next(
        (item.get("selected_model") for item in parent.artifact_manifest.get("candidates", []) if item.get("selected_model")),
        None,
    )
    selected_predictions = parent.predictions if model_name is None else load_candidate_predictions(run_id, model_name)
    candidate_record = next(
        (
            item
            for item in parent.artifact_manifest.get("candidates", [])
            if item.get("model") == selected_name and item.get("status") == "completed"
        ),
        None,
    )
    selected_metrics = (
        tuple(MetricResult.model_validate(item) for item in candidate_record.get("metrics", []))
        if candidate_record and candidate_record.get("metrics")
        else parent.metrics
    )
    suffix = hashlib.sha256(
        json.dumps([model_name, allocation_overrides, backtest_overrides], sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    child_id = f"{run_id}-child-{suffix}"
    if (store.run_dir(child_id) / "manifest.json").exists():
        ledger_path = get_platform_config().paths.artifacts_dir / "trials.sqlite"
        existing = TrialLedger.read_only(ledger_path)
        try:
            matches = [
                attempt
                for trial in existing.trials(snapshot_hash=parent.ref.snapshot_hash)
                if trial["settings"].get("parent_prediction_run") == run_id
                for attempt in existing.attempts(trial["logical_trial_id"])
                if attempt["status"] == "completed" and (attempt["result"] or {}).get("run_id") == child_id
            ]
        finally:
            existing.close()
        if not matches:
            raise ValueError("Saved prediction variant has no matching completed ledger attempt")
        return load_run(child_id)
    ledger = TrialLedger(get_platform_config().paths.artifacts_dir / "trials.sqlite")
    reservation = parent.artifact_manifest.get("snapshot_reservation")
    if not isinstance(reservation, dict) or not reservation.get("holdout_start") or not reservation.get("holdout_end"):
        ledger.close()
        raise ValueError("Saved development source has no verified snapshot reservation")
    try:
        ledger.reserve_snapshot(
            snapshot_hash=parent.ref.snapshot_hash,
            holdout_start=reservation["holdout_start"],
            holdout_end=reservation["holdout_end"],
        )
        dataset = prepare_dataset(config, snapshot=parent.snapshot)
        if dataset.snapshot.snapshot_hash != parent.ref.snapshot_hash or _split_hash(dataset) != parent.ref.split_hash:
            raise ValueError("Saved evaluation schedule or snapshot changed")
    except Exception:
        ledger.close()
        raise
    parent_models = saved_config.models.model_dump(mode="json")
    logical, attempt, _ = ledger.register_attempt(
        campaign=config.campaign.name,
        config_hash=config.resolved_config_hash,
        snapshot_hash=parent.ref.snapshot_hash,
        evaluation_hash=parent.ref.split_hash,
        settings={
            "evaluation": "development",
            "parent_prediction_run": run_id,
            "model": selected_name,
            "model_spec": parent_models.get(selected_name),
            "model_global_settings": {
                key: value for key, value in parent_models.items() if key not in {"baseline", "xgboost", "gru", "lstm"}
            },
            "task": config.task.kind,
            "frequency": config.statistics.frequency,
            "return_convention": return_convention(config.statistics.frequency),
            "execution_evidence_version": "period-bounded-next-open.v2",
            "calendar": parent.snapshot.calendar,
            "cost_treatment": "fees+slippage",
            "candidate_context": {
                key: value for key, value in config.model_dump(mode="json").items() if key not in {"campaign", "data", "models"}
            },
            "allocation": config.allocation.model_dump(mode="json"),
            "backtest": config.backtest.model_dump(mode="json"),
        },
    )
    store.create_run(child_id)
    try:
        allocations, backtest = _strategy_result(dataset, selected_predictions, config, period_name="development")
        result = persist_run(
            store=store,
            run_id=child_id,
            config=config,
            dataset=dataset,
            predictions=selected_predictions,
            metrics=selected_metrics,
            model_dirs=[],
            allocations=allocations,
            backtest=backtest,
            parent_run_id=run_id,
            parent_model_ref=selected_name or parent.ref.model_ref,
            inherited_scoring_coverage=parent.artifact_manifest.get("scored_coverage"),
            candidate_status=[{"model": model_name or parent.ref.model_ref, "selection_source": "saved_development_predictions"}],
        )
        series = _statistical_backtest_returns(backtest, frequency=config.statistics.frequency, calendar=parent.snapshot.calendar)
        ledger.update_attempt(
            attempt,
            status="completed",
            result={
                "run_id": child_id,
                "logical_trial_id": logical,
                "challenger_returns": {
                    selected_name: {
                        "returns": series.to_list(),
                        "timestamps": [timestamp.isoformat() for timestamp in series.index],
                    }
                }
                if selected_name != "baseline" and series is not None
                else {},
            },
        )
        return result
    except Exception as exc:
        store.mark_failed(child_id, repr(exc), {"parent_run_id": run_id})
        ledger.update_attempt(attempt, status="failed", error=repr(exc))
        raise
    finally:
        ledger.close()
