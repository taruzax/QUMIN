"""Offline Phase 1 null-control campaign and holdout-gate demonstration."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import yaml

from lab.core.config import CampaignMatrixConfig, PipelineConfig, Timeframe, get_platform_config
from lab.platform.artifacts import sha256_file
from lab.platform.campaign_store import CampaignStore
from lab.platform.data_access import build_market_snapshot
from lab.platform.market_store import SnapshotCatalog
from lab.platform.report_store import ReportStore
from lab.research.campaign import execute_campaign, plan_campaign
from lab.research.experiment import campaign_evidence, evaluate_holdout, load_run, preview_holdout
from lab.research.holdout_gate import HoldoutRejected
from lab.research.reporting import generate_run_report
from lab.research.trial_ledger import TrialLedger


def _demo_config(snapshot_id: str) -> PipelineConfig:
    base = PipelineConfig.from_yaml("config/demo.yaml")
    values = base.model_dump(mode="python")
    values["data"].update({"source": "snapshot", "snapshot_id": snapshot_id, "timeframe": Timeframe.D1})
    values["features"].update({"return_lags": [1], "target_horizons": [1]})
    values["tensor"]["sequence_len"] = 10
    values["task"]["kind"] = "regression"
    values["splits"].update({"n_folds": 3, "holdout_fraction": 0.20, "development_train_fraction": 0.5})
    values["models"].update({"device": "cpu", "seed": 42, "early_stopping": False})
    values["models"]["xgboost"].update({"enabled": True, "params": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.05}})
    values["models"]["gru"]["enabled"] = False
    values["models"]["lstm"]["enabled"] = False
    values["allocation"].update({"lookback_bars": 40, "rebalance_every_bars": 5})
    values["statistics"].update({"frequency": "daily", "min_observations": 30})
    values["campaign"]["name"] = "phase1-null-control"
    return PipelineConfig(**values)


def _highest_scored_candidate(previews: list[tuple[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]] | None:
    """Return the highest finite eligible score, breaking ties by ascending run ID."""
    scored = []
    for run_id, preview in previews:
        if preview.get("verdict") not in {"PASSED", "REJECTED_BY_DSR"}:
            continue
        score = preview.get("evidence", {}).get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
            scored.append((float(score), run_id, preview))
    if not scored:
        return None
    score, run_id, preview = min(scored, key=lambda item: (-item[0], item[1]))
    return run_id, preview


def _require_demo_trial_count(decision: dict[str, Any]) -> int:
    """Require the gate's canonical candidate count to match the 20-variation demo."""
    trial_count = decision.get("evidence", {}).get("trial_count")
    if type(trial_count) is not int or trial_count != 20:
        raise RuntimeError(f"Expected holdout gate K = 20; got {trial_count!r}")
    return trial_count


def run_phase1_demo(*, store_root: str | Path = "artifacts/phase1-demo") -> dict[str, Any]:
    """Publish the null fixture, run 20 CPU variations, gate once, and report."""
    root = Path(store_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_root = root / "snapshots"
    platform_path = root / "platform.yaml"
    platform_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(root),
                    "snapshot_dir": str(snapshot_root),
                    "mlflow_tracking_uri": f"sqlite:///{root / 'mlflow.db'}",
                },
                "mlflow": {"enabled": False, "experiment_name": "phase1-null-control"},
            },
            sort_keys=False,
        )
    )
    old_platform = os.environ.get("QUANT_ML_PLATFORM_CONFIG")
    thread_env = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ["QUANT_ML_PLATFORM_CONFIG"] = str(platform_path)
    for name in thread_env:
        os.environ[name] = "1"
    get_platform_config.cache_clear()
    started = time.perf_counter()
    try:
        fixture_path = Path("tests/fixtures/phase1-null-control.csv")
        metadata_path = Path("tests/fixtures/phase1-null-control.json")
        metadata = json.loads(metadata_path.read_text())
        if sha256_file(fixture_path) != metadata["fixture_sha256"]:
            raise ValueError("Committed null-control fixture checksum mismatch")
        snapshot = build_market_snapshot(
            [fixture_path],
            calendar="XNYS",
            timeframe=Timeframe.D1,
            provenance={"fixture": "phase1-null-control-v1", "seed": metadata["seed"], "sha256": metadata["fixture_sha256"]},
            source_manifest=[{"path": str(fixture_path), "sha256": metadata["fixture_sha256"]}],
            metadata=metadata,
        )
        catalog = SnapshotCatalog(snapshot_root)
        entry = catalog.publish(snapshot, stream_id="phase1-null-control", coverage_status="accepted")
        catalog.connection.close()
        base_config = _demo_config(entry.snapshot_id)
        matrix = CampaignMatrixConfig(
            name="phase1-null-control",
            snapshot_ids=(entry.snapshot_id,),
            feature_column_sets=(("open", "high", "low", "close", "volume"),),
            seeds=tuple(range(20261108, 20261128)),
            max_planned_experiments=20,
            worker_count=4,
        )
        plan = plan_campaign(base_config, matrix)
        campaign_started = time.perf_counter()
        results = execute_campaign(base_config, plan)
        campaign_elapsed = time.perf_counter() - campaign_started
        campaign_store = CampaignStore(root)
        try:
            completed = [
                {"variation_id": item.variation_id, "status": item.status, "run_id": item.run_id}
                for item in campaign_store.entries(plan.campaign_id)
                if item.status == "completed" and item.run_id
            ]
        finally:
            campaign_store.close()
        if len(completed) != 20:
            raise RuntimeError(
                f"Expected 20 completed null-control experiments; got {len(completed)} "
                f"from {len(results)} new or reconciled results"
            )
        for item in completed:
            if load_run(item["run_id"]).ref.snapshot_hash != entry.snapshot_hash:
                raise RuntimeError("Campaign result uses a different snapshot from the published control")
        gate_started = time.perf_counter()
        previews = []
        for item in completed:
            preview = preview_holdout(item["run_id"])
            previews.append((item["run_id"], preview))
        selected_candidate = _highest_scored_candidate(previews)
        if selected_candidate is not None:
            selected_run_id, selected_preview = selected_candidate
        else:
            baseline_winners = []
            for item in completed:
                result = load_run(item["run_id"])
                selected = next(
                    (
                        row.get("selected_model")
                        for row in result.artifact_manifest.get("candidates", [])
                        if row.get("selected_model")
                    ),
                    None,
                )
                if selected == "baseline":
                    baseline_winners.append(item["run_id"])
            if not baseline_winners:
                raise RuntimeError("Null-control campaign has no gate-eligible challenger or baseline-winning run")
            selected_run_id = sorted(baseline_winners)[0]
            selected_preview = None
            for preview_run_id, preview in previews:
                if preview_run_id == selected_run_id:
                    selected_preview = preview
                    break
            if selected_preview is None:
                raise RuntimeError(f"Null-control preview is missing for selected baseline run: {selected_run_id}")
        try:
            holdout = evaluate_holdout(selected_run_id)
            raise RuntimeError(f"Null-control holdout unexpectedly passed and created child {holdout.ref.run_id}")
        except HoldoutRejected as rejection:
            decision = rejection.decision
        trial_count = _require_demo_trial_count(decision)
        if decision.get("verdict") not in {"REJECTED_BY_BASELINE", "REJECTED_BY_DSR"}:
            raise RuntimeError(f"Null-control gate rejected for an unexpected reason: {decision}")
        gate_elapsed = time.perf_counter() - gate_started
        selected_result = load_run(selected_run_id)
        config = PipelineConfig.from_artifact_config(
            selected_result.artifact_manifest["config"],
            schema_version=selected_result.artifact_manifest.get("config_schema_version"),
            expected_hash=selected_result.ref.config_hash,
        )
        report_started = time.perf_counter()
        evidence = campaign_evidence(selected_run_id)
        report_root = root / "reports"
        report_root.mkdir(parents=True, exist_ok=True)
        report_path = report_root / f"{decision['decision_id']}.html"
        report = generate_run_report(
            report_path,
            result=selected_result,
            config=config,
            evidence=evidence,
            report_type="development",
            gate_decision=decision,
        )
        locator = ReportStore(root).write_locator(
            report_path=report,
            run_id=selected_run_id,
            manifest_path=root / "runs" / selected_run_id / "manifest.json",
            evidence=evidence.get("evidence_sidecar"),
            report_type="development",
            decision_id=decision["decision_id"],
        )
        report_elapsed = time.perf_counter() - report_started
        ledger = TrialLedger.read_only(root / "trials.sqlite")
        try:
            seal = ledger.seal(entry.snapshot_hash)
        finally:
            ledger.close()
        if seal is not None:
            raise RuntimeError("Null-control rejection must not create a holdout seal")
        return {
            "snapshot_id": entry.snapshot_id,
            "snapshot_hash": entry.snapshot_hash,
            "campaign_id": plan.campaign_id,
            "completed_variations": len(completed),
            "K": trial_count,
            "selected_run_id": selected_run_id,
            "candidate_selection": {
                "verdict": selected_preview.get("verdict") if selected_preview else None,
                "score": selected_preview.get("evidence", {}).get("score") if selected_preview else None,
                "reason": selected_preview.get("evidence", {}).get("reason") if selected_preview else None,
            },
            "decision": decision,
            "report_path": str(report.resolve()),
            "locator_path": locator["locator_path"],
            "timings_seconds": {
                "startup_and_fixture": campaign_started - started,
                "campaign": campaign_elapsed,
                "gate": gate_elapsed,
                "report": report_elapsed,
                "total": time.perf_counter() - started,
            },
        }
    finally:
        if old_platform is None:
            os.environ.pop("QUANT_ML_PLATFORM_CONFIG", None)
        else:
            os.environ["QUANT_ML_PLATFORM_CONFIG"] = old_platform
        for name, value in thread_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_platform_config.cache_clear()
