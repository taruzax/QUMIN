"""End-to-end positive control for a learnable next-open signal."""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from lab.core.config import Timeframe, get_platform_config
from lab.demo_phase1 import _demo_config
from lab.platform.data_access import build_market_snapshot
from lab.platform.market_store import SnapshotCatalog
from lab.research.experiment import evaluate_holdout, preview_holdout, run_experiment
from lab.research.holdout_gate import HoldoutRejected
from lab.research.preprocessing import prepare_dataset
from lab.research.trial_ledger import TrialLedger


def test_positive_control_opens_once_and_replays_verified_child(tmp_path, monkeypatch):
    platform = tmp_path / "platform.yaml"
    platform.write_text(
        yaml.safe_dump(
            {
                "paths": {"artifacts_dir": str(tmp_path), "snapshot_dir": str(tmp_path / "snapshots")},
                "mlflow": {"enabled": False},
            }
        )
    )
    monkeypatch.setenv("QUANT_ML_PLATFORM_CONFIG", str(platform))
    get_platform_config.cache_clear()
    try:
        fixture = Path("tests/fixtures/phase1-positive-control.csv")
        metadata = json.loads(Path("tests/fixtures/phase1-positive-control.json").read_text())
        snapshot = build_market_snapshot(
            [fixture],
            calendar="XNYS",
            timeframe=Timeframe.D1,
            provenance={"fixture": "phase1-positive-control-v1"},
            source_manifest=[{"path": str(fixture)}],
            metadata=metadata,
        )
        catalog = SnapshotCatalog(tmp_path / "snapshots")
        entry = catalog.publish(snapshot, stream_id="positive-control", coverage_status="accepted")
        catalog.connection.close()
        config = _demo_config(entry.snapshot_id)
        source = run_experiment(config)
        repeat = run_experiment(config)
        assert len(source.predictions) == len(repeat.predictions)
        for original, reproduced in zip(source.predictions, repeat.predictions, strict=True):
            assert original.keys == reproduced.keys
            np.testing.assert_allclose(original.predictions, reproduced.predictions, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(source.backtest.net_returns, repeat.backtest.net_returns, rtol=1e-7, atol=1e-9)
        cutoff = prepare_dataset(config, snapshot=snapshot).split_plan.holdout_start
        future = pl.col("timestamp") >= cutoff
        changed_bars = snapshot.bars.with_columns(
            pl.when(future).then(pl.col("open") * 1.01).otherwise(pl.col("open")).alias("open"),
            pl.when(future).then(pl.col("high") * 1.02).otherwise(pl.col("high")).alias("high"),
            pl.when(future).then(pl.col("low") * 0.98).otherwise(pl.col("low")).alias("low"),
            pl.when(future).then(pl.col("close") * 1.01).otherwise(pl.col("close")).alias("close"),
        )
        changed_path = tmp_path / "perturbed.csv"
        changed_bars.write_csv(changed_path)
        changed_snapshot = build_market_snapshot(
            [changed_path],
            calendar="XNYS",
            timeframe=Timeframe.D1,
            provenance={"fixture": "phase1-positive-reserved-perturbation"},
            source_manifest=[{"path": str(changed_path)}],
            metadata=metadata,
        )
        catalog = SnapshotCatalog(tmp_path / "snapshots")
        changed_entry = catalog.publish(changed_snapshot, stream_id="reserved-perturbation", coverage_status="accepted")
        catalog.connection.close()
        changed_run = run_experiment(_demo_config(changed_entry.snapshot_id))
        for original, changed in zip(source.predictions, changed_run.predictions, strict=True):
            assert original.keys == changed.keys
            np.testing.assert_allclose(original.predictions, changed.predictions, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(source.backtest.net_returns, changed_run.backtest.net_returns, rtol=1e-7, atol=1e-9)
        preview = preview_holdout(source.ref.run_id)
        assert preview_holdout(source.ref.run_id)["evidence"]["evidence_digest"] == preview["evidence"]["evidence_digest"]
        assert preview["verdict"] == "PASSED"
        with pytest.raises(ValueError, match="overrides are not supported"):
            evaluate_holdout(source.ref.run_id, allocation_overrides={})
        assert preview["evidence"]["baseline_loss"] > preview["evidence"]["selected_loss"]
        assert preview["evidence"]["trial_count"] == 1
        winner = preview["evidence"]["selected_model"]
        suffix = hashlib.sha256(
            json.dumps([winner, source.ref.config_hash, source.ref.split_hash], sort_keys=True).encode()
        ).hexdigest()[:12]
        reserved_child = f"{source.ref.run_id}-holdout-{suffix}"
        ledger = TrialLedger(tmp_path / "trials.sqlite")
        try:
            reservation = ledger.record_decision(
                decision_id="decision-before-claim",
                snapshot_hash=entry.snapshot_hash,
                source_run_id=source.ref.run_id,
                model_name=winner,
                verdict="PASSED",
                evidence=preview["evidence"],
                child_run_id=reserved_child,
            )
            assert reservation["verdict"] == "PASSED"
            assert ledger.seal(entry.snapshot_hash)["execution_status"] == "reserved"
        finally:
            ledger.close()
        update_attempt = TrialLedger.update_attempt

        def interrupt_after_artifact(self, attempt_id, *, status, result=None, error=None):
            if status == "completed" and result and result.get("run_id", "").find("-holdout-") >= 0:
                raise KeyboardInterrupt("simulated crash after finalized child artifact")
            return update_attempt(self, attempt_id, status=status, result=result, error=error)

        monkeypatch.setattr(TrialLedger, "update_attempt", interrupt_after_artifact)
        with pytest.raises(KeyboardInterrupt, match="simulated crash"):
            evaluate_holdout(source.ref.run_id)
        monkeypatch.setattr(TrialLedger, "update_attempt", update_attempt)
        recovering = TrialLedger.read_only(tmp_path / "trials.sqlite")
        try:
            assert recovering.seal(entry.snapshot_hash)["execution_status"] == "running"
        finally:
            recovering.close()
        first = evaluate_holdout(source.ref.run_id)
        replay = evaluate_holdout(source.ref.run_id)
        assert replay.ref.run_id == first.ref.run_id
        assert replay.artifact_manifest["decision_id"] == first.artifact_manifest["decision_id"]
        assert first.artifact_manifest["decision_id"] == "decision-before-claim"
        with pytest.raises(HoldoutRejected, match="HOLDOUT_COMPLETED"):
            evaluate_holdout(source.ref.run_id, model_name="baseline")
        ledger = TrialLedger.read_only(tmp_path / "trials.sqlite")
        try:
            assert ledger.seal(entry.snapshot_hash)["execution_status"] == "completed"
        finally:
            ledger.close()
    finally:
        get_platform_config.cache_clear()
