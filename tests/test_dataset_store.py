import hashlib
import json
from datetime import datetime, timezone

import polars as pl
import pytest

from lab.core.config import PipelineConfig, Timeframe
from lab.core.contracts import ExecutionPeriod, FoldSpec, PreparedDataset, SplitPlan
from lab.platform.artifacts import canonical_json_bytes
from lab.platform.data_access import build_market_snapshot
from lab.platform.dataset_store import DatasetStore
from lab.platform.market_store import SnapshotCatalog
from lab.quant.timing import bar_times_for_session


def _prepared_dataset():
    opened, closed, session_id = bar_times_for_session("XNYS", "2024-01-02", Timeframe.D1)[0]
    bars = pl.DataFrame(
        {
            "ticker": ["AAA"],
            "raw_bar_index": [0],
            "bar_open_time": [opened],
            "bar_close_time": [closed],
            "timestamp": [closed],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [500.0],
            "session_id": [session_id],
        }
    )
    snapshot = build_market_snapshot([bars], calendar="XNYS", timeframe=Timeframe.D1, provenance={"source": "test"})
    utc = timezone.utc
    split_plan = SplitPlan(
        decision_timestamps=(datetime(2024, 1, 3, tzinfo=utc), datetime(2024, 1, 4, tzinfo=utc)),
        development_start=datetime(2024, 1, 1, tzinfo=utc),
        holdout_start=datetime(2024, 1, 5, tzinfo=utc),
        holdout_end=datetime(2024, 1, 8, tzinfo=utc),
        folds=(
            FoldSpec(
                fold_id=0,
                train_start=datetime(2024, 1, 1, tzinfo=utc),
                train_end=datetime(2024, 1, 3, tzinfo=utc),
                validation_start=datetime(2024, 1, 3, tzinfo=utc),
                validation_end=datetime(2024, 1, 4, tzinfo=utc),
            ),
        ),
        execution_periods=(
            ExecutionPeriod(
                name="development",
                first_decision=datetime(2024, 1, 3, tzinfo=utc),
                last_decision=datetime(2024, 1, 4, tzinfo=utc),
                final_liquidation_open=datetime(2024, 1, 5, tzinfo=utc),
            ),
        ),
    )
    config = PipelineConfig(data={"timeframe": "1d"})
    return PreparedDataset(
        snapshot=snapshot,
        features=pl.DataFrame({"timestamp": [closed], "ticker": ["AAA"], "return_1b": [0.01]}),
        feature_specification=("return_1b",),
        labels=pl.DataFrame({"ticker": ["AAA"], "decision_time": [closed], "label": [1]}),
        split_plan=split_plan,
        exclusions=pl.DataFrame(),
        config=config,
    )


def test_prepared_dataset_round_trip_checksums_and_snapshot_identity(tmp_path):
    artifacts = tmp_path / "artifacts"
    snapshots = tmp_path / "snapshots"
    dataset = _prepared_dataset()
    SnapshotCatalog(snapshots).publish(dataset.snapshot)
    store = DatasetStore(artifacts, snapshot_root=snapshots)
    reference = store.write(dataset, config_hash=dataset.config.resolved_config_hash, split_hash="split-hash")
    loaded = store.load(reference)

    assert loaded.snapshot.snapshot_hash == dataset.snapshot.snapshot_hash
    assert loaded.split_plan.model_dump() == dataset.split_plan.model_dump()
    assert loaded.features.equals(dataset.features)
    (tmp_path / "artifacts" / "datasets" / reference.dataset_id / "labels.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file checksum"):
        store.verify(reference)


def test_dataset_write_publishes_snapshot_and_reuses_immutable_catalog_entry(tmp_path):
    artifacts = tmp_path / "custom-artifacts"
    snapshots = tmp_path / "custom-snapshots"
    dataset = _prepared_dataset()
    store = DatasetStore(artifacts, snapshot_root=snapshots)

    reference = store.write(dataset, split_hash="split-hash")
    repeated = store.write(dataset, split_hash="split-hash")
    catalog = SnapshotCatalog(snapshots)

    assert repeated == reference
    assert catalog.resolve(dataset.snapshot.snapshot_id).coverage_status == "accepted"
    assert store.load(reference).snapshot.snapshot_hash == dataset.snapshot.snapshot_hash


def test_dataset_write_preserves_snapshot_coverage_and_rejects_catalog_tampering(tmp_path):
    artifacts = tmp_path / "artifacts"
    snapshots = tmp_path / "snapshots"
    dataset = _prepared_dataset()
    catalog = SnapshotCatalog(snapshots)
    catalog.publish(dataset.snapshot, coverage_status="incomplete")
    store = DatasetStore(artifacts, snapshot_root=snapshots)

    store.write(dataset, split_hash="split-hash")

    assert catalog.resolve(dataset.snapshot.snapshot_id).coverage_status == "incomplete"
    bars_path = snapshots / dataset.snapshot.snapshot_id / "bars.parquet"
    bars_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Snapshot bars checksum mismatch"):
        store.write(dataset, split_hash="split-hash")


def test_prepared_dataset_loader_migrates_versioned_legacy_noop_config(tmp_path):
    artifacts = tmp_path / "artifacts"
    snapshots = tmp_path / "snapshots"
    dataset = _prepared_dataset()
    SnapshotCatalog(snapshots).publish(dataset.snapshot)
    store = DatasetStore(artifacts, snapshot_root=snapshots)
    reference = store.write(dataset, split_hash="split-hash")
    bundle = artifacts / "datasets" / reference.dataset_id
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("checksum")
    manifest.pop("config_schema_version")
    manifest.pop("config_source_provenance")
    manifest["config"]["data"]["raw_data_dir"] = "data/raw"
    manifest["config"]["data"]["processed_data_dir"] = "data/processed"
    manifest["config"]["validation"]["require_timezone"] = True
    manifest["config"]["tensor"]["train_cutoff_date"] = "2025-11-20"
    manifest["checksum"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    loaded = store.load(store.resolve(reference.dataset_id))

    assert loaded.config is not None
    assert loaded.config.source_provenance["migrated_legacy_fields"]
    assert loaded.config.source_provenance["legacy_artifact_config_hash"] == reference.config_hash
