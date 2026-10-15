from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import pytest
from pydantic import ValidationError

from lab.core.contracts import CampaignPlan, CampaignRunEntry, ExperimentVariation
from lab.platform.campaign_store import CampaignStore


def _entry(status, *, created_at=None, previous=None, run_id=None, error=None):
    now = datetime.now(timezone.utc)
    return CampaignRunEntry(
        variation_id="variation-1", planned_identity="campaign-1", effective_config_hash="config-hash",
        snapshot_hash="snapshot-hash", status=status, created_at=created_at or now, updated_at=now,
        previous_status=previous, run_id=run_id, error=error,
    )


def test_campaign_status_transitions_and_completed_immutability(tmp_path):
    store = CampaignStore(tmp_path)
    planned = _entry("planned")
    store.upsert(planned)
    store.upsert(_entry("running", created_at=planned.created_at, previous="planned"))
    completed = _entry("completed", created_at=planned.created_at, previous="running", run_id="run-1")
    store.upsert(completed)

    assert store.entries("campaign-1")[0].run_id == "run-1"
    variation = ExperimentVariation(
        variation_id="variation-1", campaign="campaign", snapshot_id="snapshot-1",
        snapshot_hash="snapshot-hash", effective_config_hash="config-hash",
        feature_specification=("close",), seed=1,
    )
    plan = CampaignPlan(
        campaign_id="campaign-1", campaign="campaign", base_config_hash="base-hash",
        variations=(variation,), max_planned_experiments=1, worker_count=1,
    )
    result_ref = store.finalize(plan)
    assert result_ref is not None
    assert store.finalize(plan).result_id == result_ref.result_id
    with pytest.raises(ValidationError, match="Illegal campaign status transition"):
        _entry("failed", previous="completed", error="late failure")
    store.close()


def test_failed_campaign_result_is_preserved_when_retry_completes(tmp_path):
    store = CampaignStore(tmp_path)
    planned = _entry("planned")
    store.upsert(planned)
    store.upsert(_entry("running", created_at=planned.created_at, previous="planned"))
    store.upsert(_entry("failed", created_at=planned.created_at, previous="running", error="transient"))
    variation = ExperimentVariation(
        variation_id="variation-1", campaign="campaign", snapshot_id="snapshot-1",
        snapshot_hash="snapshot-hash", effective_config_hash="config-hash",
        feature_specification=("close",), seed=1,
    )
    plan = CampaignPlan(
        campaign_id="campaign-1", campaign="campaign", base_config_hash="base-hash",
        variations=(variation,), max_planned_experiments=1, worker_count=1,
    )

    failed_ref = store.finalize(plan)
    assert failed_ref is not None
    failed_bytes = Path(failed_ref.bundle_path).read_bytes()
    assert store.finalize(plan).bundle_path == failed_ref.bundle_path
    store.upsert(_entry("running", created_at=planned.created_at, previous="failed"))
    store.upsert(_entry("completed", created_at=planned.created_at, previous="running", run_id="run-1"))

    completed_ref = store.finalize(plan)

    assert completed_ref is not None
    assert completed_ref.bundle_path != failed_ref.bundle_path
    assert Path(failed_ref.bundle_path).read_bytes() == failed_bytes
    assert store.latest_result(plan.campaign_id).result_id == completed_ref.result_id
    assert store.finalize(plan).result_id == completed_ref.result_id
    store.close()


def test_campaign_result_create_only_write_rejects_conflicting_content(tmp_path):
    store = CampaignStore(tmp_path)
    path = store.root / "campaign-result-test.json"
    path.write_bytes(b"first")

    with pytest.raises(ValueError, match="identity collision"):
        store._write_create_only(path, b"different")

    assert path.read_bytes() == b"first"
    store.close()


def test_campaign_result_sidecar_recovers_after_index_commit_crash(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    planned = _entry("planned")
    store.upsert(planned)
    store.upsert(_entry("running", created_at=planned.created_at, previous="planned"))
    store.upsert(_entry("completed", created_at=planned.created_at, previous="running", run_id="run-1"))
    variation = ExperimentVariation(
        variation_id="variation-1", campaign="campaign", snapshot_id="snapshot-1",
        snapshot_hash="snapshot-hash", effective_config_hash="config-hash",
        feature_specification=("close",), seed=1,
    )
    plan = CampaignPlan(
        campaign_id="campaign-1", campaign="campaign", base_config_hash="base-hash",
        variations=(variation,), max_planned_experiments=1, worker_count=1,
    )
    write_create_only = store._write_create_only
    failed_once = False

    def crash_after_sidecar(path, data):
        nonlocal failed_once
        write_create_only(path, data)
        if not failed_once:
            failed_once = True
            raise OSError("simulated crash before index commit")

    monkeypatch.setattr(store, "_write_create_only", crash_after_sidecar)
    with pytest.raises(OSError, match="before index commit"):
        store.finalize(plan)
    monkeypatch.setattr(store, "_write_create_only", write_create_only)

    recovered = store.finalize(plan)

    assert recovered is not None
    assert store.latest_result(plan.campaign_id).result_id == recovered.result_id
    store.close()


def test_two_finalizers_publish_one_immutable_state_record(tmp_path):
    seed = CampaignStore(tmp_path)
    planned = _entry("planned")
    seed.upsert(planned)
    seed.upsert(_entry("running", created_at=planned.created_at, previous="planned"))
    seed.upsert(_entry("completed", created_at=planned.created_at, previous="running", run_id="run-1"))
    variation = ExperimentVariation(
        variation_id="variation-1", campaign="campaign", snapshot_id="snapshot-1",
        snapshot_hash="snapshot-hash", effective_config_hash="config-hash",
        feature_specification=("close",), seed=1,
    )
    plan = CampaignPlan(
        campaign_id="campaign-1", campaign="campaign", base_config_hash="base-hash",
        variations=(variation,), max_planned_experiments=1, worker_count=1,
    )
    seed.close()

    def finalize():
        store = CampaignStore(tmp_path)
        result = store.finalize(plan)
        store.close()
        return result.result_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        result_ids = list(executor.map(lambda _: finalize(), range(2)))
    latest = CampaignStore(tmp_path)

    assert result_ids[0] == result_ids[1]
    assert latest.latest_result(plan.campaign_id).result_id == result_ids[0]
    assert len(list(latest.root.glob(f"{plan.campaign_id}-result-*.json"))) == 1
    latest.close()


def test_campaign_mutations_require_current_fencing_generation(tmp_path):
    store = CampaignStore(tmp_path)
    now = datetime.now(timezone.utc)
    assert store.acquire_claim("campaign-1", "owner-1", lease_seconds=0.05, now=now)
    first_generation = int(store.claim("campaign-1")["fencing_generation"])
    planned = _entry("planned")
    store.upsert(planned, owner_id="owner-1", fencing_generation=first_generation)

    assert store.acquire_claim(
        "campaign-1", "owner-2", lease_seconds=30, now=now + timedelta(seconds=1),
    )
    second_generation = int(store.claim("campaign-1")["fencing_generation"])
    assert second_generation > first_generation
    with pytest.raises(RuntimeError, match="ownership lost"):
        store.upsert(
            _entry("running", created_at=planned.created_at, previous="planned"),
            owner_id="owner-1", fencing_generation=first_generation,
        )
    plan = CampaignPlan(
        campaign_id="campaign-1", campaign="campaign", base_config_hash="base-hash",
        variations=(ExperimentVariation(
            variation_id="variation-1", campaign="campaign", snapshot_id="snapshot-1",
            snapshot_hash="snapshot-hash", effective_config_hash="config-hash",
            feature_specification=("close",), seed=1,
        ),), max_planned_experiments=1, worker_count=1,
    )
    with pytest.raises(RuntimeError, match="ownership lost"):
        store.finalize(plan, owner_id="owner-1", fencing_generation=first_generation)
    assert not store.release_claim("campaign-1", "owner-1", fencing_generation=first_generation)
    assert store.claim("campaign-1")["owner_id"] == "owner-2"
    store.close()


def test_claim_mutation_credentials_must_be_paired(tmp_path):
    store = CampaignStore(tmp_path)
    with pytest.raises(ValueError, match="both owner_id and fencing_generation"):
        store.upsert(_entry("planned"), owner_id="owner-1")
    store.close()


def test_campaign_claim_table_migrates_existing_owner_rows(tmp_path):
    database = tmp_path / "campaigns" / "campaigns.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE campaign_claims (campaign_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, expires_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO campaign_claims VALUES (?, ?, ?, ?)",
        ("campaign-1", "expired-owner", "2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    store = CampaignStore(tmp_path)

    assert store.acquire_claim("campaign-1", "new-owner", lease_seconds=30)
    assert store.claim("campaign-1")["fencing_generation"] == 1
    store.close()
