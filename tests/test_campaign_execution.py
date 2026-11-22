from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import polars as pl

import lab.research.campaign as campaign
from lab.core.config import PipelineConfig
from lab.core.contracts import CampaignPlan, CampaignRunEntry, ExperimentVariation, MarketSnapshot
from lab.platform.artifacts import ArtifactStore
from lab.platform.campaign_store import CampaignStore
from lab.research.trial_ledger import TrialLedger


def _plan():
    variations = tuple(
        ExperimentVariation(
            variation_id=f"variation-{index}",
            campaign="execution",
            snapshot_id="snapshot-1",
            snapshot_hash="snapshot-hash",
            effective_config_hash=f"config-{index}",
            feature_specification=("return_1b",),
            seed=index,
        )
        for index in (1, 2)
    )
    return CampaignPlan(
        campaign_id="campaign-execution",
        campaign="execution",
        base_config_hash="base-hash",
        variations=variations,
        max_planned_experiments=2,
        worker_count=2,
    )


def test_campaign_records_independent_results_and_skips_completed_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "get_platform_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    calls = []

    def execute_one(_config, variation, *_claim):
        ledger = TrialLedger.read_only(tmp_path / "trials.sqlite")
        try:
            assert ledger.connection.execute("SELECT 1 FROM schema_migrations WHERE version=3").fetchone() == (1,)
        finally:
            ledger.close()
        calls.append(variation.variation_id)
        if variation.variation_id == "variation-2":
            return {"variation_id": variation.variation_id, "status": "failed", "error": "fixture failure"}
        return {"variation_id": variation.variation_id, "status": "completed", "run_id": "run-1"}

    monkeypatch.setattr(campaign, "_execute_one", execute_one)
    monkeypatch.setattr(campaign, "index_deferred_run", lambda run_id: {"status": "disabled", "run_id": run_id})
    executor_limits = []

    class StubProcessPoolExecutor:
        def __init__(self, max_workers, mp_context):
            executor_limits.append(max_workers)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.executor.shutdown(wait=True)

        def submit(self, function, *args):
            return self.executor.submit(function, *args)

    monkeypatch.setattr(campaign, "ProcessPoolExecutor", StubProcessPoolExecutor)
    plan = _plan()
    base_config = PipelineConfig()
    base_config.models.device = "cpu"

    first = campaign.execute_campaign(base_config, plan)
    second = campaign.execute_campaign(base_config, plan)

    assert [item["status"] for item in first] == ["completed", "failed"]
    assert [item["status"] for item in second] == ["failed"]
    assert calls.count("variation-1") == 1
    assert calls.count("variation-2") == 2
    assert executor_limits == [2]


def test_campaign_claim_allows_one_concurrent_owner(tmp_path):
    from threading import Barrier

    barrier = Barrier(2)

    def claim(owner):
        store = CampaignStore(tmp_path)
        barrier.wait()
        result = store.acquire_claim("campaign-lock", owner, lease_seconds=30)
        store.close()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(claim, ("owner-1", "owner-2")))

    assert sorted(outcomes) == [False, True]


def test_campaign_claim_rejects_live_owner_and_takes_over_expired_lease(tmp_path):
    store = CampaignStore(tmp_path)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert store.acquire_claim("campaign-lock", "owner-1", lease_seconds=30, now=now)
    assert not store.acquire_claim("campaign-lock", "owner-2", lease_seconds=30, now=now + timedelta(seconds=10))
    assert store.acquire_claim("campaign-lock", "owner-2", lease_seconds=30, now=now + timedelta(seconds=31))
    assert not store.renew_claim("campaign-lock", "owner-1", lease_seconds=30)
    assert store.claim("campaign-lock")["owner_id"] == "owner-2"
    store.close()


def _seed_running_campaign(tmp_path, plan):
    store = CampaignStore(tmp_path)
    store.write_plan(plan)
    variation = plan.variations[0]
    created = datetime.now(timezone.utc) - timedelta(minutes=1)
    planned = CampaignRunEntry(
        variation_id=variation.variation_id,
        planned_identity=plan.campaign_id,
        effective_config_hash=variation.effective_config_hash,
        snapshot_hash=variation.snapshot_hash,
        status="planned",
        created_at=created,
        updated_at=created,
    )
    store.upsert(planned)
    store.upsert(
        planned.model_copy(update={"status": "running", "previous_status": "planned", "updated_at": datetime.now(timezone.utc)})
    )
    store.close()


def test_campaign_reconciles_completed_run_after_status_write_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "get_platform_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    plan = _plan().model_copy(update={"variations": (_plan().variations[0],), "max_planned_experiments": 1, "worker_count": 1})
    _seed_running_campaign(tmp_path, plan)
    monkeypatch.setattr(campaign, "_reconcile_completed_run", lambda *_args: "run-recovered")
    monkeypatch.setattr(campaign, "_execute_one", lambda *_args: (_ for _ in ()).throw(AssertionError("must not retrain")))
    monkeypatch.setattr(campaign, "index_deferred_run", lambda *_args: (_ for _ in ()).throw(RuntimeError("MLflow unavailable")))

    results = campaign.execute_campaign(PipelineConfig(), plan)

    assert results[0]["reconciled"] is True
    assert results[0]["tracking"]["status"] == "failed"
    store = CampaignStore(tmp_path)
    entry = store.entries(plan.campaign_id)[0]
    assert entry.status == "completed"
    assert entry.run_id == "run-recovered"
    assert store.latest_result(plan.campaign_id) is not None
    store.close()


def test_campaign_marks_interrupted_variation_failed_when_no_completed_run_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "get_platform_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    plan = _plan().model_copy(update={"variations": (_plan().variations[0],), "max_planned_experiments": 1, "worker_count": 1})
    _seed_running_campaign(tmp_path, plan)
    monkeypatch.setattr(campaign, "_reconcile_completed_run", lambda *_args: None)
    monkeypatch.setattr(
        campaign,
        "_execute_one",
        lambda _config, variation, *_claim: {"variation_id": variation.variation_id, "status": "failed", "error": "interrupted"},
    )

    result = campaign.execute_campaign(PipelineConfig(), plan)

    assert result[0]["status"] == "failed"
    store = CampaignStore(tmp_path)
    assert store.entries(plan.campaign_id)[0].status == "failed"
    store.close()


def _persist_completed_root(artifacts_root, base_config, variation, run_id, *, parent_run_id=None, markers=None, complete=True):
    config = campaign._variation_config(base_config, variation)
    snapshot = MarketSnapshot(
        snapshot_id=variation.snapshot_id,
        snapshot_hash=variation.snapshot_hash,
        bars=pl.DataFrame(
            {
                "ticker": ["AAA"],
                "timestamp": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [10.0],
            }
        ),
        ticker_order=("AAA",),
        calendar=config.data.calendar,
        timeframe=config.data.timeframe,
    )
    artifacts = ArtifactStore(artifacts_root)
    artifacts.create_run(run_id)
    records = [
        artifacts.write_json(run_id, "predictions.json", []),
        artifacts.write_json(run_id, "snapshot/metadata.json", snapshot.model_dump(mode="json", exclude={"bars"})),
        artifacts.write_parquet(run_id, "snapshot/bars.parquet", snapshot.bars),
        artifacts.write_json(run_id, "allocations.json", []),
    ]
    if complete:
        records.append(artifacts.write_json(run_id, "metrics.json", []))
    manifest = {
        "config": config.model_dump(mode="json"),
        "config_hash": config.resolved_config_hash,
        "snapshot_hash": variation.snapshot_hash,
        "split_hash": "split-hash",
        "candidates": [{"model": "baseline", "status": "completed"}],
        "artifacts": records,
        "parent_run_id": parent_run_id,
        **(markers or {}),
    }
    if complete:
        artifacts.finalize(run_id, manifest, required_artifacts=[item["path"] for item in records])
    else:
        artifacts.write_json(
            run_id,
            "manifest.json",
            {
                **manifest,
                "schema_version": "run-manifest.v1",
                "run_id": run_id,
                "status": "completed",
            },
        )


def test_reconcile_accepts_only_verified_root_development_artifacts(tmp_path):
    base_config = PipelineConfig()
    variation = _plan().variations[0]
    variation = variation.model_copy(
        update={
            "effective_config_hash": campaign._variation_config(base_config, variation).resolved_config_hash,
        }
    )
    _persist_completed_root(tmp_path, base_config, variation, "a-root")
    _persist_completed_root(tmp_path, base_config, variation, "b-holdout-child", parent_run_id="a-root")
    _persist_completed_root(tmp_path, base_config, variation, "c-evaluation", markers={"evaluation": "holdout"})
    _persist_completed_root(tmp_path, base_config, variation, "d-backtest-child", markers={"parent_model_ref": "a-root/model"})
    _persist_completed_root(tmp_path, base_config, variation, "e-wrong-config", markers={"config_hash": "wrong-config"})
    _persist_completed_root(tmp_path, base_config, variation, "f-wrong-snapshot", markers={"snapshot_hash": "wrong-snapshot"})
    wrong_campaign = campaign._variation_config(base_config, variation).model_dump(mode="json")
    wrong_campaign["campaign"]["name"] = "another-campaign"
    _persist_completed_root(
        tmp_path,
        base_config,
        variation,
        "g-wrong-campaign",
        markers={"config": wrong_campaign},
    )
    _persist_completed_root(tmp_path, base_config, variation, "h-incomplete", complete=False)
    _persist_completed_root(tmp_path, base_config, variation, "i-corrupt")
    corrupt = ArtifactStore(tmp_path).run_dir("i-corrupt") / "metrics.json"
    corrupt.write_text("corrupted after manifest checksum creation")

    assert campaign._reconcile_completed_run(base_config, variation, tmp_path) == "a-root"


def test_reconcile_is_deterministic_across_matching_root_runs(tmp_path):
    base_config = PipelineConfig()
    variation = _plan().variations[0]
    variation = variation.model_copy(
        update={
            "effective_config_hash": campaign._variation_config(base_config, variation).resolved_config_hash,
        }
    )
    _persist_completed_root(tmp_path, base_config, variation, "z-root")
    _persist_completed_root(tmp_path, base_config, variation, "a-root")

    assert campaign._reconcile_completed_run(base_config, variation, tmp_path) == "a-root"


def test_execute_campaign_recovers_verified_root_without_retraining(tmp_path, monkeypatch):
    base_config = PipelineConfig()
    original = _plan().variations[0]
    variation = original.model_copy(
        update={
            "effective_config_hash": campaign._variation_config(base_config, original).resolved_config_hash,
        }
    )
    plan = _plan().model_copy(update={"variations": (variation,), "max_planned_experiments": 1, "worker_count": 1})
    monkeypatch.setattr(
        campaign,
        "get_platform_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    _seed_running_campaign(tmp_path, plan)
    _persist_completed_root(tmp_path, base_config, variation, "verified-root")
    monkeypatch.setattr(campaign, "_execute_one", lambda *_args: (_ for _ in ()).throw(AssertionError("must not retrain")))
    monkeypatch.setattr(campaign, "index_deferred_run", lambda run_id: {"status": "disabled", "run_id": run_id})

    results = campaign.execute_campaign(base_config, plan)

    assert results[0]["run_id"] == "verified-root"
    assert results[0]["reconciled"] is True
    store = CampaignStore(tmp_path)
    assert store.entries(plan.campaign_id)[0].run_id == "verified-root"
    assert store.latest_result(plan.campaign_id) is not None
    store.close()


def test_worker_refuses_to_start_after_campaign_claim_is_lost(tmp_path, monkeypatch):
    plan = _plan()
    store = CampaignStore(tmp_path)
    assert store.acquire_claim(plan.campaign_id, "current-owner", lease_seconds=30)
    generation = int(store.claim(plan.campaign_id)["fencing_generation"])
    store.close()
    monkeypatch.setattr(
        campaign, "run_experiment", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker must not launch"))
    )

    result = campaign._execute_one(
        PipelineConfig(),
        plan.variations[0],
        tmp_path,
        plan.campaign_id,
        "stale-owner",
        generation,
    )

    assert result["status"] == "cancelled"
