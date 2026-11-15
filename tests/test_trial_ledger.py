from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pytest

from lab.research.trial_ledger import TrialLedger


def _racing_seal(path, index):
    ledger = TrialLedger(path)
    try:
        return ledger.record_decision(
            decision_id=f"decision-{index}",
            snapshot_hash="same",
            source_run_id=f"run-{index}",
            model_name="xgboost",
            verdict="PASSED",
            evidence={"policy_version": "test"},
            child_run_id=f"child-{index}",
        )
    finally:
        ledger.close()


def _wait_for_ledger_race(barrier):
    barrier.wait(timeout=15)


def _initialize_cold_ledger(path):
    ledger = TrialLedger(path)
    try:
        return ledger.reserve_snapshot(
            snapshot_hash="cold-start",
            holdout_start="start",
            holdout_end="end",
        )
    finally:
        ledger.close()


def test_snapshot_reservation_is_immutable_and_idempotent(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite")
    first = ledger.reserve_snapshot(
        snapshot_hash="snapshot-1", holdout_start="2024-01-10T00:00:00+00:00", holdout_end="2024-01-20T00:00:00+00:00"
    )
    second = ledger.reserve_snapshot(
        snapshot_hash="snapshot-1", holdout_start="2024-01-10T00:00:00+00:00", holdout_end="2024-01-20T00:00:00+00:00"
    )
    assert first == second
    with pytest.raises(ValueError, match="boundary differs"):
        ledger.reserve_snapshot(
            snapshot_hash="snapshot-1", holdout_start="2024-01-11T00:00:00+00:00", holdout_end="2024-01-20T00:00:00+00:00"
        )
    ledger.close()


def test_reservation_created_after_legacy_trials_is_ineligible(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite")
    ledger.register_attempt(
        campaign="old",
        config_hash="config",
        snapshot_hash="snapshot-legacy",
        evaluation_hash="split",
        settings={"task": "regression"},
    )
    reservation = ledger.reserve_snapshot(
        snapshot_hash="snapshot-legacy", holdout_start="2024-01-10T00:00:00+00:00", holdout_end="2024-01-20T00:00:00+00:00"
    )
    assert reservation["eligibility"] == "legacy_boundary_unverified"
    ledger.close()


def test_candidate_accounting_deduplicates_aliases_and_counts_variants(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite")
    settings = {
        "task": "regression",
        "models": {"xgboost": {"enabled": True, "seed": 2}, "gru": {"enabled": True, "seed": 3}, "baseline": {"enabled": True}},
        "candidate_context": {"allocation": {"policy": "fixed"}},
        "frequency": "daily",
        "return_convention": "simple",
        "calendar": "XNYS",
        "cost_treatment": "fees+slippage",
        "execution_evidence_version": "period-bounded-next-open.v2",
    }
    for campaign in ("one", "renamed"):
        _, attempt_id, _ = ledger.register_attempt(
            campaign=campaign, config_hash=campaign, snapshot_hash="snapshot", evaluation_hash="split", settings=settings
        )
        if campaign == "renamed":
            ledger.update_attempt(
                attempt_id,
                status="completed",
                result={
                    "run_id": "alias-run",
                    "challenger_returns": {
                        "xgboost": {
                            "returns": [0.01 + index * 0.0001 for index in range(30)],
                            "timestamps": [str(index) for index in range(30)],
                        }
                    },
                },
            )
    ledger.register_attempt(
        campaign="variant",
        config_hash="other",
        snapshot_hash="snapshot",
        evaluation_hash="split",
        settings={
            **settings,
            "evaluation": "development",
            "parent_prediction_run": "original",
            "model": "xgboost",
            "model_spec": settings["models"]["xgboost"],
            "candidate_context": {"allocation": {"policy": "variable"}},
        },
    )
    evidence = ledger.candidate_search_evidence(
        snapshot_hash="snapshot",
        evaluation_hash="split",
        frequency="daily",
        task="regression",
        return_convention="simple",
        min_observations=30,
    )
    assert evidence["K"] == 3
    assert len(evidence["complete"]) == 1
    assert evidence["complete"][0]["run_id"] == "alias-run"
    assert len(evidence["exclusions"]) == 2
    ledger.close()


def test_two_independent_ledger_connections_can_seal_snapshot_only_once(tmp_path):
    path = tmp_path / "trials.sqlite"
    initial = TrialLedger(path)
    initial.reserve_snapshot(snapshot_hash="same", holdout_start="start", holdout_end="end")
    initial.close()

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        decisions = list(pool.map(_racing_seal, (path, path), (1, 2)))
    assert sorted(item["verdict"] for item in decisions) == ["PASSED", "REJECTED_HOLDOUT_SEALED"]
    ledger = TrialLedger.read_only(path)
    try:
        assert ledger.seal("same")["decision_id"] == next(
            item["decision_id"] for item in decisions if item["verdict"] == "PASSED"
        )
    finally:
        ledger.close()


def test_concurrent_first_initialization_of_nonexistent_ledger_is_serialized(tmp_path):
    path = tmp_path / "cold-trials.sqlite"
    worker_count = 4
    context = get_context("spawn")
    barrier = context.Barrier(worker_count)
    assert not path.exists()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_wait_for_ledger_race,
        initargs=(barrier,),
    ) as pool:
        reservations = list(pool.map(_initialize_cold_ledger, (path,) * worker_count))

    assert all(item["eligibility"] == "eligible" for item in reservations)
    assert all(item == reservations[0] for item in reservations)
    ledger = TrialLedger.read_only(path)
    try:
        assert ledger.connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall() == [
            (1,),
            (2,),
            (3,),
        ]
    finally:
        ledger.close()


def test_incompatible_calendar_and_split_still_count_in_global_search(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite")
    base = {
        "task": "regression",
        "models": {"xgboost": {"enabled": True, "seed": 1}},
        "candidate_context": {"schedule": "first"},
        "frequency": "daily",
        "return_convention": "simple",
        "calendar": "XNYS",
        "cost_treatment": "fees+slippage",
        "execution_evidence_version": "period-bounded-next-open.v2",
    }
    for index, (split, calendar) in enumerate((("split", "XNYS"), ("split", "XHKG"), ("other", "XNYS"))):
        ledger.register_attempt(
            campaign=str(index),
            config_hash=str(index),
            snapshot_hash="same",
            evaluation_hash=split,
            settings={**base, "calendar": calendar, "models": {"xgboost": {"enabled": True, "seed": index + 1}}},
        )
    evidence = ledger.candidate_search_evidence(
        snapshot_hash="same",
        evaluation_hash="split",
        frequency="daily",
        task="regression",
        return_convention="simple",
        min_observations=30,
        calendar="XNYS",
    )
    assert evidence["K"] == 3
    assert len(evidence["exclusions"]) == 3
    assert any(item["reason"] == "incompatible candidate evidence" for item in evidence["exclusions"])
    ledger.close()
