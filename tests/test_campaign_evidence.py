from lab.platform.evidence_store import EvidenceStore
from lab.research.trial_ledger import TrialLedger


def test_comparable_evidence_counts_logical_trials_and_deduplicates_attempts(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite")
    shared = {
        "campaign": "research",
        "config_hash": "config-a",
        "snapshot_hash": "snapshot-a",
        "evaluation_hash": "split-a",
        "settings": {
            "task": "regression",
            "frequency": "daily",
            "cost_treatment": "fees+slippage",
        },
    }
    logical, first_attempt, _ = ledger.register_attempt(**shared)
    ledger.update_attempt(first_attempt, status="completed", result={"development_net_returns": [0.01, -0.002, 0.003]})
    _, duplicate_attempt, is_retry = ledger.register_attempt(**shared)
    ledger.update_attempt(duplicate_attempt, status="completed", result={"development_net_returns": [0.02, -0.001, 0.002]})

    selected = ledger.comparable_return_series(
        campaign="research",
        snapshot_hash="snapshot-a",
        evaluation_hash="split-a",
        frequency="daily",
        task="regression",
        cost_treatment="fees+slippage",
    )

    assert is_retry
    assert selected["population"] == 1
    assert len(selected["complete"]) == 1
    assert selected["complete"][0]["logical_trial_id"] == logical
    assert len(selected["exclusions"]) == 1
    assert selected["exclusions"][0]["reason"] == "duplicate successful attempt"
    ledger.close()


def test_return_convention_filter_excludes_legacy_settings_and_attempt_results(tmp_path):
    ledger = TrialLedger(tmp_path / "conventions.sqlite")

    def register(name, settings_convention, result_convention):
        logical, attempt, _ = ledger.register_attempt(
            campaign="research",
            config_hash=name,
            snapshot_hash="snapshot-a",
            evaluation_hash="split-a",
            settings={"task": "regression", "frequency": "daily", "return_convention": settings_convention},
        )
        ledger.update_attempt(
            attempt,
            status="completed",
            result={"development_net_returns": [0.01, -0.002, 0.003], "return_convention": result_convention},
        )
        return logical

    matching = register("matching", "session-compounded.v1", "session-compounded.v1")
    register("legacy-settings", None, "session-compounded.v1")
    register("legacy-result", "session-compounded.v1", None)
    missing_logical, missing_attempt, _ = ledger.register_attempt(
        campaign="research",
        config_hash="missing",
        snapshot_hash="snapshot-a",
        evaluation_hash="split-a",
        settings={"task": "regression", "frequency": "daily", "return_convention": "session-compounded.v1"},
    )
    ledger.update_attempt(missing_attempt, status="completed", result={"return_convention": "session-compounded.v1"})

    selected = ledger.comparable_return_series(
        campaign="research",
        snapshot_hash="snapshot-a",
        evaluation_hash="split-a",
        frequency="daily",
        return_convention="session-compounded.v1",
    )

    assert [trial["logical_trial_id"] for trial in selected["complete"]] == [matching]
    assert len(selected["exclusions"]) == 3
    convention_exclusions = [item for item in selected["exclusions"] if "return_convention" in item["reason"]]
    missing_exclusion = next(item for item in selected["exclusions"] if item["logical_trial_id"] == missing_logical)
    assert len(convention_exclusions) == 2
    assert missing_exclusion["reason"] == "missing returns"
    assert selected["population"] == 4
    ledger.close()


def test_evidence_sidecar_is_immutable_and_checksum_verified(tmp_path):
    store = EvidenceStore(tmp_path)
    report = {"source_run_id": "run-1", "included_logical_trial_ids": ["trial-1"], "metric_values": {"dsr": 0.7}}

    first = store.write(report)
    second = store.write(report)

    assert first == second
    assert store.load(first["evidence_id"]) == report
