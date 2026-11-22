import numpy as np

from lab.research.holdout_gate import evaluate_gate


def _candidate(name, loss):
    return {
        "model": name,
        "status": "completed",
        "metrics": [{"name": "mse", "status": "available", "value": loss, "inputs": {"n": 80}}],
    }


def _gate(*, trial_count=1, comparable=None, returns=None, winner_loss=0.5):
    return evaluate_gate(
        candidates=[_candidate("baseline", 1.0), _candidate("xgboost", winner_loss)],
        selected_model="xgboost",
        task="regression",
        returns=returns or list(np.linspace(0.001, 0.03, 60)),
        comparable=comparable or [],
        exclusions=[],
        trial_count=trial_count,
        frequency="native",
        min_observations=30,
    )


def test_baseline_tie_is_rejected_before_statistical_scoring():
    result = _gate(winner_loss=1.0)
    assert result["verdict"] == "REJECTED_BY_BASELINE"
    assert result["evidence"]["score"] is None


def test_classification_requires_strict_log_loss_win_with_equal_coverage():
    base = {
        "model": "baseline",
        "status": "completed",
        "metrics": [{"name": "log_loss", "status": "available", "value": 0.6, "inputs": {"n": 80}}],
    }
    challenger = {
        "model": "xgboost",
        "status": "completed",
        "metrics": [{"name": "log_loss", "status": "available", "value": 0.6, "inputs": {"n": 80}}],
    }

    def check():
        return evaluate_gate(
            candidates=[base, challenger],
            selected_model="xgboost",
            task="classification",
            returns=list(np.linspace(0.001, 0.03, 60)),
            comparable=[],
            exclusions=[],
            trial_count=1,
            frequency="native",
            min_observations=30,
        )

    assert check()["verdict"] == "REJECTED_BY_BASELINE"
    challenger["metrics"][0]["value"] = 0.5
    challenger["metrics"][0]["inputs"]["n"] = 79
    assert check()["verdict"] == "REJECTED_INSUFFICIENT_EVIDENCE"
    challenger["metrics"][0]["inputs"]["n"] = 80
    assert check()["verdict"] == "PASSED"


def test_single_trial_uses_psr_against_zero():
    result = _gate()
    assert result["evidence"]["statistic"] == "psr"
    assert result["evidence"]["benchmark"] == 0.0
    assert result["verdict"] == "PASSED"


def test_nonfinite_requested_returns_fail_closed():
    result = _gate(returns=[0.01] * 59 + [float("nan")])
    assert result["verdict"] == "REJECTED_INSUFFICIENT_EVIDENCE"
    assert result["evidence"]["reason"] == "REQUESTED_RETURN_SERIES_UNUSABLE"


def test_multiple_trial_gate_requires_comparable_series_and_positive_variance():
    missing = _gate(trial_count=2)
    assert missing["verdict"] == "REJECTED_INSUFFICIENT_EVIDENCE"
    assert missing["evidence"]["reason"] == "FEWER_THAN_TWO_COMPARABLE_CANDIDATES"
    constant = [0.01] * 60
    zero_variance = _gate(
        trial_count=2,
        comparable=[{"attempt_id": "a", "returns": constant}, {"attempt_id": "b", "returns": constant}],
    )
    assert zero_variance["verdict"] == "REJECTED_INSUFFICIENT_EVIDENCE"
    assert zero_variance["evidence"]["reason"] == "COMPARABLE_SHARPE_VARIANCE_UNAVAILABLE"


def test_dsr_rejects_weak_challenger_despite_baseline_loss_win_and_passes_strong_one():
    innovations = np.sin(np.arange(90) * 1.7) * 0.01
    other = (innovations + 0.0005).tolist()
    for mean, expected in [(0.0001, "REJECTED_BY_DSR"), (0.004, "PASSED")]:
        selected = (innovations + mean).tolist()
        result = _gate(
            trial_count=3,
            returns=selected,
            comparable=[{"attempt_id": "a", "returns": selected}, {"attempt_id": "b", "returns": other}],
        )
        assert result["verdict"] == expected
        assert result["evidence"]["statistic"] == "dsr"
        assert result["evidence"]["variance"] > 0.0
        assert result["evidence"]["baseline_loss"] > result["evidence"]["selected_loss"]
