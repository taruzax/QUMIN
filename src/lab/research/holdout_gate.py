"""Pure Phase 1 holdout eligibility and statistical score rules."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from lab.quant.metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_statistics

POLICY_VERSION = "phase1-containment.v1"
THRESHOLD = 0.95


class HoldoutRejected(RuntimeError):
    def __init__(self, decision: dict[str, Any]):
        self.decision = decision
        super().__init__(str(decision.get("evidence", {}).get("reason", decision.get("verdict"))))


def _loss(candidate: dict[str, Any], metric_name: str) -> tuple[float | None, int | None]:
    metric = next((item for item in candidate.get("metrics", []) if item.get("name") == metric_name), None)
    if not metric or metric.get("status") != "available":
        return None, None
    value = metric.get("value")
    count = metric.get("inputs", {}).get("n")
    if value is None or not np.isfinite(float(value)) or count is None:
        return None, None
    return float(value), int(count)


def evaluate_gate(
    *,
    candidates: list[dict[str, Any]],
    selected_model: str,
    task: str,
    returns: list[float] | None,
    comparable: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    trial_count: int,
    frequency: str,
    min_observations: int,
) -> dict[str, Any]:
    """Compare the saved winner with baseline, then score its OOF net returns."""
    metric_name = "mse" if task == "regression" else "log_loss"
    baseline = next((item for item in candidates if item.get("model") == "baseline" and item.get("status") == "completed"), None)
    winner = next(
        (item for item in candidates if item.get("model") == selected_model and item.get("status") == "completed"), None
    )
    evidence: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "threshold": THRESHOLD,
        "statistic": "psr" if trial_count <= 1 else "dsr",
        "trial_count": int(trial_count),
        "comparable_count": len(comparable),
        "included_attempt_ids": [item.get("attempt_id") for item in comparable],
        "comparable_sharpes": [],
        "exclusions": exclusions,
        "selected_model": selected_model,
        "task": task,
        "metric": metric_name,
        "raw_sharpe": None,
        "variance": None,
        "benchmark": 0.0,
        "score": None,
        "baseline_loss": None,
        "selected_loss": None,
        "scoring_coverage": None,
        "returns_hash": None,
    }
    if baseline is None or winner is None:
        evidence["reason"] = "BASELINE_OR_WINNER_UNAVAILABLE"
        return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
    baseline_loss, baseline_count = _loss(baseline, metric_name)
    winner_loss, winner_count = _loss(winner, metric_name)
    evidence.update({"baseline_loss": baseline_loss, "selected_loss": winner_loss, "scoring_coverage": winner_count})
    if baseline_loss is None or winner_loss is None or baseline_count != winner_count:
        evidence["reason"] = "BASELINE_SCORING_COVERAGE_MISMATCH"
        return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
    if winner_loss >= baseline_loss:
        evidence["reason"] = "WINNER_DID_NOT_STRICTLY_BEAT_BASELINE"
        return {"verdict": "REJECTED_BY_BASELINE", "evidence": evidence}
    if returns is None or len(returns) == 0:
        evidence["reason"] = "REQUESTED_RUN_RETURN_EVIDENCE_MISSING"
        return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
    values = np.asarray(returns, dtype=float)
    if not np.isfinite(values).all():
        evidence["reason"] = "REQUESTED_RETURN_SERIES_UNUSABLE"
        return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
    evidence["returns_hash"] = hashlib.sha256(
        json.dumps(values.tolist(), separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if trial_count <= 1:
        score = probabilistic_sharpe_ratio(values, threshold=0.0, frequency=frequency, min_observations=max(30, min_observations))
        evidence["statistic"] = "psr"
        evidence["benchmark"] = 0.0
    else:
        if len(comparable) < 2:
            evidence["reason"] = "FEWER_THAN_TWO_COMPARABLE_CANDIDATES"
            return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
        sharpes = []
        try:
            for item in comparable:
                sharpes.append(
                    float(
                        sharpe_statistics(item["returns"], frequency=frequency, min_observations=max(30, min_observations))[
                            "sharpe"
                        ]
                    )
                )
        except (TypeError, ValueError) as exc:
            evidence["reason"] = f"COMPARABLE_RETURN_SERIES_UNUSABLE: {exc}"
            return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
        variance = float(np.var(np.asarray(sharpes), ddof=1))
        evidence["comparable_sharpes"] = [
            {"attempt_id": item.get("attempt_id"), "sharpe": value} for item, value in zip(comparable, sharpes, strict=True)
        ]
        evidence["variance"] = variance
        if not np.isfinite(variance) or variance <= 0.0:
            evidence["reason"] = "COMPARABLE_SHARPE_VARIANCE_UNAVAILABLE"
            return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
        score = deflated_sharpe_ratio(
            values,
            effective_trial_count=float(trial_count),
            trial_sharpe_variance=variance,
            frequency=frequency,
            min_observations=max(30, min_observations),
        )
        evidence["statistic"] = "dsr"
        evidence["benchmark"] = score.inputs.get("benchmark")
    if score.status != "available" or score.value is None or not np.isfinite(score.value):
        evidence["reason"] = score.reason or "STATISTICAL_SCORE_UNAVAILABLE"
        return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
    try:
        evidence["raw_sharpe"] = float(
            sharpe_statistics(values, frequency=frequency, min_observations=max(30, min_observations))["sharpe"]
        )
    except (TypeError, ValueError):
        evidence["reason"] = "REQUESTED_RETURN_SERIES_UNUSABLE"
        return {"verdict": "REJECTED_INSUFFICIENT_EVIDENCE", "evidence": evidence}
    evidence["score"] = float(score.value)
    if evidence["raw_sharpe"] <= 0.0:
        evidence["reason"] = "NONPOSITIVE_NET_SHARPE"
        return {"verdict": "REJECTED_BY_DSR", "evidence": evidence}
    evidence["reason"] = "SCORE_BELOW_THRESHOLD" if score.value < THRESHOLD else "GATE_PASSED"
    return {"verdict": "PASSED" if score.value >= THRESHOLD else "REJECTED_BY_DSR", "evidence": evidence}
