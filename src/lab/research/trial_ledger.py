"""Transactional local campaign and trial accounting."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_INITIALIZATION_ATTEMPTS = 8
_INITIALIZATION_BUSY_TIMEOUT_MS = 100
_OPERATION_BUSY_TIMEOUT_MS = 30_000


def _is_sqlite_busy_error(error: sqlite3.OperationalError) -> bool:
    """Identify lock contention without retrying unrelated SQLite failures."""
    error_code = getattr(error, "sqlite_errorcode", None)
    if isinstance(error_code, int) and error_code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(error).lower()
    return "locked" in message or "busy" in message


class TrialLedger:
    def __init__(self, path: str | Path = "artifacts/trials.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=_INITIALIZATION_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._initialize_schema_with_retry()

    def _initialize_schema_with_retry(self) -> None:
        for attempt in range(_INITIALIZATION_ATTEMPTS):
            try:
                self.connection.execute(f"PRAGMA busy_timeout={_INITIALIZATION_BUSY_TIMEOUT_MS}")
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA foreign_keys=ON")
                self._initialize_schema()
                self.connection.execute(f"PRAGMA busy_timeout={_OPERATION_BUSY_TIMEOUT_MS}")
                return
            except sqlite3.OperationalError as exc:
                self._discard_initialization_connection()
                if not _is_sqlite_busy_error(exc) or attempt + 1 == _INITIALIZATION_ATTEMPTS:
                    raise
                time.sleep(min(0.025 * (2**attempt), 0.4))
                self.connection = sqlite3.connect(
                    self.path,
                    timeout=_INITIALIZATION_BUSY_TIMEOUT_MS / 1000,
                    isolation_level=None,
                    check_same_thread=False,
                )
            except Exception:
                self._discard_initialization_connection()
                raise

    def _discard_initialization_connection(self) -> None:
        try:
            if self.connection.in_transaction:
                self.connection.rollback()
        except sqlite3.Error:
            pass
        self.connection.close()

    def _initialize_schema(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS trials (
                logical_trial_id TEXT PRIMARY KEY,
                campaign TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                snapshot_hash TEXT NOT NULL,
                evaluation_hash TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS snapshot_reservations (
                snapshot_hash TEXT PRIMARY KEY,
                holdout_start TEXT NOT NULL,
                holdout_end TEXT NOT NULL,
                boundary_policy_version TEXT NOT NULL,
                eligibility TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                logical_trial_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                error TEXT,
                result_json TEXT,
                FOREIGN KEY(logical_trial_id) REFERENCES trials(logical_trial_id)
            )"""
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS holdout_decisions (
                decision_id TEXT PRIMARY KEY,
                snapshot_hash TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                model_name TEXT,
                verdict TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS holdout_seals (
                snapshot_hash TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                child_run_id TEXT NOT NULL UNIQUE,
                execution_status TEXT NOT NULL,
                owner_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(decision_id) REFERENCES holdout_decisions(decision_id)
            )"""
        )
        self.connection.execute("INSERT OR IGNORE INTO schema_migrations VALUES (1, ?)", (_now(),))
        self.connection.execute("INSERT OR IGNORE INTO schema_migrations VALUES (2, ?)", (_now(),))
        self.connection.execute(
            """CREATE TRIGGER IF NOT EXISTS trials_are_immutable
            BEFORE UPDATE ON trials BEGIN SELECT RAISE(ABORT, 'trial rows are append-only'); END"""
        )
        self.connection.execute(
            """CREATE TRIGGER IF NOT EXISTS attempt_identity_is_immutable
            BEFORE UPDATE OF logical_trial_id, started_at ON attempts
            BEGIN SELECT RAISE(ABORT, 'attempt identity is immutable'); END"""
        )
        self.connection.execute(
            """CREATE TRIGGER IF NOT EXISTS attempt_terminal_is_immutable
            BEFORE UPDATE ON attempts WHEN OLD.status != 'running'
            BEGIN SELECT RAISE(ABORT, 'terminal attempt is immutable'); END"""
        )
        migrated = self.connection.execute("SELECT 1 FROM schema_migrations WHERE version=3").fetchone()
        if migrated is None:
            self._migrate_legacy_holdouts()
            self.connection.execute("INSERT INTO schema_migrations VALUES (3, ?)", (_now(),))
        self.connection.commit()

    def _migrate_legacy_holdouts(self) -> None:
        consumed: dict[str, tuple[str, str, str]] = {}
        rows = self.connection.execute(
            "SELECT t.snapshot_hash, t.settings_json, a.status, a.result_json, a.attempt_id "
            "FROM trials t JOIN attempts a USING(logical_trial_id) ORDER BY a.finished_at, a.started_at"
        ).fetchall()
        for snapshot_hash, settings_json, status, result_json, attempt_id in rows:
            if json.loads(settings_json).get("evaluation") != "holdout":
                continue
            result = json.loads(result_json) if result_json else {}
            consumed.setdefault(
                snapshot_hash,
                (result.get("run_id") or f"legacy-holdout-{attempt_id}", status, "legacy-ledger-attempt"),
            )
        run_root = self.path.parent / "runs"
        if run_root.is_dir():
            for manifest_path in sorted(run_root.glob("*/manifest.json")):
                try:
                    manifest = json.loads(manifest_path.read_text())
                    if not manifest.get("parent_run_id") or not any(
                        item.get("evaluation") == "holdout" for item in manifest.get("candidates", [])
                    ):
                        continue
                    valid = all(
                        (manifest_path.parent / record["path"]).is_file()
                        and hashlib.sha256((manifest_path.parent / record["path"]).read_bytes()).hexdigest() == record["sha256"]
                        for record in manifest.get("artifacts", [])
                    )
                    consumed.setdefault(
                        manifest["snapshot_hash"],
                        (manifest_path.parent.name, "completed" if valid else "interrupted", "legacy-artifact-manifest"),
                    )
                except (KeyError, OSError, ValueError, TypeError):
                    continue
        for snapshot_hash, (child_id, status, source) in consumed.items():
            if self.connection.execute("SELECT 1 FROM holdout_seals WHERE snapshot_hash=?", (snapshot_hash,)).fetchone():
                continue
            decision_id = f"legacy-decision-{hashlib.sha256(snapshot_hash.encode()).hexdigest()[:20]}"
            evidence = json.dumps(
                {"reason": "HISTORIC_HOLDOUT_CONSUMED", "source": source, "execution_status": status},
                sort_keys=True,
                separators=(",", ":"),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO holdout_decisions VALUES (?, ?, 'legacy', NULL, 'REJECTED_HOLDOUT_SEALED', ?, ?)",
                (decision_id, snapshot_hash, evidence, _now()),
            )
            execution_status = (
                "completed" if status == "completed" else "failed" if status in {"failed", "cancelled"} else "interrupted"
            )
            now = _now()
            self.connection.execute(
                "INSERT OR IGNORE INTO holdout_seals VALUES (?, ?, ?, ?, NULL, ?, ?)",
                (snapshot_hash, decision_id, child_id, execution_status, now, now),
            )

    @classmethod
    def read_only(cls, path: str | Path) -> TrialLedger:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Trial ledger does not exist: {path}")
        ledger = cls.__new__(cls)
        ledger.path = Path(path)
        ledger.connection = sqlite3.connect(
            f"file:{ledger.path.resolve()}?mode=ro", uri=True, timeout=30, isolation_level=None, check_same_thread=False
        )
        ledger.connection.execute("PRAGMA foreign_keys=ON")
        ledger.connection.execute("PRAGMA query_only=ON")
        return ledger

    def reserve_snapshot(
        self,
        *,
        snapshot_hash: str,
        holdout_start: str,
        holdout_end: str,
        boundary_policy_version: str = "reserved-bar-boundary.v1",
    ) -> dict[str, Any]:
        """Freeze a snapshot's holdout interval before managed research begins."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT holdout_start, holdout_end, boundary_policy_version, eligibility "
                "FROM snapshot_reservations WHERE snapshot_hash=?",
                (snapshot_hash,),
            ).fetchone()
            if row is not None:
                if row[:3] != (holdout_start, holdout_end, boundary_policy_version):
                    raise ValueError("Snapshot holdout boundary differs from its persisted reservation")
                self.connection.commit()
                return {"holdout_start": row[0], "holdout_end": row[1], "boundary_policy_version": row[2], "eligibility": row[3]}
            historical = self.connection.execute(
                "SELECT 1 FROM trials WHERE snapshot_hash=? LIMIT 1", (snapshot_hash,)
            ).fetchone()
            eligibility = "legacy_boundary_unverified" if historical else "eligible"
            self.connection.execute(
                "INSERT INTO snapshot_reservations VALUES (?, ?, ?, ?, ?, ?)",
                (snapshot_hash, holdout_start, holdout_end, boundary_policy_version, eligibility, _now()),
            )
            self.connection.commit()
            return {
                "holdout_start": holdout_start,
                "holdout_end": holdout_end,
                "boundary_policy_version": boundary_policy_version,
                "eligibility": eligibility,
            }
        except Exception:
            self.connection.rollback()
            raise

    def snapshot_reservation(self, snapshot_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT holdout_start, holdout_end, boundary_policy_version, eligibility, created_at "
            "FROM snapshot_reservations WHERE snapshot_hash=?",
            (snapshot_hash,),
        ).fetchone()
        if row is None:
            return None
        return {
            "holdout_start": row[0],
            "holdout_end": row[1],
            "boundary_policy_version": row[2],
            "eligibility": row[3],
            "created_at": row[4],
        }

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT snapshot_hash, source_run_id, model_name, verdict, evidence_json, created_at "
            "FROM holdout_decisions WHERE decision_id=?",
            (decision_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "decision_id": decision_id,
            "snapshot_hash": row[0],
            "source_run_id": row[1],
            "model_name": row[2],
            "verdict": row[3],
            "evidence": json.loads(row[4]),
            "created_at": row[5],
        }

    def seal(self, snapshot_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT decision_id, child_run_id, execution_status, owner_id, created_at, updated_at "
            "FROM holdout_seals WHERE snapshot_hash=?",
            (snapshot_hash,),
        ).fetchone()
        if row is None:
            return None
        return {
            "snapshot_hash": snapshot_hash,
            "decision_id": row[0],
            "child_run_id": row[1],
            "execution_status": row[2],
            "owner_id": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    def update_seal(self, snapshot_hash: str, *, status: str, owner_id: str | None = None) -> None:
        if status not in {"reserved", "running", "completed", "failed", "interrupted"}:
            raise ValueError(f"Invalid holdout execution status: {status}")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT execution_status, owner_id FROM holdout_seals WHERE snapshot_hash=?", (snapshot_hash,)
            ).fetchone()
            if row is None:
                raise ValueError("No holdout seal exists for snapshot")
            allowed = {
                "reserved": {"running"},
                "running": {"completed", "failed", "interrupted"},
                "completed": set(),
                "failed": set(),
                "interrupted": set(),
            }
            if row[0] == status and row[1] == owner_id:
                self.connection.commit()
                return
            if status not in allowed[row[0]]:
                raise ValueError(f"Invalid holdout seal transition: {row[0]} -> {status}")
            self.connection.execute(
                "UPDATE holdout_seals SET execution_status=?, owner_id=?, updated_at=? WHERE snapshot_hash=?",
                (status, owner_id, _now(), snapshot_hash),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def record_decision(
        self,
        *,
        decision_id: str,
        snapshot_hash: str,
        source_run_id: str,
        model_name: str | None,
        verdict: str,
        evidence: dict[str, Any],
        child_run_id: str | None = None,
        evidence_factory: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if evidence_factory is not None:
                frozen_gate = evidence_factory()
                verdict = frozen_gate["verdict"]
                evidence = frozen_gate["evidence"]
            existing = self.connection.execute(
                "SELECT decision_id FROM holdout_seals WHERE snapshot_hash=?", (snapshot_hash,)
            ).fetchone()
            if existing is not None and verdict == "PASSED":
                verdict = "REJECTED_HOLDOUT_SEALED"
                evidence["reason"] = "HOLDOUT_ALREADY_SEALED"
            payload = json.dumps(evidence, default=str, sort_keys=True, separators=(",", ":"))
            self.connection.execute(
                "INSERT INTO holdout_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (decision_id, snapshot_hash, source_run_id, model_name, verdict, payload, _now()),
            )
            if verdict == "PASSED" and existing is None and child_run_id is not None:
                running = self.connection.execute(
                    "SELECT COUNT(*) FROM attempts a JOIN trials t USING(logical_trial_id) "
                    "WHERE t.snapshot_hash=? AND a.status='running' "
                    "AND COALESCE(json_extract(t.settings_json, '$.evaluation'), 'development') != 'holdout'",
                    (snapshot_hash,),
                ).fetchone()[0]
                if running:
                    verdict = "REJECTED_INSUFFICIENT_EVIDENCE"
                    evidence["reason"] = "TRIALS_IN_PROGRESS"
                    payload = json.dumps(evidence, default=str, sort_keys=True, separators=(",", ":"))
                    self.connection.execute(
                        "UPDATE holdout_decisions SET verdict=?, evidence_json=? WHERE decision_id=?",
                        (verdict, payload, decision_id),
                    )
                else:
                    now = _now()
                    self.connection.execute(
                        "INSERT INTO holdout_seals VALUES (?, ?, ?, 'reserved', NULL, ?, ?)",
                        (snapshot_hash, decision_id, child_run_id, now, now),
                    )
            self.connection.commit()
            return self.decision(decision_id) or {}
        except Exception:
            self.connection.rollback()
            raise

    def register_attempt(
        self,
        *,
        campaign: str,
        config_hash: str,
        snapshot_hash: str,
        evaluation_hash: str,
        settings: dict[str, Any],
        logical_trial_id: str | None = None,
    ) -> tuple[str, str, bool]:
        logical = logical_trial_id or self._logical_id(campaign, config_hash, snapshot_hash, evaluation_hash, settings)
        attempt = f"attempt-{uuid.uuid4().hex}"
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if (
                settings.get("evaluation") != "holdout"
                and self.connection.execute("SELECT 1 FROM holdout_seals WHERE snapshot_hash=?", (snapshot_hash,)).fetchone()
            ):
                raise ValueError("Snapshot is sealed against additional development registrations")
            self.connection.execute(
                "INSERT OR IGNORE INTO trials VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    logical,
                    campaign,
                    config_hash,
                    snapshot_hash,
                    evaluation_hash,
                    json.dumps(settings, default=str, sort_keys=True),
                    _now(),
                ),
            )
            self.connection.execute(
                "INSERT INTO attempts VALUES (?, ?, 'running', ?, NULL, NULL, NULL)",
                (attempt, logical, _now()),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return logical, attempt, self._attempt_count(logical) > 1

    def update_attempt(
        self, attempt_id: str, *, status: str, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        if status not in {"running", "completed", "failed", "cancelled"}:
            raise ValueError(f"Invalid attempt status: {status}")
        result_json = json.dumps(result, default=str, sort_keys=True) if result is not None else None
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT status, error, result_json FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown attempt: {attempt_id}")
            if row[0] != "running":
                if row == (status, error, result_json):
                    self.connection.commit()
                    return
                raise ValueError(f"Attempt {attempt_id} already has terminal status {row[0]}")
            if status == "running":
                self.connection.commit()
                return
            self.connection.execute(
                "UPDATE attempts SET status=?, finished_at=?, error=?, result_json=? WHERE attempt_id=? AND status='running'",
                (status, _now(), error, result_json, attempt_id),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def trials(self, campaign: str | None = None, *, snapshot_hash: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT logical_trial_id, campaign, config_hash, snapshot_hash, evaluation_hash, settings_json, created_at "
            "FROM trials"
        )
        params: tuple[Any, ...] = ()
        if campaign is not None:
            query += " WHERE campaign=?"
            params = (campaign,)
        if snapshot_hash is not None:
            query += " WHERE snapshot_hash=?" if campaign is None else " AND snapshot_hash=?"
            params = (*params, snapshot_hash)
        rows = self.connection.execute(query + " ORDER BY created_at", params).fetchall()
        return [
            {
                "logical_trial_id": row[0],
                "campaign": row[1],
                "config_hash": row[2],
                "snapshot_hash": row[3],
                "evaluation_hash": row[4],
                "settings": json.loads(row[5]),
                "created_at": row[6],
            }
            for row in rows
        ]

    def attempts(self, logical_trial_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT attempt_id, logical_trial_id, status, started_at, finished_at, error, result_json FROM attempts"
        params: tuple[Any, ...] = ()
        if logical_trial_id is not None:
            query += " WHERE logical_trial_id=?"
            params = (logical_trial_id,)
        rows = self.connection.execute(query + " ORDER BY started_at", params).fetchall()
        return [
            {
                "attempt_id": row[0],
                "logical_trial_id": row[1],
                "status": row[2],
                "started_at": row[3],
                "finished_at": row[4],
                "error": row[5],
                "result": json.loads(row[6]) if row[6] else None,
            }
            for row in rows
        ]

    def comparable_evidence(
        self,
        *,
        campaign: str | None = None,
        snapshot_hash: str,
        evaluation_hash: str,
        frequency: str,
        cost_treatment: str | None = None,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return comparable logical trials and retain attempts for exclusion reporting."""
        query = (
            "SELECT logical_trial_id, campaign, config_hash, snapshot_hash, evaluation_hash, settings_json, created_at "
            "FROM trials WHERE snapshot_hash=? AND evaluation_hash=?"
        )
        params: list[Any] = [snapshot_hash, evaluation_hash]
        if campaign is not None:
            query += " AND campaign=?"
            params.append(campaign)
        rows = self.connection.execute(query + " ORDER BY created_at", tuple(params)).fetchall()
        evidence = []
        for row in rows:
            settings = json.loads(row[5])
            if settings.get("frequency", frequency) != frequency:
                continue
            if cost_treatment is not None and settings.get("cost_treatment") != cost_treatment:
                continue
            if task is not None and settings.get("task") != task:
                continue
            attempts = self.attempts(row[0])
            evidence.append(
                {
                    "logical_trial_id": row[0],
                    "campaign": row[1],
                    "config_hash": row[2],
                    "snapshot_hash": row[3],
                    "evaluation_hash": row[4],
                    "settings": settings,
                    "created_at": row[6],
                    "attempts": attempts,
                }
            )
        return evidence

    def comparable_return_series(
        self,
        *,
        campaign: str | None = None,
        snapshot_hash: str,
        evaluation_hash: str,
        frequency: str,
        cost_treatment: str | None = None,
        task: str | None = None,
        return_convention: str | None = None,
        execution_evidence_version: str | None = None,
    ) -> dict[str, Any]:
        """Select one verified successful attempt per logical trial."""
        evidence = self.comparable_evidence(
            campaign=campaign,
            snapshot_hash=snapshot_hash,
            evaluation_hash=evaluation_hash,
            frequency=frequency,
            cost_treatment=cost_treatment,
            task=task,
        )
        complete: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for trial in evidence:
            if (
                execution_evidence_version is not None
                and trial["settings"].get("execution_evidence_version") != execution_evidence_version
            ):
                exclusions.extend(
                    {
                        "logical_trial_id": trial["logical_trial_id"],
                        "attempt_id": attempt["attempt_id"],
                        "status": attempt["status"],
                        "reason": "execution evidence version is missing or incompatible",
                    }
                    for attempt in trial["attempts"]
                )
                if not trial["attempts"]:
                    exclusions.append(
                        {
                            "logical_trial_id": trial["logical_trial_id"],
                            "attempt_id": None,
                            "status": "missing",
                            "reason": "execution evidence version is missing or incompatible",
                        }
                    )
                continue
            if return_convention is not None and trial["settings"].get("return_convention") != return_convention:
                for attempt in trial["attempts"] or [None]:
                    exclusions.append(
                        {
                            "logical_trial_id": trial["logical_trial_id"],
                            "attempt_id": attempt["attempt_id"] if attempt else None,
                            "status": attempt["status"] if attempt else "missing",
                            "reason": (
                                f"trial settings return_convention {trial['settings'].get('return_convention')!r} "
                                f"does not match requested {return_convention!r}"
                            ),
                        }
                    )
                continue
            successful = [
                attempt
                for attempt in trial["attempts"]
                if attempt["status"] == "completed"
                and (attempt.get("result") or {}).get("development_net_returns")
                and (return_convention is None or (attempt.get("result") or {}).get("return_convention") == return_convention)
            ]
            selected = min(successful, key=lambda item: item.get("finished_at") or item["started_at"]) if successful else None
            if selected is not None:
                complete.append(
                    {
                        "logical_trial_id": trial["logical_trial_id"],
                        "attempt_id": selected["attempt_id"],
                        "run_id": selected["result"].get("run_id"),
                        "returns": selected["result"]["development_net_returns"],
                    }
                )
            for attempt in trial["attempts"]:
                if selected is not None and attempt["attempt_id"] == selected["attempt_id"]:
                    continue
                result = attempt.get("result") or {}
                has_returns = bool(result.get("development_net_returns"))
                if attempt["status"] != "completed":
                    reason = attempt.get("error") or "missing returns"
                elif return_convention is not None and result.get("return_convention") != return_convention:
                    actual_convention = result.get("return_convention")
                    reason = (
                        f"attempt result return_convention {actual_convention!r} does not match requested {return_convention!r}"
                    )
                elif not has_returns:
                    reason = "missing returns"
                else:
                    reason = "duplicate successful attempt"
                exclusions.append(
                    {
                        "logical_trial_id": trial["logical_trial_id"],
                        "attempt_id": attempt["attempt_id"],
                        "status": attempt["status"],
                        "reason": reason,
                    }
                )
        return {"complete": complete, "exclusions": exclusions, "population": len(evidence)}

    def candidate_search_evidence(
        self,
        *,
        snapshot_hash: str,
        evaluation_hash: str,
        frequency: str,
        task: str,
        return_convention: str,
        min_observations: int,
        calendar: str | None = None,
        backtest_policy: dict[str, Any] | None = None,
        target_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Count distinct searched challengers and retain one comparable result each."""
        keys: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for trial in self.trials(snapshot_hash=snapshot_hash):
            settings = trial["settings"]
            if settings.get("evaluation") == "holdout":
                continue
            models = settings.get("models", {})
            if settings.get("parent_prediction_run"):
                names = [settings["model"]] if settings.get("model") not in (None, "baseline") else []
            elif isinstance(models, dict):
                names = [name for name in ("xgboost", "gru", "lstm") if models.get(name, {}).get("enabled")]
            else:
                names = []
            for name in names:
                context = settings.get("candidate_context")
                identity = (
                    [
                        snapshot_hash,
                        trial["evaluation_hash"],
                        settings.get("task"),
                        context,
                        name,
                        (settings.get("model_spec") if settings.get("parent_prediction_run") else models.get(name)),
                        (
                            settings.get("model_global_settings")
                            if settings.get("parent_prediction_run")
                            else {
                                field: value
                                for field, value in models.items()
                                if field not in {"baseline", "xgboost", "gru", "lstm"}
                            }
                        ),
                    ]
                    if context is not None
                    else [trial["logical_trial_id"], name]
                )
                key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
                keys.setdefault(key, []).append((trial, name))
        complete: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        reference_grid: list[str] | None = None
        for key, duplicates in keys.items():
            trial, name = duplicates[0]
            settings = trial["settings"]
            reason = None
            if (
                trial["evaluation_hash"] != evaluation_hash
                or settings.get("task") != task
                or settings.get("frequency") != frequency
                or settings.get("return_convention") != return_convention
                or settings.get("calendar") is None
                or (calendar is not None and settings.get("calendar") != calendar)
                or (backtest_policy is not None and (settings.get("candidate_context") or {}).get("backtest") != backtest_policy)
                or (target_policy is not None and (settings.get("candidate_context") or {}).get("task") != target_policy)
                or settings.get("cost_treatment") != "fees+slippage"
                or settings.get("execution_evidence_version") != "period-bounded-next-open.v2"
            ):
                reason = "incompatible candidate evidence"
            selected = None
            if reason is None:
                for source_trial, attempt in sorted(
                    (
                        (source_trial, attempt)
                        for source_trial, _ in duplicates
                        for attempt in self.attempts(source_trial["logical_trial_id"])
                    ),
                    key=lambda item: (item[1]["finished_at"] or item[1]["started_at"], item[1]["attempt_id"]),
                ):
                    if attempt["status"] != "completed":
                        continue
                    candidate = (attempt["result"] or {}).get("challenger_returns", {}).get(name, {})
                    values = candidate.get("returns")
                    grid = candidate.get("timestamps")
                    if (
                        not isinstance(values, list)
                        or len(values) < min_observations
                        or not isinstance(grid, list)
                        or len(grid) != len(values)
                    ):
                        continue
                    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                        continue
                    if len(set(values)) < 2:
                        continue
                    selected = {
                        "candidate_key": key,
                        "logical_trial_id": source_trial["logical_trial_id"],
                        "attempt_id": attempt["attempt_id"],
                        "run_id": (attempt["result"] or {}).get("run_id"),
                        "model": name,
                        "returns": values,
                        "timestamps": grid,
                    }
                    break
                if selected is None:
                    reason = "missing usable challenger returns"
                elif reference_grid is not None and selected["timestamps"] != reference_grid:
                    reason = "incompatible statistical timestamp grid"
                else:
                    reference_grid = selected["timestamps"]
            if reason is not None:
                exclusions.append({"candidate_key": key, "model": name, "reason": reason})
            elif selected is not None:
                complete.append(selected)
        return {"K": len(keys), "complete": complete, "exclusions": exclusions, "candidate_keys": list(keys)}

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _logical_id(campaign: str, config_hash: str, snapshot_hash: str, evaluation_hash: str, settings: dict[str, Any]) -> str:
        payload = json.dumps([campaign, config_hash, snapshot_hash, evaluation_hash, settings], default=str, sort_keys=True)
        return f"trial-{uuid.uuid5(uuid.NAMESPACE_URL, payload).hex}"

    def _attempt_count(self, logical_trial_id: str) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM attempts WHERE logical_trial_id=?", (logical_trial_id,)).fetchone()[0]
        )
