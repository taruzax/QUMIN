import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import lab.research.reporting as reporting
from lab.core.config import PipelineConfig
from lab.core.contracts import MetricResult
from lab.platform.artifacts import sha256_file
from lab.research.reporting import compound_session_returns, equity_statistics, generate_demo_report, generate_local_report, generate_run_report, write_demo_locator
from lab.platform.report_store import ReportStore


def test_reporting_compounds_returns_and_writes_local_html(tmp_path):
    timestamps = ["2024-01-02T15:00:00Z", "2024-01-02T16:00:00Z", "2024-01-03T15:00:00Z"]
    daily = compound_session_returns(timestamps, [0.01, 0.02, -0.01])
    report = generate_local_report(tmp_path / "report.html", net_returns=np.array([0.01, -0.01]), gross_returns=np.array([0.02, -0.005]), frequency="native")

    assert len(daily) == 2
    assert daily.iloc[0] == pytest.approx(0.01 * 1.02 + 0.02)
    assert report.exists()
    assert equity_statistics([0.01, -0.01])["status"] == "available"


def test_local_daily_report_requires_timestamps_and_quantstats_receives_session_returns(tmp_path, monkeypatch):
    calls = {}

    class FakeReports:
        @staticmethod
        def html(returns, *, output, title, compounded):
            calls["returns"] = returns.copy()
            calls["compounded"] = compounded
            Path(output).write_text("<html><body>quantstats</body></html>")

    monkeypatch.setattr(reporting, "qs", SimpleNamespace(reports=FakeReports))
    with pytest.raises(ValueError, match="timestamps are required"):
        generate_local_report(tmp_path / "missing-dates.html", net_returns=[0.01], frequency="daily")

    report = generate_local_report(
        tmp_path / "daily.html",
        net_returns=[0.01, 0.02, -0.01],
        gross_returns=[0.02, 0.03, -0.02],
        timestamps=["2024-01-02T15:00:00Z", "2024-01-02T16:00:00Z", "2024-01-03T15:00:00Z"],
        frequency="daily",
        calendar="XNYS",
    )

    assert len(calls["returns"]) == 2
    assert calls["returns"].iloc[0] == pytest.approx(1.01 * 1.02 - 1.0)
    assert calls["compounded"] is False
    assert report.exists()


def test_quantstats_always_receives_session_daily_returns_when_configured_native(tmp_path, monkeypatch):
    seen = {}

    class FakeReports:
        @staticmethod
        def html(returns, *, output, title, compounded):
            seen["returns"] = returns.copy()
            Path(output).write_text("<html><body>quantstats</body></html>")

    monkeypatch.setattr(reporting, "qs", SimpleNamespace(reports=FakeReports))
    generate_local_report(
        tmp_path / "native-config.html",
        net_returns=[0.01, 0.02, 0.03, -0.01],
        gross_returns=[0.02, 0.03, 0.04, -0.02],
        timestamps=[
            "2024-01-02T15:00:00Z", "2024-01-02T16:00:00Z",
            "2024-01-03T15:00:00Z", "2024-01-03T16:00:00Z",
        ],
        frequency="native",
    )

    assert len(seen["returns"]) == 2
    assert seen["returns"].iloc[0] == pytest.approx(1.01 * 1.02 - 1.0)


def _fake_run(run_id, *, all_cash):
    timestamps = ["2024-01-02T15:00:00Z", "2024-01-03T15:00:00Z"]
    orders = pl.DataFrame({
        "timestamp": [timestamps[0]] if not all_cash else [],
        "ticker": ["ALFA"] if not all_cash else [],
        "quantity": [1.0] if not all_cash else [],
        "price": [100.0] if not all_cash else [],
        "side": ["buy"] if not all_cash else [],
        "reason": ["entry"] if not all_cash else [],
        "fees": [0.5] if not all_cash else [],
    })
    equity = pl.DataFrame({"timestamp": timestamps, "equity": [100.0, 100.0 if all_cash else 101.0], "gross_equity": [100.0, 100.0 if all_cash else 101.5]})
    backtest = SimpleNamespace(
        orders=orders,
        equity=equity,
        net_returns=pl.Series("net_returns", [0.0, 0.0 if all_cash else 0.01]),
        gross_returns=pl.Series("gross_returns", [0.0, 0.0 if all_cash else 0.015]),
    )
    return SimpleNamespace(
        ref=SimpleNamespace(run_id=run_id, snapshot_hash="snapshot-hash", config_hash="config-hash", split_hash="split-hash", parent_run_id=None),
        predictions=(),
        backtest=backtest,
        metrics=(MetricResult(name="sharpe", value=None, status="unavailable", reason="constant returns"),),
        snapshot=SimpleNamespace(timeframe=SimpleNamespace(value="1d"), calendar="XNYS", provenance={"source": "bundled_demo", "label": "<synthetic>"}),
        artifact_manifest={"candidates": [{"model": "baseline", "status": "completed"}], "artifacts": [{"path": "metrics.json"}]},
    )


def test_demo_report_and_locator_keep_all_cash_honest_and_manifest_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "qs", None)
    development = _fake_run("dev-1", all_cash=True)
    development.artifact_manifest["candidates"].append({"selected_model": "baseline"})
    holdout = _fake_run("holdout-1", all_cash=False)
    original_manifest = repr(development.artifact_manifest)
    config = PipelineConfig()
    report = generate_demo_report(tmp_path / "dev-1.html", development=development, holdout=holdout, config=config, selected_model="baseline")
    locator = write_demo_locator(tmp_path / "dev-1.locator.json", development=development, holdout=holdout, report_path=report, selected_model="baseline", artifacts_root=tmp_path / "artifacts")
    content = report.read_text()

    assert report.exists()
    assert locator.exists()
    assert "All-cash result" in content
    assert "2024-01-02" in content
    assert "Costs paid" in content
    assert "0.5" in content
    assert "unavailable" in content
    assert "Recomputed statistical metrics using session-compounded.v1" in content
    assert "Saved statistical metrics as recorded" in content
    assert "0 probability" not in content
    assert repr(development.artifact_manifest) == original_manifest
    locator_content = json.loads(locator.read_text())
    assert locator_content["development"]["run_id"] == "dev-1"
    assert locator_content["holdout"]["run_id"] == "holdout-1"
    assert locator_content["report_path"] == str(report.resolve())


def test_report_locator_is_idempotent_and_verifies_all_references(tmp_path):
    report = tmp_path / "report.html"
    manifest = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.json"
    report.write_text("<html>immutable</html>")
    manifest.write_text('{"status":"completed"}')
    evidence.write_text('{"metric_values":{"dsr":0.5}}')
    store = ReportStore(tmp_path / "artifacts")
    evidence_ref = {
        "evidence_id": "evidence-test",
        "path": str(evidence),
        "sha256": sha256_file(evidence),
    }

    first = store.write_locator(
        report_path=report, run_id="run-1", manifest_path=manifest, evidence=evidence_ref
    )
    second = store.write_locator(
        report_path=report, run_id="run-1", manifest_path=manifest, evidence=evidence_ref
    )

    assert first["report_id"] == second["report_id"]
    assert store.load_locator(first["report_id"])["source_run_id"] == "run-1"
    report.write_text("<html>changed</html>")
    with pytest.raises(ValueError, match="Report checksum"):
        store.load_locator(first["report_id"])


def test_generic_report_includes_identity_costs_evidence_and_escapes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "qs", None)
    result = _fake_run("run-generic", all_cash=False)
    result.backtest.turnover = pl.Series("turnover", [2.0, 3.0])
    result.snapshot.provenance = {"source": "<provider>"}
    path = generate_run_report(
        tmp_path / "generic.html", result=result, config=PipelineConfig(),
        evidence={"excluded": "<script>alert(1)</script>"},
    )
    content = path.read_text()

    assert "snapshot-hash" in content and "config-hash" in content and "split-hash" in content
    assert "Configured fees (bps)" in content and "Configured slippage (bps)" in content
    assert "Absolute quantity turnover" in content
    assert "&lt;provider&gt;" in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content


def test_period_summary_recomputes_requested_returns_with_snapshot_calendar(monkeypatch):
    result = _fake_run("unit-check", all_cash=False)
    hourly_timestamps = [
        "2024-07-03T15:00:00Z",
        "2024-07-03T16:00:00Z",
        "2024-07-04T15:00:00Z",
    ]
    result.backtest.equity = pl.DataFrame({
        "timestamp": hourly_timestamps,
        "equity": [100.0, 101.0, 103.0],
        "gross_equity": [100.0, 102.0, 105.0],
    })
    result.backtest.net_returns = pl.Series("net_returns", [0.01, 0.02, 0.03])
    result.backtest.gross_returns = pl.Series("gross_returns", [0.02, 0.03, 0.04])
    result.snapshot.calendar = "XHKG"
    original = reporting.statistical_returns
    calls = []

    def capture(timestamps, returns, *, frequency, calendar):
        calls.append((frequency, calendar))
        return original(timestamps, returns, frequency=frequency, calendar=calendar)

    monkeypatch.setattr(reporting, "statistical_returns", capture)
    summary = reporting._period_summary(
        "Development", result, frequency="daily", min_observations=30,
        fees_bps=0.0, slippage_bps=0.0,
    )

    assert calls == [("daily", "XHKG"), ("daily", "XHKG"), ("daily", "XHKG")]
    assert summary["native_net"]["observations"] == 3
    assert summary["session_net"]["frequency"] == "daily"
    assert summary["requested_net"]["observations"] == 2
    assert summary["return_convention"] == "session-compounded.v1"
    assert summary["saved_statistical_metrics"] == result.metrics
    assert all(metric.inputs.get("recomputed_from_saved_returns") for metric in summary["recomputed_statistical_metrics"])
