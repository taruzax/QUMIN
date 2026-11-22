from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.core.config import PipelineConfig
from lab.core.contracts import RunResult
from lab.quant.metrics import (
    return_convention,
    sharpe_evidence_metrics,
    sharpe_statistics,
    statistical_returns,
)

try:
    import quantstats as qs
except ImportError:  # pragma: no cover - optional fallback is covered through the public function
    qs = None


def equity_statistics(returns: Any, *, frequency: str = "daily", min_observations: int = 30) -> dict[str, Any]:
    """Return explicit total return, drawdown and descriptive Sharpe statistics."""
    values = np.asarray(returns, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return {"status": "unavailable", "reason": "empty or non-finite returns"}
    equity = np.cumprod(1.0 + values)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    result: dict[str, Any] = {
        "status": "available",
        "total_return": float(equity[-1] - 1.0),
        "maximum_drawdown": float(drawdown.min()),
        "observations": int(values.size),
        "frequency": frequency,
    }
    try:
        result.update(sharpe_statistics(values, frequency=frequency, min_observations=min_observations))
    except ValueError as exc:
        result["sharpe_status"] = "unavailable"
        result["sharpe_reason"] = str(exc)
    return result


def _quantstats_report(net: np.ndarray, index: pd.Index, title: str) -> tuple[str | None, str | None]:
    """Render QuantStats to a temporary file and return its local HTML."""
    temporary_path: Path | None = None
    try:
        if qs is None:
            raise ImportError("quantstats is not installed")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as handle:
            temporary_path = Path(handle.name)
        qs.reports.html(pd.Series(net, index=index), output=str(temporary_path), title=title, compounded=False)
        if not temporary_path.exists():
            raise RuntimeError("QuantStats did not create an HTML report")
        return temporary_path.read_text(), None
    except Exception as exc:
        return None, str(exc)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def generate_local_report(
    path: str | Path,
    *,
    net_returns: Any,
    gross_returns: Any | None = None,
    timestamps: Any | None = None,
    title: str = "Qumin report",
    frequency: str = "daily",
    calendar: str = "XNYS",
    summary_html: str | None = None,
) -> Path:
    """Write an offline HTML report from explicit local return series."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    net = np.asarray(net_returns, dtype=float).reshape(-1)
    gross = None if gross_returns is None else np.asarray(gross_returns, dtype=float).reshape(-1)
    if timestamps is None:
        if frequency == "daily":
            raise ValueError("timestamps are required for daily session-compounded reports")
        index: pd.Index = pd.RangeIndex(len(net), name="observation")
        statistical_net = pd.Series(net, index=index, name="native_return")
        statistical_gross = None if gross is None else pd.Series(gross, index=index, name="native_return")
        quantstats_net = None
    else:
        statistical_net = statistical_returns(timestamps, net, frequency=frequency, calendar=calendar)
        statistical_gross = (
            statistical_returns(timestamps, gross, frequency=frequency, calendar=calendar) if gross is not None else None
        )
        index = statistical_net.index
        # QuantStats uses daily conventions; keep its input independent of the
        # requested native-bar diagnostic convention.
        quantstats_net = statistical_returns(timestamps, net, frequency="daily", calendar=calendar)
    statistics = equity_statistics(statistical_net.to_numpy(), frequency=frequency)
    if quantstats_net is None:
        quantstats_html, quantstats_error = None, "timestamps are required for a daily QuantStats report"
    elif not len(quantstats_net):
        quantstats_html, quantstats_error = None, "empty return series"
    else:
        quantstats_html, quantstats_error = _quantstats_report(quantstats_net.to_numpy(), quantstats_net.index, title)
    if quantstats_html is not None:
        body = quantstats_html
        if summary_html:
            insertion = body.lower().rfind("</body>")
            body = body[:insertion] + summary_html + body[insertion:] if insertion >= 0 else summary_html + body
        output.write_text(body)
        return output
    gross_summary = (
        equity_statistics(statistical_gross.to_numpy(), frequency=frequency) if statistical_gross is not None else None
    )
    fallback_summary = summary_html or ""
    output.write_text(
        "<html><head><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title></head><body>"
        + f"<h1>{html.escape(title)}</h1>"
        + f"<p>QuantStats unavailable or unsupported for this input: "
        f"{html.escape(quantstats_error or 'no report renderer')}</p>"
        + fallback_summary
        + f"<h2>Net returns</h2><pre>{html.escape(str(statistics))}</pre>"
        + f"<h2>Gross returns</h2><pre>{html.escape(str(gross_summary))}</pre>"
        + "</body></html>"
    )
    return output


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def _metric_rows(metrics: Any) -> str:
    rows = []
    for metric in metrics:
        if metric.status == "available":
            value = _format_value(metric.value)
            detail = f"{value} ({html.escape(metric.status)})"
        else:
            detail = f"unavailable — {html.escape(metric.reason or 'no reason recorded')}"
        rows.append(f"<tr><th>{html.escape(metric.name)}</th><td>{detail}</td></tr>")
    return "".join(rows) or "<tr><td colspan='2'>No predictive metrics were saved.</td></tr>"


def _period_summary(
    label: str,
    result: RunResult,
    *,
    frequency: str,
    min_observations: int,
    fees_bps: float,
    slippage_bps: float,
    calendar: str | None = None,
) -> dict[str, Any]:
    backtest = result.backtest
    prediction_count = sum(len(frame.keys) for frame in result.predictions)
    if calendar is None:
        calendar = getattr(result.snapshot, "calendar", None) or "XNYS"
    if backtest is None or backtest.net_returns is None:
        return {
            "label": label,
            "run_id": result.ref.run_id,
            "prediction_count": prediction_count,
            "filled_trades": 0,
            "status": "unavailable",
            "reason": "backtest or net return series was not saved",
            "metrics": result.metrics,
        }
    timestamps = backtest.equity.get_column("timestamp").to_list()
    net = backtest.net_returns.to_numpy()
    gross = backtest.gross_returns.to_numpy() if backtest.gross_returns is not None else None
    orders = backtest.orders
    costs = float(orders.get_column("fees").sum()) if "fees" in orders.columns and orders.height else 0.0
    total_cost_bps = fees_bps + slippage_bps
    fee_cost = costs * fees_bps / total_cost_bps if total_cost_bps else 0.0
    slippage_cost = costs * slippage_bps / total_cost_bps if total_cost_bps else 0.0
    native_net = equity_statistics(net, frequency="native")
    session_net = statistical_returns(timestamps, net, frequency="daily", calendar=calendar)
    session_summary = equity_statistics(session_net.to_numpy(), frequency="daily")
    gross_summary = equity_statistics(gross, frequency="native") if gross is not None else None
    statistical_net = statistical_returns(timestamps, net, frequency=frequency, calendar=calendar)
    statistical_gross = (
        statistical_returns(timestamps, gross, frequency=frequency, calendar=calendar) if gross is not None else None
    )
    saved_statistical_metrics = tuple(metric for metric in result.metrics if metric.name in {"sharpe", "psr", "dsr"})
    predictive_metrics = [metric for metric in result.metrics if metric.name not in {"sharpe", "psr", "dsr"}]
    recomputed_metrics = list(
        sharpe_evidence_metrics(
            statistical_net.to_numpy(),
            frequency=frequency,
            min_observations=min_observations,
            trial_sharpes=[],
        ).values()
    )
    convention = return_convention(frequency)
    for metric in recomputed_metrics:
        metric.inputs.update({"return_convention": convention, "calendar": calendar, "recomputed_from_saved_returns": True})
        metric.conventions.update({"return_convention": convention, "calendar": calendar})
    return {
        "label": label,
        "run_id": result.ref.run_id,
        "prediction_count": prediction_count,
        "filled_trades": orders.height,
        "costs": costs,
        "fee_cost": fee_cost,
        "slippage_cost": slippage_cost,
        "turnover": float(backtest.turnover.sum()) if getattr(backtest, "turnover", None) is not None else "unavailable",
        "native_net": native_net,
        "requested_net": equity_statistics(statistical_net.to_numpy(), frequency=frequency),
        "session_net": session_summary,
        "native_gross": gross_summary,
        "requested_gross": equity_statistics(statistical_gross.to_numpy(), frequency=frequency)
        if statistical_gross is not None
        else None,
        "net_final_equity": float(backtest.equity.get_column("equity")[-1]),
        "gross_final_equity": (
            float(backtest.equity.get_column("gross_equity")[-1]) if "gross_equity" in backtest.equity.columns else "unavailable"
        ),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "all_cash": orders.height == 0,
        "metrics": tuple(predictive_metrics),
        "recomputed_statistical_metrics": tuple(recomputed_metrics),
        "saved_statistical_metrics": saved_statistical_metrics,
        "return_convention": convention,
        "calendar": calendar,
        "status": "available",
    }


def _run_metadata(
    label: str,
    result: RunResult,
    *,
    frequency: str,
    min_observations: int,
    fees_bps: float,
    slippage_bps: float,
    calendar: str = "XNYS",
) -> str:
    summary = _period_summary(
        label,
        result,
        frequency=frequency,
        min_observations=min_observations,
        fees_bps=fees_bps,
        slippage_bps=slippage_bps,
        calendar=calendar,
    )
    rows = [
        ("Run ID", summary["run_id"]),
        ("Predictions", summary["prediction_count"]),
        ("Filled trades", summary.get("filled_trades", 0)),
        ("First saved timestamp", summary.get("first_timestamp", "unavailable")),
        ("Last saved timestamp", summary.get("last_timestamp", "unavailable")),
        ("Costs paid", summary.get("costs", "unavailable")),
        ("Fees paid (allocated from combined transaction costs)", summary.get("fee_cost", "unavailable")),
        ("Slippage paid (allocated from combined transaction costs)", summary.get("slippage_cost", "unavailable")),
        ("Configured fees (bps)", fees_bps),
        ("Configured slippage (bps)", slippage_bps),
        ("Absolute quantity turnover", summary.get("turnover", "unavailable")),
        ("Native net return", summary.get("native_net", {}).get("total_return", "unavailable")),
        (
            f"Configured {frequency} net return ({summary.get('return_convention', 'unknown')})",
            summary.get("requested_net", {}).get("total_return", "unavailable"),
        ),
        ("Session-daily net return", summary.get("session_net", {}).get("total_return", "unavailable")),
        ("Native gross return", (summary.get("native_gross") or {}).get("total_return", "unavailable")),
        (
            f"Configured {frequency} gross return ({summary.get('return_convention', 'unknown')})",
            (summary.get("requested_gross") or {}).get("total_return", "unavailable"),
        ),
        ("Net final equity", summary.get("net_final_equity", "unavailable")),
        ("Gross final equity", summary.get("gross_final_equity", "unavailable")),
        ("Native maximum drawdown", summary.get("native_net", {}).get("maximum_drawdown", "unavailable")),
    ]
    rows_html = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_format_value(value))}</td></tr>" for key, value in rows
    )
    metrics_html = _metric_rows(summary["metrics"])
    recomputed_html = _metric_rows(summary.get("recomputed_statistical_metrics", ()))
    saved_statistical_html = _metric_rows(summary.get("saved_statistical_metrics", ()))
    if summary.get("all_cash"):
        cash_note = (
            "<p><strong>All-cash result:</strong> no orders were filled, so the saved equity stays at initial "
            "capital and cost totals are zero.</p>"
        )
    elif summary["status"] != "available":
        cash_note = f"<p>Result unavailable: {html.escape(summary.get('reason', 'unknown reason'))}</p>"
    else:
        cash_note = ""
    return (
        f"<section><h2>{html.escape(label)}</h2>"
        f"<table><tbody>{rows_html}</tbody></table>{cash_note}"
        f"<h3>Predictive metrics as saved</h3><table><tbody>{metrics_html}</tbody></table>"
        f"<h3>Recomputed statistical metrics using {html.escape(summary.get('return_convention', 'unknown'))}</h3>"
        f"<table><tbody>{recomputed_html}</tbody></table>"
        "<h3>Saved statistical metrics as recorded</h3>"
        "<p>Historical saved values are preserved; these may use legacy return units.</p>"
        f"<table><tbody>{saved_statistical_html}</tbody></table></section>"
    )


def _provenance_html(result: RunResult) -> str:
    snapshot = result.snapshot
    if snapshot is None:
        return "<p>Dataset provenance unavailable.</p>"
    provenance = {"timeframe": snapshot.timeframe.value, **snapshot.provenance}
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_format_value(value))}</td></tr>"
        for key, value in sorted(provenance.items())
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _result_calendar(result: RunResult, config: PipelineConfig) -> str:
    """Prefer the immutable source snapshot calendar, then configured data calendar."""
    if result.snapshot is not None and result.snapshot.calendar:
        return result.snapshot.calendar
    return config.data.calendar


def generate_demo_report(
    path: str | Path,
    *,
    development: RunResult,
    holdout: RunResult,
    config: PipelineConfig,
    selected_model: str,
) -> Path:
    """Write one local report that separates development from synthetic holdout."""
    if development.backtest is None or development.backtest.net_returns is None:
        raise RuntimeError("Development result has no saved net return series")
    if holdout.backtest is None or holdout.backtest.net_returns is None:
        raise RuntimeError("Holdout result has no saved net return series")
    candidates = development.artifact_manifest.get("candidates", [])
    selected_status = next(
        (item for item in candidates if item.get("model") == selected_model and item.get("status") == "completed"),
        None,
    )
    if selected_status is None:
        raise RuntimeError(f"Selected model is missing from the development manifest: {selected_model}")
    summary_html = (
        "<style>body{font-family:system-ui,sans-serif;line-height:1.4;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;margin:0 0 1.5rem;min-width:28rem}"
        "th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;vertical-align:top}"
        "th{background:#f3f3f3}section{border-top:3px solid #555;margin-top:2rem;padding-top:1rem}"
        "code{word-break:break-all}</style>"
        "<h1>Qumin offline demo</h1>"
        "<p>This report is generated only from bundled synthetic data and saved local run artifacts. "
        "It is not market-performance evidence.</p>"
        f"<h2>Dataset provenance and timeframe</h2>{_provenance_html(development)}"
        f"<p>Selected model: <strong>{html.escape(selected_model)}</strong>; candidate status: "
        f"<code>{html.escape(_format_value(selected_status))}</code>.</p>"
        + (
            f"<p>Saved execution timeframe: <strong>{html.escape(development.snapshot.timeframe.value)}</strong>; "
            f"calendar: <strong>{html.escape(_result_calendar(development, config))}</strong>.</p>"
            if development.snapshot is not None
            else ""
        )
        + _run_metadata(
            "Development",
            development,
            frequency=config.statistics.frequency,
            min_observations=config.statistics.min_observations,
            fees_bps=config.backtest.fees_bps,
            slippage_bps=config.backtest.slippage_bps,
            calendar=_result_calendar(development, config),
        )
        + _run_metadata(
            "Synthetic holdout",
            holdout,
            frequency=config.statistics.frequency,
            min_observations=config.statistics.min_observations,
            fees_bps=config.backtest.fees_bps,
            slippage_bps=config.backtest.slippage_bps,
            calendar=_result_calendar(holdout, config),
        )
    )
    timestamps = development.backtest.equity.get_column("timestamp").to_list()
    return generate_local_report(
        path,
        net_returns=development.backtest.net_returns.to_numpy(),
        gross_returns=development.backtest.gross_returns.to_numpy() if development.backtest.gross_returns is not None else None,
        timestamps=timestamps,
        title="Qumin offline demo",
        frequency=config.statistics.frequency,
        calendar=_result_calendar(development, config),
        summary_html=summary_html,
    )


def generate_run_report(
    path: str | Path,
    *,
    result: RunResult,
    config: PipelineConfig,
    evidence: dict[str, Any] | None = None,
    report_type: str = "development",
    gate_decision: dict[str, Any] | None = None,
) -> Path:
    """Generate a standalone report for any verified saved run."""
    if report_type not in {"development", "holdout", "saved-prediction"}:
        raise ValueError(f"Unsupported report type: {report_type}")
    comparable_sharpes = (gate_decision or {}).get("evidence", {}).get("comparable_sharpes", [])
    scale = max((abs(row["sharpe"]) for row in comparable_sharpes), default=1.0) or 1.0
    sharpe_chart = (
        "<h3>Comparable candidate Sharpes</h3><div role='img' aria-label='Comparable candidate Sharpe chart'>"
        + "".join(
            "<p><code>"
            + html.escape(str(row.get("attempt_id")))
            + "</code> "
            + f"<span style='display:inline-block;background:{'#287a55' if row['sharpe'] >= 0 else '#b54d48'};"
            + f"height:1rem;width:{max(1, abs(row['sharpe']) / scale * 20):.2f}rem'></span> "
            + html.escape(_format_value(row["sharpe"]))
            + "</p>"
            for row in comparable_sharpes
        )
        + "</div>"
        if comparable_sharpes
        else ""
    )
    summary_html = (
        "<style>body{font-family:system-ui,sans-serif;line-height:1.4;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;margin:0 0 1.5rem;min-width:28rem}"
        "th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;vertical-align:top}"
        "th{background:#f3f3f3}section{border-top:3px solid #555;margin-top:2rem;padding-top:1rem}"
        "code{word-break:break-all}</style>"
        f"<h1>Qumin {html.escape(report_type)} report</h1>"
        f"<p>Run ID: <code>{html.escape(result.ref.run_id)}</code>. This report is derived from verified saved artifacts.</p>"
        f"<h2>Source identity</h2><p>Snapshot hash: <code>{html.escape(result.ref.snapshot_hash)}</code>"
        f"<br>Configuration hash: <code>{html.escape(result.ref.config_hash)}</code>"
        f"<br>Split hash: <code>{html.escape(result.ref.split_hash)}</code></p>"
        f"<h2>Dataset provenance</h2>{_provenance_html(result)}"
        f"<h2>Candidate comparison</h2><pre>{html.escape(_format_value(result.artifact_manifest.get('candidates', [])))}</pre>"
        "<h2>Campaign evidence</h2><pre>"
        f"{html.escape(_format_value(evidence or {'status': 'unavailable', 'reason': 'not requested'}))}</pre>"
        + (
            f"<h2>Frozen holdout decision</h2><pre>{html.escape(_format_value(gate_decision))}</pre>{sharpe_chart}"
            if gate_decision is not None
            else ""
        )
        + _run_metadata(
            report_type.title(),
            result,
            frequency=config.statistics.frequency,
            min_observations=config.statistics.min_observations,
            fees_bps=config.backtest.fees_bps,
            slippage_bps=config.backtest.slippage_bps,
            calendar=_result_calendar(result, config),
        )
    )
    if result.backtest is None or result.backtest.net_returns is None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"<html><body>{summary_html}<p>Backtest metrics unavailable: no saved net returns.</p></body></html>")
        return output
    timestamps = result.backtest.equity.get_column("timestamp").to_list()
    return generate_local_report(
        path,
        net_returns=result.backtest.net_returns.to_numpy(),
        gross_returns=result.backtest.gross_returns.to_numpy() if result.backtest.gross_returns is not None else None,
        timestamps=timestamps,
        title=f"Qumin {report_type} report",
        frequency=config.statistics.frequency,
        calendar=_result_calendar(result, config),
        summary_html=summary_html,
    )


def _run_locator(result: RunResult, root: Path) -> dict[str, Any]:
    run_dir = root / "runs" / result.ref.run_id
    artifact_paths = [run_dir / "manifest.json"]
    artifact_paths.extend(run_dir / record["path"] for record in result.artifact_manifest.get("artifacts", []))
    return {
        "run_id": result.ref.run_id,
        "run_directory": str(run_dir.resolve()),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "artifact_paths": [str(path.resolve()) for path in artifact_paths],
    }


def write_demo_locator(
    path: str | Path,
    *,
    development: RunResult,
    holdout: RunResult | None,
    report_path: str | Path,
    selected_model: str,
    artifacts_root: str | Path,
    decision: dict[str, Any] | None = None,
) -> Path:
    """Write a write-once locator for a development report and optional holdout."""
    output = Path(path)
    report = Path(report_path)
    if not report.exists():
        raise FileNotFoundError(f"Cannot write demo locator; report is missing: {report}")
    root = Path(artifacts_root)
    payload = {
        "schema_version": "demo-locator.v1",
        "selected_model": selected_model,
        "report_path": str(report.resolve()),
        "development": _run_locator(development, root),
        "holdout": _run_locator(holdout, root) if holdout is not None else None,
        "decision": decision,
        "decision_id": decision.get("decision_id") if decision else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Demo locator already exists: {output}")
    data = json.dumps(payload, default=str, sort_keys=True, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
