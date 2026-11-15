from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from lab.core.config import PipelineConfig, Timeframe, get_platform_config
from lab.platform.synthetic_data import demo_metadata, generate_demo_bars
from lab.research.experiment import run_experiment
from lab.research.reporting import generate_run_report, write_demo_locator


def _selected_model(candidate_status: list[dict[str, Any]]) -> str:
    """Return the completed model selected by the development run."""
    selected = next(
        (item.get("selected_model") for item in candidate_status if item.get("selected_model")),
        None,
    )
    if not isinstance(selected, str):
        raise RuntimeError("Demo development run did not record a selected model")
    candidate = next(
        (item for item in candidate_status if item.get("model") == selected and item.get("status") == "completed"),
        None,
    )
    if candidate is None:
        raise RuntimeError(f"Demo selected model is not a completed candidate: {selected}")
    return selected


def _require_backtest(result: Any, label: str) -> None:
    """Reject a result that cannot support saved return reporting."""
    if result.backtest is None or result.backtest.net_returns is None:
        raise RuntimeError(f"Demo {label} result has no saved net return series")


def run_demo(config_path: str | Path = "config/demo.yaml", *, timeframe: str | None = None) -> Path:
    """Run the finite bundled-data development demonstration."""
    config = PipelineConfig.from_yaml(Path(config_path))
    if config.data.source != "bundled_demo":
        raise ValueError("The offline demo requires data.source: bundled_demo")
    if timeframe is not None:
        selected_timeframe = Timeframe(timeframe)
        if selected_timeframe != config.data.timeframe and selected_timeframe == Timeframe.D1:
            raise ValueError(
                f"Cannot override a {config.data.timeframe.value} demo config with --timeframe 1d; "
                "use a daily config or omit --timeframe"
            )
        if selected_timeframe == Timeframe.H1:
            demo_root = Path(get_platform_config().paths.artifacts_dir) / "demo"
            input_path = demo_root / "input-hourly.csv"
            metadata_path = demo_root / "input-hourly.yaml"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            generate_demo_bars(timeframe=selected_timeframe).write_csv(input_path)
            metadata_path.write_text(yaml.safe_dump(demo_metadata(timeframe=selected_timeframe), sort_keys=False))
            config = PipelineConfig.from_yaml(
                Path(config_path),
                overrides={"data": {"timeframe": "1h", "input_path": input_path, "metadata_path": metadata_path}},
            )
    development = run_experiment(config)
    _require_backtest(development, "development")
    selected_model = _selected_model(development.artifact_manifest.get("candidates", []))
    report_root = Path(get_platform_config().paths.artifacts_dir) / "demo"
    report_path = report_root / f"{development.ref.run_id}.html"
    report = generate_run_report(report_path, result=development, config=config, report_type="development")
    locator = write_demo_locator(
        report_root / f"{development.ref.run_id}.locator.json",
        development=development,
        holdout=None,
        report_path=report,
        selected_model=selected_model,
        artifacts_root=Path(get_platform_config().paths.artifacts_dir),
    )
    print(f"development_run_id={development.ref.run_id}")
    print(f"selected_model={selected_model}")
    print(f"report_path={report.resolve()}")
    print(f"locator_path={locator.resolve()}")
    print(f"development_run_dir={Path(get_platform_config().paths.artifacts_dir, 'runs', development.ref.run_id).resolve()}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the finite offline research demo")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--timeframe", choices=[item.value for item in Timeframe])
    parser.add_argument("--store-root", type=Path, default=Path("artifacts/phase1-demo"))
    args = parser.parse_args(argv)
    if args.config is None and args.timeframe is None:
        from lab.demo_phase1 import run_phase1_demo

        outcome = run_phase1_demo(store_root=args.store_root)
        selection = outcome["candidate_selection"]
        print(f"report_path={outcome['report_path']}")
        print(f"candidate_selection_run_id={outcome['selected_run_id']}")
        print(f"candidate_selection_verdict={selection['verdict']}")
        print(f"candidate_selection_score={selection['score']}")
        print(f"candidate_selection_reason={selection['reason']}")
        print(f"gate_K={outcome['K']}")
        print(f"holdout_verdict={outcome['decision']['verdict']}")
    else:
        run_demo(args.config or Path("config/demo.yaml"), timeframe=args.timeframe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
