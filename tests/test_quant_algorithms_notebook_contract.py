import ast
import json
import re
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/quant_algorithms_walkthrough.ipynb")
README_PATH = Path("README.md")
QUANT_ROOT = Path("src/lab/quant")


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text())


def _source():
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])


def _code_cells():
    return [cell for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


def test_notebook_is_unexecuted_and_controls_are_the_only_researcher_inputs():
    notebook = _notebook()
    code_cells = _code_cells()

    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)

    controls = next(cell for cell in code_cells if "DATA_MODE" in "".join(cell["source"]))
    tree = ast.parse("".join(controls["source"]))
    assigned = [node.targets[0].id for node in tree.body if isinstance(node, ast.Assign)]
    assert assigned == ["DATA_MODE", "RUN_ID", "SAMPLE_TICKER"]
    assert len(tree.body) == 3


def test_snapshot_resolver_has_deterministic_real_local_and_single_live_paths():
    source = _source()

    assert 'DATA_MODE not in {"local", "live"}' in source
    assert "load_verified_local_run(RUN_ID.strip())" in source
    assert "load_run(manifest_path.parent.name, artifacts_dir=artifacts_root)" in source
    assert "key=lambda path: (path.stat().st_mtime_ns, path.parent.name)" in source
    assert "manifest_has_snapshot" in source
    assert 'manifest.get("status") != "completed"' in source
    assert "bundled-demo/synthetic provenance" in source
    assert "Rejected candidates" in source
    assert "live mode ignores RUN_ID" in source
    assert 'config.data.source != "provider"' in source
    assert source.count("load_market_snapshot(config)") == 1
    assert "snapshot.ticker_order[0]" in source
    assert "Unknown SAMPLE_TICKER" in source
    for field in ("snapshot_id", "snapshot_hash", "timeframe", "calendar", "tickers", "row_count", "date_start", "date_end"):
        assert f'"{field}"' in source


def test_timing_validation_and_features_use_public_apis_and_standard_headings():
    source = _source()

    for public_call in (
        "get_exchange_calendar(", "as_utc(", "session_schedule(", "bar_times_for_session(", "normalize_provider_timestamp(",
        "normalize_explicit_bar_times(", "validate_schema(", "validate_nulls(",
        "validate_prices(", "validate_monotonic_timestamps(", "validate_canonical_bars(",
        "validate_explicit_bar_times(", "run_all_validations(", "apply_all_features(",
        "feature_specification(",
    ):
        assert public_call in source
    for heading in ("ALGORITHM:", "INPUT:", "PARAMETERS:", "OUTPUT", "INTERPRETATION:"):
        assert heading in source
    assert "downstream demonstrations are intentionally stopped" in source
    assert "warmup_nulls" in source
    assert "raw bars remain unchanged" in source
    assert "close / close.shift" not in source
    assert "rolling_mean(window_size" not in source


def test_ffd_barrier_labeling_and_cv_sections_are_visual_and_honest():
    source = _source()

    for public_call in (
        "run_adf_test(", "find_min_d_grid(", "frac_diff_ffd(", "get_weights_ffd(",
        "compute_memory_corr(", "compute_labels(", "barrier_prices(", "first_barrier_touch(",
        "build_split_plan(", "fold_training_labels(", "scoreable_labels(",
    ):
        assert public_call in source
    assert "plt.subplots(1, 2" in source
    assert "upper barrier" in source
    assert "lower barrier" in source
    assert "first touch" in source
    assert "expiry (no touch)" in source
    assert "purge/embargo" in source
    assert "reserved holdout boundary (not evaluated)" in source
    assert source.count("UNAVAILABLE:") >= 8


def test_covariance_hrp_and_constraints_are_risk_allocation_not_forecasting():
    source = _source()
    lowered = source.lower()

    for public_call in (
        "estimate_ledoit_wolf(", "covariance_array(", "cov_to_corr(", "hrp_pypfort(",
        "hrp_pipe(", "hrp_custom(", "apply_long_only(", "apply_min_position(",
        "apply_max_position(", "apply_all_constraints(", "validate_budget(",
    ):
        assert public_call in source
    for field in (
        "hrp_weight", "weight_rank", "standalone_volatility", "strongest_correlation_partner",
        "strongest_correlation", "cluster_leaf_order", "allocation_note",
    ):
        assert field in source
    assert "hrp does not forecast returns" in lowered
    assert "risk allocation" in lowered
    assert "recursive bisection" in lowered
    assert "unused capacity stays cash" in lowered
    assert "unsupported as designed" in lowered
    assert "inverse-variance" not in lowered


def test_educational_signal_uses_existing_contracts_and_reports_cost_evidence():
    source = _source()

    for public_call in (
        "PredictionFrame(", "SampleKey(", "build_allocations(", "predictions_to_signals",
        "simulate_strategy(", "regression_metrics(", "classification_metrics(",
        "sharpe_evidence_metrics(",
    ):
        assert public_call in source
    assert "educational_signal" in source
    assert 'lag_source = f"return_{lag_days}b_lag1"' in source
    assert "return_{lag_days}d_lag1" not in source
    assert 'signal_frame["timestamp"].unique().sort().tail(3).to_list()' in source
    assert "information timestamp < decision timestamp <= execution timestamp" in source
    assert "gross equity (no costs)" in source
    assert "net equity (fees + slippage)" in source
    assert "educational buy" in source and "educational sell" in source
    assert "trial_sharpes=None" in source
    assert "no model selection occurred" in source
    assert "all-cash accounting" in source


def test_coverage_map_matches_every_quant_module_exactly():
    source = _source()
    match = re.search(r"EXPECTED_QUANT_MODULES = \((.*?)\)\ncoverage_rows", source, re.DOTALL)
    assert match is not None
    expected = set(re.findall(r'"([_a-z]+\.py)"', match.group(1)))
    actual = {path.name for path in QUANT_ROOT.glob("*.py")}

    assert expected == actual
    for module in actual:
        assert f'"module": "{module}"' in source
    assert '"module": "__init__.py", "public_algorithm_or_api": "package import/export boundary"' in source
    assert "custom route intentionally unsupported" in source
    assert "No model was trained, no holdout outcome was inspected" in source


def test_readme_documents_modes_roles_visuals_and_preserves_old_notebook():
    readme = README_PATH.read_text().lower()

    assert "quant_algorithms_walkthrough.ipynb" in readme
    assert "data_mode" in readme and "run_id" in readme and "sample_ticker" in readme
    assert "latest completed real-data run" in readme
    assert "one provider snapshot" in readme
    assert "make demo" in readme
    assert "research_walkthrough.ipynb" in readme
    assert "covariance" in readme and "hrp" in readme and "gross-versus-net" in readme
    assert Path("notebooks/quant_sins_and_corruption_test.ipynb").is_file()
