import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from lab.platform.synthetic_data import generate_null_control_bars, generate_positive_control_bars

ROOT = Path(__file__).parent / "fixtures"


def test_committed_null_fixture_matches_generator_and_checksum():
    csv_path = ROOT / "phase1-null-control.csv"
    metadata = json.loads((ROOT / "phase1-null-control.json").read_text())
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    saved = pl.read_csv(csv_path, try_parse_dates=True)
    regenerated = generate_null_control_bars()
    assert saved.equals(regenerated)
    assert metadata["generator_version"] == "phase1-null-control-v1"
    assert metadata["calendar"] == "XNYS"


def test_null_open_to_open_returns_have_zero_mean_and_independent_features():
    bars = pl.read_csv(ROOT / "phase1-null-control.csv")
    for _, group in bars.partition_by("ticker", as_dict=True).items():
        opened = group["open"].to_numpy()
        future_returns = opened[1:] / opened[:-1] - 1.0
        close_signal = group["close"].to_numpy()[:-1] / opened[:-1] - 1.0
        volume = group["volume"].to_numpy()[:-1]
        assert abs(float(np.mean(future_returns))) < 0.003
        assert abs(float(np.corrcoef(close_signal, future_returns)[0, 1])) < 0.15
        assert abs(float(np.corrcoef(volume, future_returns)[0, 1])) < 0.15


def test_positive_control_plants_signal_on_exact_executed_open_to_open_horizon():
    fixture_path = ROOT / "phase1-positive-control.csv"
    metadata = json.loads((ROOT / "phase1-positive-control.json").read_text())
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    assert pl.read_csv(fixture_path, try_parse_dates=True).equals(generate_positive_control_bars())
    bars = pl.read_csv(fixture_path)
    first = bars.filter(pl.col("ticker") == "ALFA")
    signal = (np.log(first["volume"].to_numpy()) - 12.0) / 0.35
    opens = first["open"].to_numpy()
    executed_returns = opens[2:] / opens[1:-1] - 1.0
    assert float(np.corrcoef(signal[:-2], executed_returns)[0, 1]) > 0.8
