"""Fold training is invariant to OHLCV data first observed after its cutoff."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from lab.core.config import PipelineConfig, Timeframe
from lab.core.contracts import FoldSpec
from lab.demo_phase1 import _demo_config
from lab.platform.data_access import build_market_snapshot
from lab.research.preprocessing import prepare_dataset, prepare_fold
from lab.research.training import train_candidate


def test_training_inputs_and_fitted_state_ignore_future_valid_bars():
    fixture = Path("tests/fixtures/phase1-positive-control.csv")
    snapshot = build_market_snapshot(
        [fixture],
        calendar="XNYS",
        timeframe=Timeframe.D1,
        provenance={"fixture": "phase1-positive"},
        source_manifest=[{"path": str(fixture)}],
        metadata=json.loads(Path("tests/fixtures/phase1-positive-control.json").read_text()),
    )
    values = _demo_config(snapshot.snapshot_id).model_dump(mode="python")
    values["models"]["early_stopping"] = True
    config = PipelineConfig(**values)
    original = prepare_dataset(config, snapshot=snapshot)
    folds = [
        *original.split_plan.folds,
        FoldSpec(
            fold_id=-1,
            train_start=original.split_plan.development_start,
            train_end=original.split_plan.holdout_start,
            validation_start=original.split_plan.holdout_start,
            validation_end=original.split_plan.holdout_end,
        ),
    ]
    for fold in folds:
        future = pl.col("timestamp") >= fold.train_end
        altered = snapshot.bars.with_columns(
            pl.when(future).then(pl.col("open") * 1.01).otherwise(pl.col("open")).alias("open"),
            pl.when(future).then(pl.col("high") * 1.02).otherwise(pl.col("high")).alias("high"),
            pl.when(future).then(pl.col("low") * 0.98).otherwise(pl.col("low")).alias("low"),
            pl.when(future).then(pl.col("close") * 1.01).otherwise(pl.col("close")).alias("close"),
            pl.when(future).then(pl.col("volume") * 1.1).otherwise(pl.col("volume")).alias("volume"),
        )
        rebuilt = prepare_dataset(config, snapshot=snapshot.model_copy(update={"bars": altered}))
        before, after = prepare_fold(original, fold), prepare_fold(rebuilt, fold)
        assert before.preprocessor.state == after.preprocessor.state
        assert before.train_samples.keys == after.train_samples.keys
        np.testing.assert_allclose(before.train_samples.X.to_numpy(), after.train_samples.X.to_numpy())
        np.testing.assert_allclose(before.train_samples.y, after.train_samples.y)
        trained = train_candidate(before, config, model_name="xgboost", dataset=original, fold=fold)
        reproduced = train_candidate(after, config, model_name="xgboost", dataset=rebuilt, fold=fold)
        assert trained.selected_duration == reproduced.selected_duration
        assert trained.stopping_diagnostics == reproduced.stopping_diagnostics
        probe = before.train_samples.X
        np.testing.assert_allclose(trained.predict(probe), reproduced.predict(probe), rtol=1e-7, atol=1e-9)
