from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import polars as pl
import pytest

import lab.research.experiment as experiment
from lab.core.config import PipelineConfig, SplitsConfig, Timeframe
from lab.core.contracts import MarketSnapshot
from lab.quant.cv import fold_training_labels
from lab.research.splits import build_split_plan


def _bars(count: int = 20) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(days=index) for index in range(count)]
    return pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"] * (count // 2),
            "timestamp": timestamps,
            "bar_open_time": timestamps,
        }
    )


def test_split_plan_uses_unique_timestamps_and_latest_holdout():
    bars = _bars()
    plan = build_split_plan(bars, SplitsConfig(n_folds=3))
    assert len(plan.decision_timestamps) == 20
    assert plan.holdout_start == plan.decision_timestamps[16]
    assert [fold.validation_start for fold in plan.folds] == sorted(fold.validation_start for fold in plan.folds)
    development = next(period for period in plan.execution_periods if period.name == "development")
    expected_last_development_open = bars.filter(pl.col("timestamp") < plan.holdout_start)["bar_open_time"][-1]
    assert development.final_liquidation_open == expected_last_development_open
    assert development.final_liquidation_open < plan.holdout_start


def test_development_liquidation_is_invariant_to_reserved_bar_open_changes():
    bars = _bars()
    plan = build_split_plan(bars, SplitsConfig(n_folds=3))
    reserved = bars.with_columns(
        pl.when(pl.col("timestamp") == plan.holdout_start)
        .then(pl.col("bar_open_time") + timedelta(hours=4))
        .otherwise(pl.col("bar_open_time"))
        .alias("bar_open_time")
    )
    changed_plan = build_split_plan(reserved, SplitsConfig(n_folds=3))
    assert (
        next(period for period in changed_plan.execution_periods if period.name == "development").final_liquidation_open
        == next(period for period in plan.execution_periods if period.name == "development").final_liquidation_open
    )


def test_holdout_liquidation_respects_an_explicit_early_end():
    bars = _bars()
    explicit_end = bars["timestamp"][18]
    plan = build_split_plan(
        bars,
        SplitsConfig(
            n_folds=3,
            explicit_boundaries={
                "holdout_start": bars["timestamp"][16].isoformat(),
                "holdout_end": explicit_end.isoformat(),
            },
        ),
    )
    holdout = next(period for period in plan.execution_periods if period.name == "holdout")
    assert holdout.last_decision == bars["timestamp"][17]
    assert holdout.final_liquidation_open == bars["bar_open_time"][17]
    assert holdout.final_liquidation_open < explicit_end


def test_strategy_allocation_input_cannot_observe_reserved_ohlc(monkeypatch):
    bars = _bars().with_columns(
        pl.lit(100.0).alias("open"),
        pl.lit(101.0).alias("high"),
        pl.lit(99.0).alias("low"),
        pl.lit(100.5).alias("close"),
        pl.lit(1_000.0).alias("volume"),
    )
    plan = build_split_plan(bars, SplitsConfig(n_folds=3))
    snapshot = MarketSnapshot(
        snapshot_id="fixture",
        snapshot_hash="fixture",
        bars=bars,
        ticker_order=("AAA", "BBB"),
        calendar="XNYS",
        timeframe=Timeframe.D1,
    )
    dataset = SimpleNamespace(snapshot=snapshot, split_plan=plan)
    config = PipelineConfig()
    observed = []

    def capture_allocations(allowed_snapshot, *args, **kwargs):
        observed.append(allowed_snapshot.bars)
        return ()

    monkeypatch.setattr(experiment, "build_allocations", capture_allocations)
    monkeypatch.setattr(experiment, "simulate_strategy", lambda *args: object())
    experiment._strategy_result(dataset, (), config, period_name="development")
    perturbed = bars.with_columns(
        pl.when(pl.col("timestamp") >= plan.holdout_start).then(pl.col("open") * 7).otherwise(pl.col("open")).alias("open"),
        pl.when(pl.col("timestamp") >= plan.holdout_start).then(pl.col("close") * 11).otherwise(pl.col("close")).alias("close"),
    )
    changed_dataset = SimpleNamespace(snapshot=snapshot.model_copy(update={"bars": perturbed}), split_plan=plan)
    experiment._strategy_result(changed_dataset, (), config, period_name="development")
    assert observed[0].equals(observed[1])
    assert observed[0].get_column("timestamp").max() < plan.holdout_start


def test_purge_removes_equality_boundary():
    labels = pl.DataFrame(
        {
            "decision_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "t1": [datetime(2024, 1, 3, tzinfo=timezone.utc)],
        }
    )
    assert fold_training_labels(
        labels,
        type(
            "Fold",
            (),
            {
                "train_start": labels["decision_time"][0],
                "train_end": labels["decision_time"][0] + timedelta(days=1),
                "validation_start": labels["t1"][0],
            },
        )(),
    ).is_empty()


def test_split_plan_rejects_short_validation_history():
    with pytest.raises(ValueError, match="too short"):
        build_split_plan(_bars(6), SplitsConfig(n_folds=3, development_train_fraction=0.8))


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"diagnostic_mode": True}, "diagnostic_mode=true"),
        ({"embargo_bars": 1}, "embargo_bars>0"),
    ],
)
def test_main_walk_forward_rejects_unsupported_split_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        build_split_plan(_bars(), SplitsConfig(**settings))
