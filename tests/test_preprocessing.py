import math
from datetime import datetime, timedelta, timezone

import polars as pl

from lab.core.config import PipelineConfig, Timeframe
from lab.core.contracts import FoldSpec, MarketSnapshot, PreparedDataset, PreprocessingState
from lab.research.preprocessing import FoldPreprocessor, fit_preprocessor, prepare_fold


def test_preprocessing_state_is_fit_only_on_train_history():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "timestamp": [start + timedelta(days=index) for index in range(4)],
            "value": [1.0, 2.0, 3.0, 10000.0],
        }
    )
    state = fit_preprocessor(frame, ["value"], start, start + timedelta(days=3), PipelineConfig())
    assert state.state.means["value"] < 10.0


def test_unknown_categories_are_encoded_without_refitting():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": [start + timedelta(days=index) for index in range(3)],
            "category": ["known", "known", "new"],
        }
    )
    preprocessor = fit_preprocessor(frame, ["category"], start, start + timedelta(days=2), PipelineConfig())
    transformed = preprocessor.transform(frame)
    assert transformed["category"].to_list()[-1] == -1.0


def test_transform_executes_fitted_ffd_before_scaling():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": [start + timedelta(days=index) for index in range(3)],
            "close": [1.0, 2.0, 4.0],
        }
    )
    preprocessor = FoldPreprocessor(
        PreprocessingState(
            feature_order=("close",),
            ffd_orders={"AAA": 1.0},
            ffd_features=("close",),
            ffd_threshold=0.001,
        )
    )
    transformed = preprocessor.transform(frame)
    assert transformed["close"][1] == math.log(2.0)
    assert transformed["close"][2] == math.log(2.0)


def test_fold_predictions_include_unresolved_tail_while_scoring_only_resolved_keys():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(days=index) for index in range(12)]
    features = pl.DataFrame(
        {
            "ticker": ["AAA"] * len(timestamps),
            "timestamp": timestamps,
            "raw_bar_index": list(range(len(timestamps))),
            "volume": [float(index + 1) for index in range(len(timestamps))],
        }
    )
    labels = pl.DataFrame(
        {
            "ticker": ["AAA"] * 5,
            "decision_time": [timestamps[index] for index in (2, 3, 5, 6, 7)],
            "target_1b_v2": [0.1, -0.1, 0.2, 0.3, -0.2],
            "event_end": [timestamps[index] for index in (3, 4, 6, 9, 8)],
            "t1": [timestamps[3], timestamps[4], timestamps[6], timestamps[9], None],
        }
    )
    snapshot = MarketSnapshot(
        snapshot_id="fixture", snapshot_hash="fixture", bars=features,
        ticker_order=("AAA",), calendar="XNYS", timeframe=Timeframe.D1,
    )
    config = PipelineConfig(tensor={"sequence_len": 3, "batch_size": 2})
    dataset = PreparedDataset(
        snapshot=snapshot, features=features, feature_specification=("volume",),
        labels=labels, split_plan={}, config=config,
    )
    fold = FoldSpec(
        fold_id=0, train_start=timestamps[0], train_end=timestamps[5],
        validation_start=timestamps[5], validation_end=timestamps[9],
    )

    bundle = prepare_fold(dataset, fold)

    assert [key.timestamp for key in bundle.evaluation_samples.keys] == timestamps[5:9]
    assert [key.timestamp for key in bundle.scoring_keys] == [timestamps[5]]
    assert bundle.diagnostics["scoring"]["unresolved_rows"] == 1
    assert bundle.evaluation_samples.y is None
    assert bundle.evaluation_samples.X.y is None

    holdout = prepare_fold(
        dataset,
        FoldSpec(
            fold_id=-1, train_start=timestamps[0], train_end=timestamps[7],
            validation_start=timestamps[9], validation_end=start + timedelta(days=12),
        ),
    )

    assert holdout.fold_ref == "holdout"
    assert [key.timestamp for key in holdout.evaluation_samples.keys] == timestamps[9:12]
    assert holdout.scoring_keys == ()
