from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from lab.research.dataset import build_sample_set


def _frame(ticker, values, raw_indices=None, start=datetime(2024, 1, 1, tzinfo=timezone.utc)):
    result = {
        "ticker": [ticker] * len(values),
        "timestamp": [start + timedelta(days=index) for index in range(len(values))],
        "close": values,
    }
    if raw_indices is not None:
        result["raw_bar_index"] = raw_indices
    return pl.DataFrame(result)


def test_materialized_windows_match_prechange_order_and_include_final_window():
    frame = pl.concat(
        [
            _frame("AAA", [10, 11, 12, 13, 14, 15], [0, 1, 2, 4, 5, 6]),
            _frame("BBB", [20, 21], [0, 1]),
        ]
    )

    samples = build_sample_set(frame, ["close"], 3)

    assert samples.X.shape == (2, 3, 1)
    np.testing.assert_array_equal(
        samples.X.to_numpy(),
        np.asarray([[[10], [11], [12]], [[13], [14], [15]]], dtype=np.float32),
    )
    assert [(key.ticker, key.timestamp.day) for key in samples.keys] == [("AAA", 3), ("AAA", 6)]
    assert samples.window_start[0].day == 1
    assert samples.window_start[1].day == 4
    assert samples.window_end == tuple(key.timestamp for key in samples.keys)


def test_nonfinite_feature_rows_exclude_every_window_that_touches_them():
    frame = _frame("AAA", [1.0, 2.0, np.nan, 4.0, 5.0])

    samples = build_sample_set(frame, ["close"], 2)

    np.testing.assert_array_equal(samples.X.to_numpy(), np.asarray([[[1.0], [2.0]], [[4.0], [5.0]]], dtype=np.float32))
    assert [key.timestamp.day for key in samples.keys] == [2, 5]


def test_allowed_decisions_and_label_join_exclude_unlabeled_endpoints():
    frame = _frame("AAA", [1, 2, 3, 4, 5])
    labels = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "decision_time": [frame["timestamp"][2], frame["timestamp"][4]],
            "target": [0.25, -0.5],
            "event_end": [frame["timestamp"][3], frame["timestamp"][4]],
            "t1": [frame["timestamp"][3], frame["timestamp"][4]],
        }
    )
    allowed = {("AAA", frame["timestamp"][2]), ("AAA", frame["timestamp"][3]), ("AAA", frame["timestamp"][4])}

    samples = build_sample_set(
        frame,
        ["close"],
        2,
        labels=labels,
        allowed_decisions=allowed,
        target_name="target",
    )

    assert [key.timestamp.day for key in samples.keys] == [3, 5]
    assert samples.y is not None
    np.testing.assert_array_equal(samples.y.reshape(-1), np.asarray([0.25, -0.5], dtype=np.float32))
    assert samples.label_metadata is not None
    assert len(samples.label_metadata) == 2


def test_empty_and_target_free_results_remain_lazy():
    empty = build_sample_set(_frame("AAA", [1, 2]), ["close"], 3)
    unsupervised = build_sample_set(_frame("BBB", [1, 2, 3]), ["close"], 3)

    assert empty.X.shape == (0, 3, 1)
    assert empty.X.to_numpy().shape == (0, 3, 1)
    assert unsupervised.y is None
    assert unsupervised.X.y is None
    item = unsupervised.X[0]
    assert isinstance(item, np.ndarray)
    assert item.shape == (3, 1)


def test_supervised_windows_require_explicit_t1():
    frame = _frame("AAA", [1, 2, 3])
    labels = pl.DataFrame({"ticker": ["AAA"], "decision_time": [frame["timestamp"][2]], "target": [1.0]})

    with pytest.raises(ValueError, match="t1"):
        build_sample_set(frame, ["close"], 3, labels=labels, target_name="target")
