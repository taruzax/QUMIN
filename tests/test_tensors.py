import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from lab.research.dataset import build_sample_set


def _frame(rows: int = 12, tickers: int = 2) -> pl.DataFrame:
    frames = []
    for ticker_index in range(tickers):
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index) for index in range(rows)],
                    "ticker": [f"T{ticker_index}"] * rows,
                    "f1": np.arange(rows, dtype=np.float64),
                    "f2": np.arange(rows, dtype=np.float64) * 2,
                }
            )
        )
    return pl.concat(frames)


def test_window_set_is_a_map_style_dataloader_dataset():
    samples = build_sample_set(_frame(), ["f1", "f2"], sequence_len=4)
    loader = DataLoader(samples.X, batch_size=5, shuffle=False, drop_last=False)

    batch = next(iter(loader))

    assert len(samples.X) == 18
    assert batch.shape == (5, 4, 2)
    assert torch.equal(batch[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert samples.keys[9].ticker == "T1"
    np.testing.assert_array_equal(samples.X[9][:, 0], np.asarray([0, 1, 2, 3], dtype=np.float32))
    assert [len(item) for item in loader][-1] == 3


def test_window_set_materializes_only_when_explicitly_requested():
    samples = build_sample_set(_frame(rows=5, tickers=1), ["f1"], sequence_len=3)
    lazy = samples.X

    assert lazy.shape == (3, 3, 1)
    assert lazy[0].shape == (3, 1)
    assert not lazy.feature_blocks[0].flags.writeable
    assert lazy[0].flags.writeable
    materialized = lazy.to_numpy()
    assert materialized.flags.writeable
    np.testing.assert_array_equal(
        materialized[:, :, 0],
        np.asarray([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.float32),
    )


def test_window_set_item_is_an_independent_copy():
    samples = build_sample_set(_frame(rows=5, tickers=1), ["f1"], sequence_len=3)
    before = samples.X.to_numpy()
    item = samples.X[0]

    assert isinstance(item, np.ndarray)
    item[0, 0] = -100.0

    np.testing.assert_array_equal(samples.X.to_numpy(), before)


def test_window_set_dataloader_emits_no_readonly_array_warning():
    samples = build_sample_set(_frame(rows=5, tickers=1), ["f1"], sequence_len=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        batches = list(DataLoader(samples.X, batch_size=2, shuffle=False, drop_last=False))

    assert [len(batch) for batch in batches] == [2, 1]


def test_supervised_window_set_batches_features_and_targets():
    frame = _frame(rows=6, tickers=1)
    labels = pl.DataFrame(
        {
            "ticker": ["T0", "T0", "T0", "T0"],
            "decision_time": frame["timestamp"][2:6],
            "target": [0.1, 0.2, 0.3, 0.4],
            "t1": frame["timestamp"][2:6],
        }
    )
    samples = build_sample_set(frame, ["f1", "f2"], sequence_len=3, labels=labels, target_name="target")
    loader = DataLoader(samples.X, batch_size=3, shuffle=False, drop_last=False)

    features, targets = next(iter(loader))

    assert features.shape == (3, 3, 2)
    assert targets.shape == (3, 1)
    torch.testing.assert_close(targets[:, 0], torch.tensor([0.1, 0.2, 0.3]))
