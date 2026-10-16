from datetime import datetime, timezone

import numpy as np
import pytest

from lab.core.contracts import FoldBundle, PredictionFrame, SampleKey, SampleSet, ScoringTargets, WindowSet


def _keys(count: int) -> tuple[SampleKey, ...]:
    return tuple(SampleKey(ticker="AAA", timestamp=datetime(2024, 1, i + 1, tzinfo=timezone.utc)) for i in range(count))


def test_sample_set_rejects_future_features_and_preserves_keys():
    with pytest.raises(ValueError, match="Future-derived"):
        SampleSet(
            X=WindowSet([np.ones((4, 1))], np.asarray([0, 0]), np.asarray([0, 1]), 3),
            keys=_keys(2),
            window_start=tuple(key.timestamp for key in _keys(2)),
            window_end=tuple(key.timestamp for key in _keys(2)),
            feature_order=("target_1b",),
        )


def test_window_set_rejects_out_of_bounds_descriptors():
    with pytest.raises(ValueError, match="exceeds its feature block bounds"):
        WindowSet([np.ones((3, 1))], np.asarray([0]), np.asarray([1]), 3)

    with pytest.raises(ValueError, match="integer block_ids and starts"):
        WindowSet([np.ones((3, 1))], np.asarray([0.5]), np.asarray([0]), 3)


def test_sample_set_metadata_must_align_without_materializing_windows():
    windows = WindowSet([np.ones((4, 1))], np.asarray([0, 0]), np.asarray([0, 1]), 3)
    with pytest.raises(ValueError, match="metadata"):
        SampleSet(
            X=windows,
            keys=_keys(2),
            window_start=(_keys(2)[0].timestamp,),
            window_end=tuple(key.timestamp for key in _keys(2)),
            feature_order=("close",),
        )


def test_prediction_frame_requires_three_probability_columns():
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        PredictionFrame(keys=_keys(2), model_ref="model", fold_ref="fold", predictions=np.ones((2, 2)))


def test_naive_sample_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        SampleKey(ticker="AAA", timestamp=datetime(2024, 1, 1))


def test_fold_bundle_keeps_scoring_targets_keyed_and_separate_from_evaluation_windows():
    keys = _keys(2)
    samples = SampleSet(
        X=WindowSet([np.ones((4, 1))], np.asarray([0, 0]), np.asarray([0, 1]), 3),
        keys=keys,
        window_start=tuple(key.timestamp.replace(day=1) for key in keys),
        window_end=tuple(key.timestamp for key in keys),
        feature_order=("close",),
    )

    bundle = FoldBundle(
        fold_ref="0",
        train_samples=samples,
        evaluation_samples=samples,
        scoring_keys=(keys[1],),
        scoring_targets=ScoringTargets(keys=(keys[1],), values=np.asarray([0.25])),
        preprocessor=None,
        boundaries={},
    )

    assert bundle.evaluation_samples.y is None
    assert bundle.evaluation_samples.X.y is None
    np.testing.assert_array_equal(bundle.scoring_targets.values, [0.25])

    with pytest.raises(ValueError, match="belong to evaluation keys"):
        FoldBundle(
            fold_ref="0",
            train_samples=samples,
            evaluation_samples=samples,
            scoring_keys=(_keys(3)[2],),
            scoring_targets=ScoringTargets(keys=(_keys(3)[2],), values=np.asarray([0.25])),
            preprocessor=None,
            boundaries={},
        )


def test_scoring_targets_reject_duplicate_keys():
    duplicate = _keys(1)[0]

    with pytest.raises(ValueError, match="keys must be unique"):
        ScoringTargets(keys=(duplicate, duplicate), values=np.asarray([0.1, 0.2]))
