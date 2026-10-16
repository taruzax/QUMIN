from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest
import torch

from lab.core.contracts import WindowSet
from lab.models.recurrent import GRUModel, LSTMModel
from lab.models.registry import create_model
from lab.research.dataset import build_sample_set


def _windows(rows: int = 20, steps: int = 4, features: int = 2) -> np.ndarray:
    values = np.arange(rows * steps * features, dtype=np.float32)
    return values.reshape(rows, steps, features) / 100.0


@pytest.mark.parametrize("model_type", [GRUModel, LSTMModel])
def test_recurrent_regression_shape_and_save_reload(model_type, tmp_path):
    X = _windows()
    y = X[:, -1, 0] * 2.0
    model = model_type(epochs=2, batch_size=5, hidden_size=4, seed=7)

    model.fit(X, y)
    predictions = model.predict(X)
    restored = model_type.load(model.save(tmp_path / model_type.__name__))

    assert predictions.shape == (len(X),)
    np.testing.assert_allclose(predictions, restored.predict(X))
    assert restored.network is not None
    assert not restored.network.training


@pytest.mark.parametrize("model_type", [GRUModel, LSTMModel])
def test_recurrent_classification_returns_fixed_probability_order(model_type):
    X = _windows()
    y = np.asarray([-1, 0, 1, 0] * 5, dtype=np.int8)
    model = model_type(task="classification", epochs=2, batch_size=5, hidden_size=4)

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    assert model.predict(X).shape == (len(X),)


@pytest.mark.parametrize("model_type", [GRUModel, LSTMModel])
def test_fresh_recurrent_fits_seed_initialization_before_weight_creation(model_type):
    X = _windows(rows=12)
    y = np.asarray([-1, 0, 1] * 4, dtype=np.int8)
    settings = {"task": "classification", "epochs": 1, "batch_size": 4, "hidden_size": 4}

    first = model_type(**settings, seed=31).fit(X, y)
    repeated = model_type(**settings, seed=31).fit(X, y)
    different = model_type(**settings, seed=32).fit(X, y)

    np.testing.assert_allclose(first.predict(X), repeated.predict(X), rtol=0, atol=0)
    assert not np.allclose(first.predict(X), different.predict(X))


def test_repeated_fit_continues_the_existing_network():
    X = _windows(rows=12)
    y = X[:, -1, 0]
    model = GRUModel(epochs=1, batch_size=4, hidden_size=4, seed=7).fit(X, y)
    fitted_network = model.network

    model.fit(X, y)

    assert model.network is fitted_network


@pytest.mark.parametrize(
    "model_type,task",
    [
        (GRUModel, "regression"),
        (LSTMModel, "regression"),
        (GRUModel, "classification"),
        (LSTMModel, "classification"),
    ],
)
def test_recurrent_models_fit_and_predict_lazy_windows_in_order(model_type, task, tmp_path):
    feature_rows = np.arange(10 * 2, dtype=np.float32).reshape(10, 2) / 20
    targets = np.asarray([0, 1, 2, 1, 0, 2, 1], dtype=np.float32)
    if task == "classification":
        targets = np.asarray([-1, 0, 1, 0, -1, 1, 0], dtype=np.int8)
    windows = WindowSet(
        [feature_rows],
        np.zeros(7, dtype=np.int64),
        np.arange(7),
        4,
        y=targets,
    )
    model = model_type(task=task, epochs=2, batch_size=3, hidden_size=4, seed=17)

    model.fit(windows, targets)
    predictions = model.predict(windows)
    with torch.no_grad():
        ordered_outputs = np.concatenate(
            [
                model.network(
                    torch.from_numpy((windows[index][0] if isinstance(windows[index], tuple) else windows[index])[None])
                ).numpy()
                for index in range(len(windows))
            ]
        )
    expected = model.classes[np.argmax(ordered_outputs, axis=1)] if task == "classification" else ordered_outputs.reshape(-1)
    restored = model_type.load(model.save(tmp_path / f"{model_type.__name__}-{task}"))

    np.testing.assert_allclose(predictions, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(predictions, restored.predict(windows))
    assert len(predictions) == 7
    if task == "classification":
        probabilities = model.predict_proba(windows)
        assert probabilities.shape == (7, 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)


def test_recurrent_fit_with_stopping_and_prediction_never_materializes_window_set(monkeypatch):
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 18,
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=index) for index in range(18)],
            "feature": np.linspace(0.0, 1.0, 18, dtype=np.float32),
        }
    )
    training_samples = build_sample_set(frame.head(12), ["feature"], 3)
    stopping_samples = build_sample_set(frame.slice(12, 6), ["feature"], 3)
    monkeypatch.setattr(WindowSet, "to_numpy", lambda self: (_ for _ in ()).throw(AssertionError("unexpected materialization")))
    model = GRUModel(epochs=2, batch_size=4, hidden_size=4, seed=23)

    model.fit(
        training_samples.X,
        np.linspace(0.0, 1.0, len(training_samples.X), dtype=np.float32),
        stopping_data=(stopping_samples.X, np.linspace(0.0, 1.0, len(stopping_samples.X), dtype=np.float32)),
    )
    predictions = model.predict(stopping_samples.X)

    assert predictions.shape == (len(stopping_samples.X),)


def test_recurrent_dropout_is_explicit():
    with pytest.raises(ValueError, match="at least two"):
        LSTMModel(num_layers=1, recurrent_dropout=0.2)

    model = LSTMModel(num_layers=2, recurrent_dropout=0.2, output_dropout=0.1)
    assert model.recurrent_dropout == 0.2
    assert model.output_dropout == 0.1


def test_registry_creates_recurrent_models_with_defaults():
    assert isinstance(create_model("gru", task="regression"), GRUModel)
    assert isinstance(create_model("lstm", task="regression"), LSTMModel)
