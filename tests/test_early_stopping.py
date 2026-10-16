import numpy as np
import pytest
import torch
from torch import nn

from lab.core.config import ModelsConfig
from lab.core.contracts import WindowSet
from lab.models.recurrent import GRUModel
from lab.models.trainer import train_recurrent
from lab.models.xgboost_model import XGBoostModel


def test_recurrent_stopping_curve_and_selected_duration_are_recorded():
    X = np.arange(24 * 3 * 2, dtype=np.float32).reshape(24, 3, 2) / 100.0
    y = X[:, -1, 0]
    stop_X = X[::-1].copy()
    stop_y = y[::-1].copy()
    model = GRUModel(epochs=6, batch_size=6, hidden_size=4, seed=11)

    model.fit(X, y, stopping_data=(stop_X, stop_y), stopping_patience=2)

    assert model.stopping_history is not None
    assert model.stopping_history
    assert model.selected_duration is not None
    assert model.stopping_config["patience"] == 2


def test_early_stopping_config_is_validated_and_disabled_by_default():
    config = ModelsConfig()
    assert config.early_stopping is False
    assert config.early_stopping_fraction == 0.20
    assert config.early_stopping_patience == 5

    try:
        ModelsConfig(early_stopping_fraction=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("fraction=1.0 must be rejected")


def test_fixed_duration_is_used_for_recurrent_and_xgboost_refits():
    X = np.arange(16 * 3 * 2, dtype=np.float32).reshape(16, 3, 2) / 100.0
    y = X[:, -1, 0]

    recurrent = GRUModel(epochs=6, batch_size=4, hidden_size=4)
    recurrent.fit(X, y, duration=2)
    xgboost = XGBoostModel(n_estimators=6, max_depth=2)
    xgboost.fit(X, y, duration=2)

    assert recurrent.selected_duration == 2
    assert xgboost.selected_duration == 2


def test_stopping_loss_is_sample_weighted_with_a_final_partial_batch():
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0))
            self.batch_sizes = []

        def forward(self, features):
            self.batch_sizes.append(len(features))
            return self.value.expand(len(features), 1)

    windows = WindowSet(
        [np.ones((3, 1), dtype=np.float32)],
        np.zeros(3, dtype=np.int64),
        np.arange(3),
        1,
    )
    model = ConstantModel()
    _, stopping_history, _ = train_recurrent(
        model,
        windows,
        np.zeros(3, dtype=np.float32),
        task="regression",
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        seed=5,
        device="cpu",
        stopping_data=(windows, np.asarray([0.0, 0.0, 3.0], dtype=np.float32)),
    )
    prediction = float(model.value.detach().item())
    expected = float(np.mean([prediction**2, prediction**2, (prediction - 3.0) ** 2]))

    assert stopping_history is not None
    assert len(stopping_history) == 1
    assert stopping_history[0]["loss"] == pytest.approx(expected)
    assert model.batch_sizes == [2, 1, 2, 1]
