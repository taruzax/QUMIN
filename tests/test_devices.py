from unittest.mock import patch

import pytest
import torch
from torch import nn

from lab.core.config import PipelineConfig
from lab.models.devices import resolve_device
from lab.models.trainer import seed_torch, train_recurrent
from lab.research.training import _new_model


def test_device_resolution_falls_back_to_cpu_when_no_accelerator_is_available():
    with (
        patch("lab.models.devices.torch.cuda.is_available", return_value=False),
        patch("lab.models.devices.torch.backends.mps.is_available", return_value=False),
    ):
        resolution = resolve_device("gru", "auto")

    assert resolution.resolved_device == "cpu"
    assert "cpu" in resolution.available_devices


def test_unavailable_explicit_cuda_request_fails_clearly():
    with patch("lab.models.devices.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
            resolve_device("xgboost", "cuda")


def test_baseline_remains_cpu_when_pipeline_requests_acceleration():
    config = PipelineConfig(models={"device": "cuda"})
    model = _new_model(config, "baseline")

    assert model.device_resolution["resolved_device"] == "cpu"
    assert model.runtime_metadata["requested_device"] == "cuda"


def test_cpu_seed_policy_is_forwarded_and_reproducible():
    with patch("lab.models.trainer.torch.use_deterministic_algorithms") as deterministic:
        seed_torch(13, "strict")
        first = torch.rand(4)
        seed_torch(13, "strict")
        second = torch.rand(4)

    assert torch.equal(first, second)
    deterministic.assert_called_with(True, warn_only=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_device_resolution_smoke_when_hardware_exists():
    assert resolve_device("xgboost", "cuda").resolved_device == "cuda"


@pytest.mark.skipif(
    not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    reason="MPS hardware is unavailable",
)
def test_mps_device_resolution_smoke_when_hardware_exists():
    assert resolve_device("gru", "mps").resolved_device == "mps"


class _TinyRecurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRU(input_size=2, hidden_size=3, batch_first=True)
        self.output = nn.Linear(3, 1)

    def forward(self, values):
        sequence, _ = self.recurrent(values)
        return self.output(sequence[:, -1, :]).squeeze(-1)


@pytest.mark.parametrize(
    "device,available",
    [
        ("cuda", lambda: torch.cuda.is_available()),
        ("mps", lambda: bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())),
    ],
)
def test_recurrent_accelerator_training_and_prediction_smoke(device, available):
    if not available():
        pytest.skip(f"{device.upper()} hardware is unavailable")
    model = _TinyRecurrent()
    X = torch.linspace(-1, 1, steps=48).numpy().reshape(8, 3, 2)
    y = X[:, -1, 0] * 0.5 - X[:, -1, 1] * 0.25
    try:
        history, stopping_history, selected_epoch = train_recurrent(
            model,
            X,
            y,
            task="regression",
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
            seed=11,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(X).to(device))
    except RuntimeError as exc:
        message = str(exc).lower()
        unsupported = ("not implemented", "not supported", "deterministic", "mps backend")
        if any(marker in message for marker in unsupported):
            pytest.skip(f"{device.upper()} recurrent operation is unsupported: {exc}")
        raise

    assert len(history) == 2
    assert stopping_history is None
    assert selected_epoch == 2
    assert predictions.device.type == device
    assert predictions.shape == (len(X),)
    assert torch.isfinite(predictions).all()
