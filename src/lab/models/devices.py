"""Capability-aware model-device resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, cast

import torch

from lab.core.contracts import DeviceResolution


def resolve_device(
    model: str,
    requested: str = "auto",
) -> DeviceResolution:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported requested device: {requested}")
    requested_device = cast(Literal["auto", "cpu", "mps", "cuda"], requested)
    normalized_model = model.lower()
    if normalized_model in {"baseline", "mean", "median"}:
        if requested_device not in {"auto", "cpu"}:
            raise ValueError(f"{model} supports CPU only")
        return DeviceResolution(
            model=model,
            requested_device=requested_device,
            resolved_device="cpu",
            reason="Baseline adapter has no accelerator implementation",
            resolved_at=datetime.now(timezone.utc),
            available_devices=("cpu",),
        )
    supports_mps = normalized_model in {"gru", "lstm", "recurrent"}
    supports_cuda = normalized_model in {"gru", "lstm", "recurrent", "xgboost"}
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    available = ("cpu",) + (("mps",) if mps_available else ()) + (("cuda",) if cuda_available else ())
    if requested_device == "mps" and not supports_mps:
        raise ValueError(f"{model} does not support MPS")
    if requested_device == "cuda" and not supports_cuda:
        raise ValueError(f"{model} does not support CUDA")
    if requested_device == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is unavailable")
    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested_device != "auto":
        resolved = requested_device
        reason = "Explicit device request"
    elif cuda_available and supports_cuda:
        resolved = "cuda"
        reason = "Auto-selected available CUDA accelerator"
    elif mps_available and supports_mps:
        resolved = "mps"
        reason = "Auto-selected available MPS accelerator"
    else:
        resolved = "cpu"
        reason = "No supported accelerator is available"
    return DeviceResolution(
        model=model,
        requested_device=requested_device,
        resolved_device=resolved,
        reason=reason,
        resolved_at=datetime.now(timezone.utc),
        available_devices=available,
    )
