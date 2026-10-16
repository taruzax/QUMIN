import os
import signal
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pytest
import torch

import lab.models.xgboost_model as xgboost_module
from lab.core.contracts import WindowSet
from lab.models.registry import create_model
from lab.models.xgboost_model import XGBoostModel


def _windows(rows: int = 24, steps: int = 3, features: int = 2) -> np.ndarray:
    values = np.arange(rows * steps * features, dtype=np.float32)
    return values.reshape(rows, steps, features) / 100.0


def test_flatten_preserves_oldest_to_newest_feature_order():
    model = XGBoostModel(task="regression", n_estimators=2)
    windows = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    flattened = model._prepare_features(windows, fitting=True)

    assert flattened.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_regression_fit_predict_and_save_reload(tmp_path):
    X = _windows()
    y = X[:, -1, 0] * 2.0 - X[:, 0, 1]
    model = XGBoostModel(task="regression", n_estimators=4, max_depth=2)

    model.fit(X, y)
    predictions = model.predict(X)
    restored = XGBoostModel.load(model.save(tmp_path / "regression"))

    assert predictions.shape == (len(X),)
    np.testing.assert_allclose(predictions, restored.predict(X))


def test_classification_expands_noncontiguous_classes_to_fixed_order():
    X = _windows(rows=18)
    y = np.asarray([-1, 1] * 9, dtype=np.int8)
    model = XGBoostModel(task="classification", n_estimators=4, max_depth=2)

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert model.classes_seen.tolist() == [-1, 1]
    assert probabilities.shape == (len(X), 3)
    assert np.allclose(probabilities[:, 1], 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_classification_one_class_uses_constant_predictor():
    X = _windows(rows=8)
    model = XGBoostModel(task="classification", n_estimators=4)

    model.fit(X, np.ones(len(X), dtype=np.int8))
    probabilities = model.predict_proba(X)

    assert model.adapter_mode == "constant_class"
    assert np.allclose(probabilities[:, 2], 1.0)
    assert np.allclose(model.predict(X), 1)


def test_registry_creates_xgboost_lazily():
    model = create_model("xgboost", task="regression", params={"n_estimators": 2})

    assert isinstance(model, XGBoostModel)
    assert model.n_estimators == 2


def test_xgboost_explicitly_materializes_window_set_before_flattening(monkeypatch):
    block = np.arange(12, dtype=np.float32).reshape(6, 2)
    windows = WindowSet([block], np.zeros(4, dtype=np.int64), np.arange(4), 3)
    original = WindowSet.to_numpy
    materializations = []

    def track_materialization(self):
        materializations.append(len(self))
        return original(self)

    monkeypatch.setattr(WindowSet, "to_numpy", track_materialization)
    flattened = XGBoostModel(n_estimators=2)._prepare_features(windows, fitting=True)

    assert flattened.shape == (4, 6)
    assert materializations == [4]


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("best_iteration", [1, None])
def test_xgboost_stopping_native_accesses_stay_within_openmp_limit(monkeypatch, task, best_iteration):
    active = False

    @contextmanager
    def openmp_limit(*, limits, user_api):
        nonlocal active
        assert limits == 1
        assert user_api == "openmp"
        active = True
        try:
            yield
        finally:
            active = False

    class FakeEstimator:
        def __init__(self, **_kwargs):
            self.fit_called = False

        def fit(self, _features, _targets, **_kwargs):
            assert active
            self.fit_called = True

        def evals_result(self):
            assert active
            return {"validation_0": {"loss": [1.0, 0.5]}, "validation_1": {"loss": [1.2, 0.7]}}

        @property
        def best_iteration(self):
            assert active
            return best_iteration

    monkeypatch.setattr(xgboost_module, "threadpool_limits", openmp_limit)
    monkeypatch.setattr(xgboost_module, "XGBClassifier", FakeEstimator)
    monkeypatch.setattr(xgboost_module, "XGBRegressor", FakeEstimator)
    features = _windows(rows=12)
    targets = (
        np.tile(np.asarray([-1, 0, 1], dtype=np.int8), 4)
        if task == "classification"
        else np.linspace(-1.0, 1.0, len(features), dtype=np.float32)
    )
    model = XGBoostModel(task=task, n_estimators=5)

    model.fit(features, targets, stopping_data=(features, targets), stopping_patience=2)

    assert model._estimator.fit_called
    assert model.selected_duration == (best_iteration + 1 if best_iteration is not None else 5)


def test_xgboost_and_gru_training_are_stable_when_interleaved(tmp_path):
    script = """
import importlib.metadata
import json
import numpy as np
import lab.models
import torch
import platform
import sys
from threadpoolctl import threadpool_info
from lab.models.recurrent import GRUModel
from lab.models.xgboost_model import XGBoostModel

from pathlib import Path

print('NATIVE_ENVIRONMENT=' + json.dumps({
    'platform': platform.platform(),
    'python': sys.version,
    'packages': {name: importlib.metadata.version(name) for name in ('torch', 'xgboost', 'threadpoolctl')},
    'threadpools': threadpool_info(),
}), flush=True)

rng = np.random.default_rng(42)
windows = rng.normal(size=(256, 30, 3)).astype(np.float32)
gru_targets = np.tile(np.asarray([-1, 0, 1], dtype=np.int8), 86)[:256]
xgb_features = rng.normal(size=(2000, 12)).astype(np.float32)
xgb_targets = np.digitize(xgb_features[:, 0] - xgb_features[:, 1], [-0.5, 0.5]).astype(np.int8) - 1

def native_operations():
    for iteration in range(3):
        xgb = XGBoostModel(task='classification', n_estimators=4, max_depth=2)
        xgb.fit(
            xgb_features,
            xgb_targets,
            stopping_data=(xgb_features[:600], xgb_targets[:600]),
            stopping_patience=2,
        )
        assert 1 <= xgb.selected_duration <= 4

        gru = GRUModel(task='classification', epochs=1, batch_size=32, hidden_size=8, seed=42 + iteration)
        gru.fit(windows, gru_targets)
        assert gru.predict(windows).shape == (256,)
        assert gru.predict_proba(windows).shape == (256, 3)

        assert xgb.predict(xgb_features).shape == (2000,)
        assert xgb.predict_proba(xgb_features).shape == (2000, 3)
        copied = xgb.copy()
        assert copied.predict(xgb_features).shape == (2000,)
        save_path = Path(r'{save_path}') / str(iteration)
        xgb.save(save_path)
        restored = XGBoostModel.load(save_path)
        assert restored.predict(xgb_features).shape == (2000,)

native_operations()
print('DIRECT_NATIVE_INTERLEAVING_OK')
""".replace("{save_path}", str(tmp_path / "xgboost-model"))
    environment = os.environ.copy()
    environment.pop("OMP_NUM_THREADS", None)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"Native interleaving timed out after 180 seconds; stdout={exc.stdout!r}; stderr={exc.stderr!r}"
        ) from exc

    signal_name = signal.Signals(-result.returncode).name if result.returncode < 0 else None
    assert result.returncode == 0, (
        f"exit_code={result.returncode}, signal={signal_name}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "DIRECT_NATIVE_INTERLEAVING_OK" in result.stdout
    assert "NATIVE_ENVIRONMENT=" in result.stdout


def test_campaign_worker_runs_native_xgboost_and_gru_operations_in_isolation(tmp_path):
    script = """
import importlib.metadata
import json
import numpy as np
import lab.models
import torch
import platform
import sys
import os
from pathlib import Path
from types import SimpleNamespace
from threadpoolctl import threadpool_info
from lab.models.recurrent import GRUModel
from lab.models.xgboost_model import XGBoostModel
import lab.research.campaign as campaign

print('NATIVE_ENVIRONMENT=' + json.dumps({
    'platform': platform.platform(),
    'python': sys.version,
    'packages': {name: importlib.metadata.version(name) for name in ('torch', 'xgboost', 'threadpoolctl')},
    'threadpools': threadpool_info(),
}), flush=True)

rng = np.random.default_rng(42)
windows = rng.normal(size=(256, 30, 3)).astype(np.float32)
gru_targets = np.tile(np.asarray([-1, 0, 1], dtype=np.int8), 86)[:256]
xgb_features = rng.normal(size=(2000, 12)).astype(np.float32)
xgb_targets = np.digitize(xgb_features[:, 0] - xgb_features[:, 1], [-0.5, 0.5]).astype(np.int8) - 1

def native_operations(root):
    for iteration in range(3):
        xgb = XGBoostModel(task='classification', n_estimators=4, max_depth=2)
        xgb.fit(
            xgb_features,
            xgb_targets,
            stopping_data=(xgb_features[:600], xgb_targets[:600]),
            stopping_patience=2,
        )
        assert 1 <= xgb.selected_duration <= 4

        gru = GRUModel(task='classification', epochs=1, batch_size=32, hidden_size=8, seed=42 + iteration)
        gru.fit(windows, gru_targets)
        assert gru.predict(windows).shape == (256,)
        assert gru.predict_proba(windows).shape == (256, 3)

        assert xgb.predict(xgb_features).shape == (2000,)
        assert xgb.predict_proba(xgb_features).shape == (2000, 3)
        copied = xgb.copy()
        assert copied.predict(xgb_features).shape == (2000,)
        save_path = root / str(iteration)
        xgb.save(save_path)
        restored = XGBoostModel.load(save_path)
        assert restored.predict(xgb_features).shape == (2000,)

def experiment_callback(_config, *, tracking_mode):
    assert tracking_mode == 'deferred'
    assert os.environ.get('OMP_NUM_THREADS') == '1'
    assert torch.get_num_threads() == 1
    openmp_pools = [pool for pool in threadpool_info() if pool.get('user_api') == 'openmp']
    assert openmp_pools and all(pool['num_threads'] == 1 for pool in openmp_pools), openmp_pools
    native_operations(Path(r'{artifact_dir}'))
    print('CAMPAIGN_CALLBACK_NATIVE_OPERATIONS_OK', flush=True)
    return SimpleNamespace(ref=SimpleNamespace(run_id='native-run'))

campaign._variation_config = lambda *_args: object()
campaign.run_experiment = experiment_callback
variation = SimpleNamespace(variation_id='native-variation')
result = campaign._execute_one(object(), variation)
assert result['status'] == 'completed', result
assert result['variation_id'] == 'native-variation'
assert result['run_id'] == 'native-run'
print('CAMPAIGN_WORKER_NATIVE_INTERLEAVING_OK')
""".replace("{artifact_dir}", str(tmp_path / "campaign-worker-native-operations"))
    environment = os.environ.copy()
    environment.pop("OMP_NUM_THREADS", None)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"Campaign-worker native interleaving timed out after 180 seconds; stdout={exc.stdout!r}; stderr={exc.stderr!r}"
        ) from exc

    signal_name = signal.Signals(-result.returncode).name if result.returncode < 0 else None
    assert result.returncode == 0, (
        f"exit_code={result.returncode}, signal={signal_name}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "NATIVE_ENVIRONMENT=" in result.stdout
    assert "CAMPAIGN_CALLBACK_NATIVE_OPERATIONS_OK" in result.stdout
    assert "CAMPAIGN_WORKER_NATIVE_INTERLEAVING_OK" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_xgboost_training_and_prediction_smoke_in_isolated_process():
    script = """
import numpy as np
from lab.models.xgboost_model import XGBoostModel
X = np.arange(48, dtype=np.float32).reshape(24, 2)
y = X[:, 0] * 0.1 - X[:, 1] * 0.05
model = XGBoostModel(task='regression', n_estimators=2, max_depth=2, device='cuda')
model.fit(X, y)
predictions = model.predict(X)
assert predictions.shape == (24,)
assert np.isfinite(predictions).all()
print('CUDA_XGBOOST_SMOKE_OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}".lower()
    unsupported = (
        "not compiled with gpu",
        "no visible gpu",
        "cuda driver version is insufficient",
        "cuda support is not enabled",
    )
    if result.returncode != 0 and any(message in output for message in unsupported):
        pytest.skip(f"Installed XGBoost build cannot train on the available CUDA device: {output[-1200:]}")
    assert result.returncode == 0, output[-3000:]
    assert "cuda_xgboost_smoke_ok" in result.stdout.lower()
