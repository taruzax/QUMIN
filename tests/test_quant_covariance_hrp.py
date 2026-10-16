import numpy as np
import polars as pl
import pytest

from lab.quant.covariance import estimate_ledoit_wolf
from lab.quant.hrp import hrp_pipe


def test_covariance_and_hrp_use_labeled_raw_returns():
    rng = np.random.default_rng(42)
    returns = pl.DataFrame(
        {
            "timestamp": list(range(300)),
            "AAA": rng.normal(0, 0.01, 300),
            "BBB": rng.normal(0, 0.02, 300),
        }
    )

    covariance, diagnostics = estimate_ledoit_wolf(returns)
    weights = hrp_pipe(returns)

    assert diagnostics.observations == 300
    assert covariance.columns == ["ticker", "AAA", "BBB"]
    assert set(weights) == {"AAA", "BBB"}
    assert sum(weights.values()) == pytest.approx(1.0)
