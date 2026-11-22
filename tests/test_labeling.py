from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from lab.core.config import PipelineConfig, TaskConfig, Timeframe, TripleBarrierConfig
from lab.core.contracts import MarketSnapshot
from lab.quant.labeling import calculate_volatility, compute_labels, triple_barrier_label


def _create_synthetic_series(prices: list[float], ticker: str = "AAPL") -> pl.DataFrame:
    n = len(prices)
    start_ts = datetime(2025, 1, 1, 10, 0)
    timestamps = [start_ts + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "ticker": [ticker] * n,
            "close": prices,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": [1000.0] * n,
        }
    )


def test_barrier_labels_known_values():
    """Hand-computed test: verify exact +1, -1, and 0 labels on deterministic price path."""
    # 20 warm-up bars at 100, then clear moves
    prices = [100.0] * 20 + [100.0, 115.0, 100.0, 85.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    df = _create_synthetic_series(prices)

    config = PipelineConfig(
        profit_taking=1.0,
        stop_loss=1.0,
        vol_lookback_bars=20,
        expiry_bars=5,
        min_volatility=0.05,  # 5% barrier: upper = 105, lower = 95
    )

    labeled = triple_barrier_label(df, config)
    assert "label" in labeled.columns
    assert "t1" in labeled.columns
    assert "t0" in labeled.columns

    # Row index 20 (price=100.0, next price is 115.0 at step 1) -> should hit upper (+1) at step 1
    row_20 = labeled.filter(pl.col("timestamp") == df["timestamp"][20]).to_dicts()[0]
    assert row_20["label"] == 1
    assert row_20["t1"] == df["timestamp"][21]

    # Row index 22 (price=100.0, next price 85.0 at step 1) -> should hit lower (-1) at step 1
    row_22 = labeled.filter(pl.col("timestamp") == df["timestamp"][22]).to_dicts()[0]
    assert row_22["label"] == -1
    assert row_22["t1"] == df["timestamp"][23]


def test_barrier_expiry_in_bars_not_days():
    """Verify that vertical expiry works strictly in bar count across timeframes."""
    prices = [100.0] * 40
    df = _create_synthetic_series(prices)

    config_10 = PipelineConfig(vol_lookback_bars=20, expiry_bars=10, min_volatility=0.05)
    labeled_10 = triple_barrier_label(df, config_10)

    # For row 0 in labeled (which corresponds to row 0 in df), t1 must be exactly 10 bars ahead
    row_0 = labeled_10.to_dicts()[0]
    assert row_0["label"] == 0
    assert row_0["t1"] == df["timestamp"][10]


def test_barrier_volatility_scaling():
    """Verify barriers automatically scale with volatility levels."""
    # Low volatility sequence vs high volatility sequence
    low_vol_prices = [100.0 + (i % 2) * 0.1 for i in range(30)]
    df_low = _create_synthetic_series(low_vol_prices)

    vol_df = calculate_volatility(df_low, vol_lookback_bars=10, min_volatility=1e-5)
    assert vol_df["volatility_t_minus_1"].mean() < 0.01


def test_barrier_short_group_skipped():
    """Verify ticker groups shorter than vol_lookback + expiry are skipped without raising errors."""
    prices = [100.0] * 10
    df = _create_synthetic_series(prices)
    config = PipelineConfig(vol_lookback_bars=20, expiry_bars=10)

    labeled = triple_barrier_label(df, config)
    assert labeled.is_empty()


def test_barrier_returns_t1():
    """Ensure t1 (first touch timestamp) is properly formatted and strictly greater than t0."""
    prices = [100.0 + i * 0.5 for i in range(50)]
    df = _create_synthetic_series(prices)
    config = PipelineConfig(vol_lookback_bars=20, expiry_bars=10, min_volatility=0.02)

    labeled = triple_barrier_label(df, config)
    assert labeled.height > 0

    t0_list = labeled["t0"].to_list()
    t1_list = labeled["t1"].to_list()
    for t0, t1 in zip(t0_list, t1_list):
        assert t1 > t0


_BASE_OPEN = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)


def _bar_open(idx: int) -> datetime:
    return _BASE_OPEN + timedelta(days=idx)


def _bar_close(idx: int) -> datetime:
    return _bar_open(idx) + timedelta(hours=6, minutes=30)


def _create_compute_labels_snapshot(
    ticker_bars: dict[str, list[tuple[float, float, float, float]]],
) -> MarketSnapshot:
    rows: list[dict[str, object]] = []
    for ticker, ohlc_list in ticker_bars.items():
        for idx, (open_px, high_px, low_px, close_px) in enumerate(ohlc_list):
            opened = _bar_open(idx)
            closed = _bar_close(idx)
            rows.append(
                {
                    "ticker": ticker,
                    "bar_open_time": opened,
                    "bar_close_time": closed,
                    "timestamp": closed,
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "volume": 1000.0,
                }
            )
    return MarketSnapshot(
        snapshot_id="snap-test-labeling",
        snapshot_hash="hash-test-labeling",
        bars=pl.DataFrame(rows),
        ticker_order=tuple(ticker_bars.keys()),
        calendar="XNYS",
        timeframe=Timeframe.D1,
    )


def test_compute_labels_triple_barrier_upper_lower_and_expiry():
    """Verify compute_labels resolves upper, lower, and expiry barriers via close observation and next-open t1."""
    warmup = [(100.0, 101.0, 99.0, 100.0)] * 3
    snapshot = _create_compute_labels_snapshot(
        {
            "UP": [
                *warmup,
                (100.0, 112.0, 99.0, 102.0),  # entry bar: high breaches 105, close does not
                (102.0, 106.0, 101.0, 105.0),  # trigger bar: close == upper (105.0)
                (104.0, 105.0, 90.0, 90.0),  # t1 execution open
                (91.0, 92.0, 90.0, 91.0),  # trailing bar for expiry window
            ],
            "DOWN": [
                *warmup,
                (100.0, 101.0, 90.0, 98.0),  # entry bar: low breaches 95, close does not
                (98.0, 99.0, 94.0, 95.0),  # trigger bar: close == lower (95.0)
                (94.0, 96.0, 93.0, 95.0),  # t1 execution open
                (96.0, 97.0, 95.0, 96.0),  # trailing bar for expiry window
            ],
            "EXPIRY": [
                *warmup,
                (100.0, 108.0, 92.0, 104.0),  # high/low breach barriers, close stays inside
                (104.0, 105.0, 95.0, 96.0),  # close stays inside (95, 105)
                (96.0, 103.0, 95.5, 101.0),  # expiry bar: no close touch
                (102.0, 103.0, 101.0, 102.0),  # t1 execution open after expiry
            ],
        }
    )
    task_config = TaskConfig(
        kind="classification",
        triple_barrier=TripleBarrierConfig(
            profit_taking=1.0,
            stop_loss=1.0,
            expiry_bars=3,
            volatility_span=2,
            volatility_warmup=2,
            volatility_floor=0.05,
        ),
    )

    result = compute_labels(snapshot, task_config)
    assert result.target_name == "label"

    rows_by_ticker = {row["ticker"]: row for row in result.labels.to_dicts()}
    assert set(rows_by_ticker) == {"UP", "DOWN", "EXPIRY"}

    up_row = rows_by_ticker["UP"]
    assert up_row["label"] == 1
    assert up_row["target"] == 1
    assert up_row["volatility"] == pytest.approx(0.05)
    assert up_row["upper_barrier"] == pytest.approx(105.0)
    assert up_row["lower_barrier"] == pytest.approx(95.0)
    assert up_row["decision_time"] == _bar_close(2)
    assert up_row["entry_time"] == _bar_open(3)
    assert up_row["event_end"] == _bar_close(4)
    assert up_row["t1"] == _bar_open(5)

    down_row = rows_by_ticker["DOWN"]
    assert down_row["label"] == -1
    assert down_row["target"] == -1
    assert down_row["upper_barrier"] == pytest.approx(105.0)
    assert down_row["lower_barrier"] == pytest.approx(95.0)
    assert down_row["decision_time"] == _bar_close(2)
    assert down_row["entry_time"] == _bar_open(3)
    assert down_row["event_end"] == _bar_close(4)
    assert down_row["t1"] == _bar_open(5)

    expiry_row = rows_by_ticker["EXPIRY"]
    assert expiry_row["label"] == 0
    assert expiry_row["target"] == 0
    assert expiry_row["upper_barrier"] == pytest.approx(105.0)
    assert expiry_row["lower_barrier"] == pytest.approx(95.0)
    assert expiry_row["decision_time"] == _bar_close(2)
    assert expiry_row["entry_time"] == _bar_open(3)
    assert expiry_row["event_end"] == _bar_close(5)
    assert expiry_row["t1"] == _bar_open(6)

    exclusions_by_ticker: dict[str, list[str]] = {}
    for item in result.exclusions.to_dicts():
        exclusions_by_ticker.setdefault(item["ticker"], []).append(item["reason"])
    assert exclusions_by_ticker == {
        "DOWN": ["warmup", "warmup", "tail", "tail", "tail", "tail"],
        "EXPIRY": ["warmup", "warmup", "tail", "tail", "tail", "tail"],
        "UP": ["warmup", "warmup", "tail", "tail", "tail", "tail"],
    }


def test_compute_labels_anchors_barriers_on_entry_open_and_resolves_immediate_touch():
    """Verify barriers anchor at entry open (not decision close) and immediate close touches resolve at next open."""
    warmup = [(100.0, 101.0, 99.0, 100.0)] * 3
    snapshot = _create_compute_labels_snapshot(
        {
            "GAP_TOUCH": [
                *warmup,
                (120.0, 127.0, 119.0, 126.0),  # entry open=120 -> upper=126; immediate touch at close
                (125.0, 126.0, 124.0, 125.0),  # t1 execution open
                (125.0, 126.0, 124.0, 125.0),
                (125.0, 126.0, 124.0, 125.0),
            ],
        }
    )
    task_config = TaskConfig(
        kind="classification",
        triple_barrier=TripleBarrierConfig(
            profit_taking=1.0,
            stop_loss=1.0,
            expiry_bars=3,
            volatility_span=2,
            volatility_warmup=2,
            volatility_floor=0.05,
        ),
    )

    result = compute_labels(snapshot, task_config)
    row = result.labels.to_dicts()[0]

    assert row["label"] == 1
    assert row["upper_barrier"] == pytest.approx(126.0)
    assert row["lower_barrier"] == pytest.approx(114.0)
    assert row["entry_time"] == _bar_open(3)
    assert row["event_end"] == _bar_close(3)
    assert row["t1"] == _bar_open(4)
