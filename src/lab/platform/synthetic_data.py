from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from lab.core.config import Timeframe
from lab.quant.timing import bar_times_for_session, get_exchange_calendar

DEMO_TICKERS = ("ALFA", "BRAV", "CHAR", "DELT", "ECHO")
DEMO_SEED = 42
DEMO_GENERATOR_VERSION = "synthetic-demo-v1"
HOURLY_DEMO_GENERATOR_VERSION = "synthetic-demo-hourly-v1"
NULL_CONTROL_GENERATOR_VERSION = "phase1-null-control-v1"


def _generate_hourly_demo_bars(sessions, *, tickers: tuple[str, ...], seed: int, calendar: str) -> pl.DataFrame:
    session_bars = [bar_times_for_session(calendar, session, Timeframe.H1) for session in sessions]
    total_bars = sum(len(items) for items in session_bars)
    rng = np.random.default_rng(seed)
    common_factors = rng.normal(0.0002, 0.008, size=total_bars)
    rows: list[dict[str, Any]] = []
    for asset_index, ticker in enumerate(tickers):
        idiosyncratic = rng.normal(0.0, 0.006, size=total_bars)
        prices = 80.0 + asset_index * 15.0
        bar_index = 0
        for bars in session_bars:
            overnight = rng.normal(0.0, 0.002)
            for interval_index, (opened, closed, session_id) in enumerate(bars):
                opened_price = prices * (1.0 + overnight if interval_index == 0 else 1.0)
                close_return = common_factors[bar_index] * (0.5 + asset_index * 0.1) + idiosyncratic[bar_index]
                closed_price = opened_price * (1.0 + close_return)
                spread = abs(rng.normal(0.002, 0.001))
                rows.append(
                    {
                        "ticker": ticker,
                        "bar_open_time": opened,
                        "bar_close_time": closed,
                        "timestamp": closed,
                        "open": round(float(opened_price), 6),
                        "high": round(float(max(opened_price, closed_price) * (1.0 + spread)), 6),
                        "low": round(float(min(opened_price, closed_price) * (1.0 - spread)), 6),
                        "close": round(float(closed_price), 6),
                        "volume": round(float(rng.lognormal(12.0, 0.15)), 3),
                        "session_id": session_id,
                    }
                )
                prices = closed_price
                bar_index += 1
    return pl.DataFrame(rows).sort(["ticker", "bar_open_time"])


def generate_demo_bars(
    *,
    n_sessions: int = 360,
    tickers: tuple[str, ...] = DEMO_TICKERS,
    seed: int = DEMO_SEED,
    calendar: str = "XNYS",
    timeframe: Timeframe = Timeframe.D1,
) -> pl.DataFrame:
    """Generate deterministic correlated demonstration bars with explicit UTC times."""
    if n_sessions < 300:
        raise ValueError("The bundled demonstration requires at least 300 sessions")
    exchange_calendar = get_exchange_calendar(calendar)
    end_session = pd.Timestamp("2024-12-31")
    sessions = exchange_calendar.sessions[exchange_calendar.sessions <= end_session][-n_sessions:]
    if timeframe == Timeframe.H1:
        return _generate_hourly_demo_bars(sessions, tickers=tickers, seed=seed, calendar=calendar)
    rng = np.random.default_rng(seed)
    common_factor = rng.normal(0.0002, 0.008, size=n_sessions)
    rows: list[dict[str, Any]] = []
    for asset_index, ticker in enumerate(tickers):
        idiosyncratic = rng.normal(0.0, 0.006, size=n_sessions)
        prices = 80.0 + asset_index * 15.0
        for row_index, session in enumerate(sessions):
            opened, closed, session_id = bar_times_for_session(calendar, session, timeframe)[0]
            overnight = rng.normal(0.0, 0.002)
            opened_price = prices * (1.0 + overnight)
            close_return = common_factor[row_index] * (0.5 + asset_index * 0.1) + idiosyncratic[row_index]
            closed_price = opened_price * (1.0 + close_return)
            spread = abs(rng.normal(0.002, 0.001))
            high = max(opened_price, closed_price) * (1.0 + spread)
            low = min(opened_price, closed_price) * (1.0 - spread)
            rows.append(
                {
                    "ticker": ticker,
                    "bar_open_time": opened,
                    "bar_close_time": closed,
                    "timestamp": closed,
                    "open": round(float(opened_price), 6),
                    "high": round(float(high), 6),
                    "low": round(float(low), 6),
                    "close": round(float(closed_price), 6),
                    "volume": round(float(rng.lognormal(12.0, 0.15)), 3),
                    "session_id": session_id,
                }
            )
            prices = closed_price
    return pl.DataFrame(rows).sort(["ticker", "bar_open_time"])


def demo_metadata(
    *,
    seed: int = DEMO_SEED,
    calendar: str = "XNYS",
    timeframe: Timeframe = Timeframe.D1,
) -> dict[str, Any]:
    return {
        "dataset": "synthetic demonstration",
        "generator_version": HOURLY_DEMO_GENERATOR_VERSION if timeframe == Timeframe.H1 else DEMO_GENERATOR_VERSION,
        "seed": seed,
        "calendar": calendar,
        "timeframe": timeframe.value,
        "timestamp_role": "bar_close",
        "timezone": "UTC",
        "price_adjustment": "unadjusted",
        "provenance": "Generated common factor plus independent asset noise; not market evidence.",
    }


def generate_null_control_bars(
    *, n_sessions: int = 360, tickers: tuple[str, ...] = DEMO_TICKERS, seed: int = 20261108, calendar: str = "XNYS"
) -> pl.DataFrame:
    """Generate independent open-to-open innovations with no predictive OHLCV signal."""
    if n_sessions < 120:
        raise ValueError("The Phase 1 null control requires at least 120 sessions")
    exchange = get_exchange_calendar(calendar)
    sessions = exchange.sessions[exchange.sessions <= pd.Timestamp("2024-12-31")][-n_sessions:]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for ticker_index, ticker in enumerate(tickers):
        innovations = rng.normal(0.0, 0.012, size=n_sessions + 1)
        close_noise = rng.normal(0.0, 0.003, size=n_sessions)
        volumes = rng.lognormal(12.0, 0.25, size=n_sessions)
        prices = (80.0 + ticker_index * 15.0) * np.cumprod(1.0 + innovations)
        for index, session in enumerate(sessions):
            opened, closed, session_id = bar_times_for_session(calendar, session, Timeframe.D1)[0]
            open_price = float(prices[index])
            close_price = open_price * (1.0 + float(close_noise[index]))
            spread = abs(float(rng.normal(0.002, 0.0005)))
            rows.append(
                {
                    "ticker": ticker,
                    "bar_open_time": opened,
                    "bar_close_time": closed,
                    "timestamp": closed,
                    "open": round(open_price, 6),
                    "high": round(max(open_price, close_price) * (1.0 + spread), 6),
                    "low": round(min(open_price, close_price) * (1.0 - spread), 6),
                    "close": round(float(close_price), 6),
                    "volume": round(float(volumes[index]), 3),
                    "session_id": session_id,
                }
            )
    return pl.DataFrame(rows).sort(["ticker", "bar_open_time"])


def null_control_metadata(*, seed: int = 20261108, calendar: str = "XNYS", n_sessions: int = 360) -> dict[str, Any]:
    return {
        "dataset": "Phase 1 null control fixture",
        "generator_version": NULL_CONTROL_GENERATOR_VERSION,
        "seed": seed,
        "parameters": {"n_sessions": n_sessions, "open_innovation_std": 0.012, "close_noise_std": 0.003},
        "calendar": calendar,
        "timeframe": Timeframe.D1.value,
        "timestamp_role": "bar_close",
        "timezone": "UTC",
        "price_adjustment": "unadjusted",
        "target_definition": "next-open to following-open simple return",
        "provenance": "Independent zero-mean innovations; not market-performance evidence.",
    }


def generate_positive_control_bars(
    *,
    n_sessions: int = 360,
    tickers: tuple[str, ...] = DEMO_TICKERS,
    seed: int = 20261109,
    calendar: str = "XNYS",
    signal_return: float = 0.012,
    noise_std: float = 0.003,
) -> pl.DataFrame:
    """Generate a volume-encoded signal for the exact next-open-to-following-open target."""
    if n_sessions < 120 or signal_return <= 0.0 or noise_std <= 0.0:
        raise ValueError("Positive control requires at least 120 sessions and positive signal/noise scales")
    exchange = get_exchange_calendar(calendar)
    sessions = exchange.sessions[exchange.sessions <= pd.Timestamp("2024-12-31")][-n_sessions:]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for ticker_index, ticker in enumerate(tickers):
        signal = rng.choice((-1.0, 1.0), size=n_sessions)
        noise = rng.normal(0.0, noise_std, size=n_sessions)
        opens = np.empty(n_sessions, dtype=float)
        opens[0] = 80.0 + ticker_index * 15.0
        for index in range(1, n_sessions):
            planted = signal_return * signal[index - 2] if index >= 2 else 0.0
            opens[index] = opens[index - 1] * (1.0 + planted + noise[index])
        close_noise = rng.normal(0.0, 0.002, size=n_sessions)
        for index, session in enumerate(sessions):
            opened, closed, session_id = bar_times_for_session(calendar, session, Timeframe.D1)[0]
            open_price = float(opens[index])
            close_price = open_price * (1.0 + float(close_noise[index]))
            volume = float(np.exp(12.0 + 0.35 * signal[index]))
            spread = abs(float(rng.normal(0.002, 0.0005)))
            rows.append(
                {
                    "ticker": ticker,
                    "bar_open_time": opened,
                    "bar_close_time": closed,
                    "timestamp": closed,
                    "open": round(open_price, 6),
                    "high": round(max(open_price, close_price) * (1.0 + spread), 6),
                    "low": round(min(open_price, close_price) * (1.0 - spread), 6),
                    "close": round(float(close_price), 6),
                    "volume": round(volume, 3),
                    "session_id": session_id,
                }
            )
    return pl.DataFrame(rows).sort(["ticker", "bar_open_time"])


def write_demo_dataset(
    data_path: Path | str,
    metadata_path: Path | str,
    *,
    n_sessions: int = 360,
    seed: int = DEMO_SEED,
) -> tuple[Path, Path]:
    """Write the committed offline fixture from deterministic generator inputs."""
    data_path = Path(data_path)
    metadata_path = Path(metadata_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_bars(n_sessions=n_sessions, seed=seed).write_csv(data_path)
    metadata_path.write_text(yaml.safe_dump(demo_metadata(seed=seed), sort_keys=False))
    return data_path, metadata_path
