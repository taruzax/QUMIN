from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from lab.core.config import BacktestConfig, TaskConfig, Timeframe
from lab.core.contracts import AllocationFrame, ExecutionPeriod, MarketSnapshot, PredictionFrame, SampleKey
from lab.quant.backtest import simulate_strategy


@pytest.mark.parametrize(("timeframe", "expected_freq"), [(Timeframe.D1, "1D"), (Timeframe.H1, "1h")])
def test_next_open_order_uses_open_price_liquidates_and_preserves_nominal_frequency(monkeypatch, timeframe, expected_freq):
    import lab.quant.backtest as backtest

    original_run_vectorbt = backtest._run_vectorbt
    received_frequencies = []

    def capture_frequency(*args, **kwargs):
        received_frequencies.append(kwargs["freq"])
        return original_run_vectorbt(*args, **kwargs)

    monkeypatch.setattr(backtest, "_run_vectorbt", capture_frequency)
    origin = datetime(2024, 1, 1, 14, tzinfo=timezone.utc)
    opens = [origin + timedelta(days=index) for index in range(4)]
    closes = [value + timedelta(hours=6) for value in opens]
    bars = pl.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "bar_open_time": opens,
            "timestamp": closes,
            "open": [100.0, 110.0, 120.0, 130.0],
            "close": [100.0, 115.0, 125.0, 135.0],
        }
    )
    snapshot = MarketSnapshot(
        snapshot_id="fixture",
        snapshot_hash="fixture",
        bars=bars,
        ticker_order=("AAA",),
        calendar="XNYS",
        timeframe=timeframe,
    )
    prediction = PredictionFrame(
        keys=(SampleKey(ticker="AAA", timestamp=closes[0]),),
        model_ref="fixture",
        fold_ref="fixture",
        predictions=np.array([0.5]),
    )
    allocation = AllocationFrame(
        decision_time=closes[0],
        valid_from=opens[1],
        ticker_budgets={"AAA": 0.2},
        config_hash="fixture",
    )
    period = ExecutionPeriod(
        name="fixture",
        first_decision=closes[0],
        last_decision=closes[2],
        final_liquidation_open=opens[3],
    )
    result = simulate_strategy(
        snapshot,
        prediction,
        (allocation,),
        TaskConfig(kind="regression"),
        BacktestConfig(initial_capital=1_000.0, fees_bps=0.0),
        period,
    )
    assert result.orders.height == 2
    assert result.orders["price"].to_list() == [110.0, 120.0]
    assert result.orders["quantity"][0] == pytest.approx(200.0 / 110.0)
    assert result.equity["equity"][-1] == pytest.approx(1_000.0 + (200.0 / 110.0) * 10.0)
    assert result.orders["reason"].to_list()[-1] == "exit"
    assert result.exposure["AAA"][-1] == pytest.approx(0.0)
    assert received_frequencies == [expected_freq, expected_freq]


def test_prices_after_execution_period_cannot_change_trade_keys_or_return_path():
    origin = datetime(2024, 1, 1, 14, tzinfo=timezone.utc)
    opens = [origin + timedelta(days=index) for index in range(5)]
    closes = [value + timedelta(hours=6) for value in opens]
    prediction = PredictionFrame(
        keys=(SampleKey(ticker="AAA", timestamp=closes[0]),),
        model_ref="fixture",
        fold_ref="0",
        predictions=np.asarray([0.5]),
    )
    allocation = AllocationFrame(
        decision_time=closes[0],
        valid_from=opens[1],
        ticker_budgets={"AAA": 0.2},
        config_hash="fixture",
    )
    period = ExecutionPeriod(
        name="development",
        first_decision=closes[0],
        last_decision=closes[0],
        final_liquidation_open=opens[3],
    )

    def simulate(prices):
        bars = pl.DataFrame(
            {
                "ticker": ["AAA"] * len(opens),
                "bar_open_time": opens,
                "timestamp": closes,
                "open": prices,
                "close": prices,
            }
        )
        snapshot = MarketSnapshot(
            snapshot_id="causal-fixture",
            snapshot_hash="causal-fixture",
            bars=bars,
            ticker_order=("AAA",),
            calendar="XNYS",
            timeframe=Timeframe.D1,
        )
        return simulate_strategy(
            snapshot,
            prediction,
            (allocation,),
            TaskConfig(kind="regression"),
            BacktestConfig(initial_capital=1_000.0, fees_bps=0.0),
            period,
        )

    baseline = simulate([100.0, 110.0, 120.0, 130.0, 140.0])
    changed_future = simulate([100.0, 110.0, 120.0, 130.0, 9_999.0])

    assert baseline.orders.select("timestamp", "ticker", "side", "reason").equals(
        changed_future.orders.select("timestamp", "ticker", "side", "reason")
    )
    assert baseline.equity["equity"].to_list() == changed_future.equity["equity"].to_list()


def _run_allocations(tickers, allocation_specs, *, capital=1_000.0, fees_bps=0.0, prices=None):
    origin = datetime(2024, 1, 1, 14, tzinfo=timezone.utc)
    opens = [origin + timedelta(days=index) for index in range(4)]
    closes = [value + timedelta(hours=6) for value in opens]
    prices = prices or [100.0] * len(opens)
    bars = pl.DataFrame(
        {
            "ticker": [ticker for ticker in tickers for _ in opens],
            "bar_open_time": opens * len(tickers),
            "timestamp": closes * len(tickers),
            "open": prices * len(tickers),
            "close": prices * len(tickers),
        }
    )
    snapshot = MarketSnapshot(
        snapshot_id="execution-fixture",
        snapshot_hash="execution-fixture",
        bars=bars,
        ticker_order=tuple(tickers),
        calendar="XNYS",
        timeframe=Timeframe.D1,
    )
    keys, predictions, allocations = [], [], []
    for decision_index, budgets, directions, diagnostics in allocation_specs:
        for ticker in tickers:
            keys.append(SampleKey(ticker=ticker, timestamp=closes[decision_index]))
            predictions.append(float(directions.get(ticker, 1)))
        allocations.append(
            AllocationFrame(
                decision_time=closes[decision_index],
                valid_from=opens[decision_index + 1],
                ticker_budgets=budgets,
                config_hash="fixture",
                diagnostics=diagnostics,
            )
        )
    prediction = PredictionFrame(
        keys=tuple(keys),
        model_ref="fixture",
        fold_ref="fixture",
        predictions=np.asarray(predictions),
    )
    last_decision = closes[max(spec[0] for spec in allocation_specs)]
    period = ExecutionPeriod(
        name="fixture",
        first_decision=closes[0],
        last_decision=last_decision,
        final_liquidation_open=opens[3],
    )
    return simulate_strategy(
        snapshot,
        prediction,
        tuple(allocations),
        TaskConfig(kind="regression"),
        BacktestConfig(initial_capital=capital, fees_bps=fees_bps),
        period,
    )


def test_terminal_liquidation_closes_position_at_final_open():
    result = _run_allocations(
        ("AAA",),
        [
            (0, {"AAA": 0.2}, {"AAA": 1}, {}),
            (1, {"AAA": 0.2}, {"AAA": 1}, {}),
            (2, {"AAA": 0.2}, {"AAA": 1}, {}),
        ],
        prices=[100.0, 110.0, 120.0, 130.0],
    )

    terminal_order = result.orders.row(-1, named=True)
    assert terminal_order["reason"] == "terminal_liquidation"
    assert terminal_order["price"] == 130.0
    assert result.exposure["AAA"][-1] == pytest.approx(0.0)


def test_negative_delta_sells_before_alphabetically_earlier_new_buy():
    result = _run_allocations(
        ("AAA", "ZZZ"),
        [
            (0, {"ZZZ": 0.9}, {"ZZZ": 1}, {}),
            (1, {"AAA": 0.9, "ZZZ": 0.0}, {"AAA": 1, "ZZZ": 1}, {}),
        ],
    )
    second_rebalance = result.orders.filter(pl.col("timestamp") == datetime(2024, 1, 3, 14, tzinfo=timezone.utc))
    assert second_rebalance["ticker"].to_list() == ["ZZZ", "AAA"]
    assert second_rebalance["quantity"].to_list() == pytest.approx([-9.0, 9.0])


@pytest.mark.parametrize(
    ("weight", "expected_quantity"),
    [(0.99, 9.9), (1.0, 1_000.0 / (100.0 * 1.01))],
)
def test_fee_bearing_orders_fit_exact_and_near_cash_boundaries(weight, expected_quantity):
    result = _run_allocations(
        ("AAA",),
        [(0, {"AAA": weight}, {"AAA": 1}, {"gross_limit": 2.0})],
        fees_bps=100.0,
    )
    entry = result.orders.filter(pl.col("reason") == "entry").row(0, named=True)
    assert entry["quantity"] == pytest.approx(expected_quantity, rel=1e-10)
    assert entry["quantity"] * entry["price"] + entry["fees"] <= 1_000.0


def test_long_exit_short_cover_and_new_entry_share_one_cash_safe_rebalance():
    result = _run_allocations(
        ("LONG", "NEW", "SHORT"),
        [
            (0, {"LONG": 0.5, "SHORT": 0.5}, {"LONG": 1, "SHORT": -1}, {}),
            (1, {"LONG": 0.0, "NEW": 0.5, "SHORT": 0.0}, {"LONG": 1, "NEW": 1, "SHORT": 1}, {}),
        ],
    )
    rebalance_time = datetime(2024, 1, 3, 14, tzinfo=timezone.utc)
    rebalance = result.orders.filter(pl.col("timestamp") == rebalance_time)
    assert rebalance["ticker"].to_list() == ["LONG", "NEW", "SHORT"]
    assert rebalance["quantity"].to_list() == pytest.approx([-5.0, 5.0, 5.0])


@pytest.mark.parametrize(
    ("initial_direction", "next_direction", "expected_tickers", "expected_quantities"),
    [
        (1, -1, ["ZZZ", "AAA"], [-10.0, 2.5]),
        (-1, 1, ["AAA", "ZZZ"], [2.5, 10.0]),
    ],
)
def test_reversals_are_ordered_by_their_actual_delta(
    initial_direction,
    next_direction,
    expected_tickers,
    expected_quantities,
):
    result = _run_allocations(
        ("AAA", "ZZZ"),
        [
            (0, {"ZZZ": 0.5}, {"ZZZ": initial_direction}, {}),
            (1, {"AAA": 0.25, "ZZZ": 0.5}, {"AAA": 1, "ZZZ": next_direction}, {}),
        ],
    )
    rebalance = result.orders.filter(pl.col("timestamp") == datetime(2024, 1, 3, 14, tzinfo=timezone.utc))
    assert rebalance["ticker"].to_list() == expected_tickers
    assert rebalance["quantity"].to_list() == pytest.approx(expected_quantities)
    assert "reversal" in rebalance["reason"].to_list()


def test_planned_cash_stays_nonnegative_and_vectorbt_fills_requested_orders():
    result = _run_allocations(
        ("AAA", "ZZZ"),
        [
            (0, {"ZZZ": 0.9}, {"ZZZ": 1}, {}),
            (1, {"AAA": 0.9, "ZZZ": 0.0}, {"AAA": 1, "ZZZ": 1}, {}),
        ],
        fees_bps=50.0,
    )
    cash = 1_000.0
    for order in result.orders.iter_rows(named=True):
        cash -= order["quantity"] * order["price"] + order["fees"]
        assert cash >= -1e-9
    assert result.diagnostics["engine"] == "vectorbt"
    assert result.orders.height == result.trade_events.height
