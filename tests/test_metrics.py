import numpy as np
import pytest
from exchange_calendars.errors import InvalidCalendarName

from lab.quant.metrics import compound_session_returns, return_convention, statistical_returns


def test_statistical_daily_returns_compound_intraday_bars_without_padding():
    timestamps = [
        "2024-01-02T15:00:00Z",
        "2024-01-02T16:00:00Z",
        "2024-01-03T15:00:00Z",
    ]
    native = statistical_returns(timestamps, [0.01, 0.02, -0.01], frequency="native")
    daily = statistical_returns(timestamps, [0.01, 0.02, -0.01], frequency="daily", calendar="XNYS")

    assert len(native) == 3
    assert len(daily) == 2
    assert daily.iloc[0] == pytest.approx((1.01 * 1.02) - 1.0)
    assert np.prod(1.0 + daily.to_numpy()) == pytest.approx(np.prod(1.0 + native.to_numpy()))
    assert return_convention("daily") == "session-compounded.v1"
    assert return_convention("native") == "native-bars.v1"


def test_daily_convention_uses_the_requested_exchange_calendar():
    timestamps = ["2024-07-03T15:00:00Z", "2024-07-04T15:00:00Z"]
    returns = [0.01, 0.02]

    nyse = compound_session_returns(timestamps, returns, calendar="XNYS")
    hong_kong = compound_session_returns(timestamps, returns, calendar="XHKG")

    assert len(nyse) == 1  # July 4 is closed in New York and maps to the preceding session.
    assert len(hong_kong) == 2
    assert np.prod(1.0 + nyse.to_numpy()) == pytest.approx(np.prod(1.0 + hong_kong.to_numpy()))


def test_overnight_calendar_groups_bars_across_local_midnight():
    timestamps = ["2024-01-02T23:30:00Z", "2024-01-03T07:30:00Z"]
    daily = compound_session_returns(timestamps, [0.01, 0.02], calendar="CMES")

    assert len(daily) == 1
    assert daily.index[0].date().isoformat() == "2024-01-03"
    assert daily.iloc[0] == pytest.approx(1.01 * 1.02 - 1.0)


@pytest.mark.parametrize("frequency", ["weekly", "1D", ""])
def test_unknown_return_frequency_fails_clearly(frequency):
    with pytest.raises(ValueError, match="Unsupported return frequency"):
        statistical_returns(["2024-01-02T15:00:00Z"], [0.01], frequency=frequency)


def test_daily_returns_require_valid_calendar_and_aligned_finite_inputs():
    with pytest.raises(ValueError, match="same length"):
        statistical_returns(["2024-01-02T15:00:00Z"], [0.01, 0.02], frequency="daily")
    with pytest.raises(ValueError, match="finite"):
        statistical_returns(["2024-01-02T15:00:00Z"], [float("nan")], frequency="daily")
    with pytest.raises(InvalidCalendarName):
        statistical_returns(["2024-01-02T15:00:00Z"], [0.01], frequency="daily", calendar="NOT_A_CALENDAR")
