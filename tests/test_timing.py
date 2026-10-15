from datetime import datetime, timezone

from lab.core.config import Timeframe
from lab.quant.timing import bar_times_for_session, expected_bar_keys


def test_daily_session_times_follow_dst():
    winter = bar_times_for_session("XNYS", "2024-01-12", Timeframe.D1)[0]
    summer = bar_times_for_session("XNYS", "2024-07-12", Timeframe.D1)[0]
    assert winter[0].hour == 14
    assert summer[0].hour == 13


def test_early_close_and_short_final_hour_are_explicit():
    daily = bar_times_for_session("XNYS", "2024-11-29", Timeframe.D1)
    hourly = bar_times_for_session("XNYS", "2024-11-29", Timeframe.H1)
    assert (daily[0][1] - daily[0][0]).total_seconds() == 3.5 * 60 * 60
    assert hourly[-1][1] == daily[0][1]
    assert (hourly[-1][1] - hourly[-1][0]).total_seconds() == 30 * 60


def test_expected_daily_keys_skip_holiday_and_only_include_completed_sessions():
    expected = expected_bar_keys(
        "XNYS", Timeframe.D1, "2024-01-01T00:00:00+00:00", "2024-01-03T00:00:00+00:00",
        now=datetime(2024, 1, 3, 23, tzinfo=timezone.utc),
    )

    assert expected["session_id"].to_list() == ["2024-01-02"]


def test_expected_daily_keys_do_not_invent_weekend_sessions():
    expected = expected_bar_keys(
        "XNYS", Timeframe.D1, "2024-01-06T00:00:00+00:00", "2024-01-08T23:00:00+00:00",
        now=datetime(2024, 1, 9, 0, tzinfo=timezone.utc),
    )

    assert expected["session_id"].to_list() == ["2024-01-08"]
