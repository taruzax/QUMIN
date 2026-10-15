from datetime import datetime, timezone

import polars as pl

from lab.core.config import IngestionConfig, IngestionStreamConfig, Timeframe
import lab.platform.dagster.schedules as schedules
from lab.platform.dagster.schedules import enabled_streams, stable_run_key


def test_ingestion_schedules_select_timeframe_and_use_deterministic_keys():
    daily = IngestionStreamConfig(name="daily", tickers=("AAA",), start_boundary="2024-01-01")
    hourly = IngestionStreamConfig(name="hourly", tickers=("AAA",), start_boundary="2024-01-01", timeframe=Timeframe.H1)
    disabled = IngestionStreamConfig(name="disabled", tickers=("AAA",), start_boundary="2024-01-01", enabled=False)
    config = IngestionConfig(streams=(daily, hourly, disabled))
    boundary = datetime(2024, 1, 2, 21, tzinfo=timezone.utc)

    assert [stream.name for stream in enabled_streams(config, Timeframe.D1)] == ["daily"]
    assert [stream.name for stream in enabled_streams(config, Timeframe.H1)] == ["hourly"]
    assert stable_run_key(daily, boundary) == stable_run_key(daily, boundary)


def test_schedule_request_uses_calendar_completed_bar_boundary(monkeypatch):
    stream = IngestionStreamConfig(name="daily", tickers=("AAA",), start_boundary="2024-01-01")
    config = IngestionConfig(streams=(stream,))
    boundary = datetime(2024, 1, 2, 21, tzinfo=timezone.utc)
    monkeypatch.setattr(schedules.IngestionConfig, "from_yaml", lambda *_args: config)
    monkeypatch.setattr(
        schedules, "expected_bar_keys",
        lambda *_args, **_kwargs: pl.DataFrame({"bar_close_time": [boundary]}),
    )
    context = type("ScheduleContext", (), {"scheduled_execution_time": datetime(2024, 1, 2, 22, tzinfo=timezone.utc)})()

    requests = schedules._schedule_requests(context, Timeframe.D1)

    assert len(requests) == 1
    assert requests[0].run_key == "daily:1d:2024-01-02T21:00:00+00:00"
