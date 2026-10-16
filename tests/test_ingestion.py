from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from lab.core.config import IngestionStreamConfig, Timeframe
from lab.platform.ingestion import IngestionPlanner, IngestionService
from lab.platform.market_store import ProviderBatchStore, SnapshotCatalog
from lab.quant.timing import expected_bar_keys


class BrokenProvider:
    def fetch(self, *args, **kwargs):
        raise TimeoutError("simulated provider timeout")


class CalendarProvider:
    def __init__(self, stream, *, omitted_sessions=(), duplicate_first=False, correction_by_call=None):
        self.stream = stream
        self.omitted_sessions = set(omitted_sessions)
        self.duplicate_first = duplicate_first
        self.correction_by_call = correction_by_call or {}
        self.calls = []

    def timestamp_metadata(self, timeframe):
        return {"timestamp_role": self.stream.timestamp_role, "timezone": self.stream.provider_timezone}

    def fetch(self, tickers, *, timeframe, start, end):
        self.calls.append((start, end, timeframe))
        expected = expected_bar_keys(
            self.stream.calendar,
            timeframe,
            start,
            end,
            now=end + timedelta(minutes=1),
        )
        rows = []
        for row in expected.to_dicts():
            if row["session_id"] in self.omitted_sessions:
                continue
            if timeframe == Timeframe.D1:
                timestamp = datetime.fromisoformat(row["session_id"])
            else:
                timestamp = row["bar_open_time"].astimezone(ZoneInfo(self.stream.provider_timezone)).replace(tzinfo=None)
            close = 101.0 + self.correction_by_call.get(len(self.calls), {}).get(row["session_id"], 0.0)
            rows.extend(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": 100.0,
                    "high": max(102.0, 100.0, close),
                    "low": min(99.0, 100.0, close),
                    "close": close,
                    "volume": 500.0,
                }
                for ticker in tickers
            )
        if self.duplicate_first and rows:
            rows.append(dict(rows[0]))
        return pl.DataFrame(rows)


def _stream():
    return IngestionStreamConfig(
        name="daily-test",
        provider="fake",
        tickers=("AAA",),
        calendar="XNYS",
        timeframe=Timeframe.D1,
        start_boundary="2024-01-02T00:00:00+00:00",
        timestamp_role="session_date",
        provider_timezone="America/New_York",
    )


def _calendar_stream(timeframe, *, gap_policy="reject"):
    return IngestionStreamConfig(
        name=f"{timeframe.value}-stream",
        provider="fake",
        tickers=("AAA",),
        calendar="XNYS",
        timeframe=timeframe,
        start_boundary="2024-01-02T00:00:00+00:00",
        refresh_overlap_bars=1,
        completion_delay_minutes=0,
        timestamp_role="session_date" if timeframe == Timeframe.D1 else "bar_open",
        provider_timezone="America/New_York",
        gap_policy=gap_policy,
    )


def test_first_ingestion_plan_is_bounded_by_completed_calendar_bars(tmp_path):
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    plan = IngestionPlanner(catalog).plan(_stream(), now=datetime(2024, 1, 2, 22, 0, tzinfo=timezone.utc))

    assert not plan.no_op
    assert plan.tickers == ("AAA",)
    assert plan.start < plan.end
    assert plan.completed_boundary <= datetime(2024, 1, 2, 22, 0, tzinfo=timezone.utc)


def test_provider_failure_is_recorded_without_snapshot_publication(tmp_path):
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "batches"),
        catalog=catalog,
        providers={"fake": BrokenProvider()},
    )
    result = service.execute(_stream(), now=datetime(2024, 1, 2, 22, 0, tzinfo=timezone.utc))

    assert result.status == "failed"
    assert "simulated provider timeout" in result.error
    assert catalog.list() == []
    assert catalog.attempts("daily-test")[0].attempt_id == result.attempt_id


def test_completed_bar_empty_poll_is_durable_noop(tmp_path):
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "batches"),
        catalog=catalog,
        providers={"fake": BrokenProvider()},
    )
    result = service.execute(_stream(), now=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc))

    assert result.status == "noop"
    assert catalog.attempts("daily-test")[0].status == "noop"


@pytest.mark.parametrize("timeframe", [Timeframe.D1, Timeframe.H1])
def test_daily_and_hourly_ingestion_incrementally_publish_normalized_calendar_bars(tmp_path, timeframe):
    stream = _calendar_stream(timeframe)
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    provider = CalendarProvider(stream)
    service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "batches"),
        catalog=catalog,
        providers={"fake": provider},
    )
    first_time = datetime(2024, 1, 2, 22, tzinfo=timezone.utc)
    second_time = datetime(2024, 1, 3, 22, tzinfo=timezone.utc)

    first = service.execute(stream, now=first_time)
    second = service.execute(stream, now=second_time)
    replay = service.execute(stream, now=second_time)

    assert first.status == second.status == "succeeded"
    assert replay.status == "noop"
    assert len(provider.calls) == 2
    latest = catalog.load(second.snapshot_id)
    expected_rows = 2 if timeframe == Timeframe.D1 else 14
    assert latest.bars.height == expected_rows
    assert latest.bars["timestamp"].dtype.time_zone == "UTC"


def test_incomplete_snapshot_repoll_repairs_known_gap_without_new_bar(tmp_path):
    stream = _calendar_stream(Timeframe.D1, gap_policy="record")
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    provider = CalendarProvider(stream, omitted_sessions=("2024-01-02",))
    service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "batches"),
        catalog=catalog,
        providers={"fake": provider},
    )
    now = datetime(2024, 1, 3, 22, tzinfo=timezone.utc)

    incomplete = service.execute(stream, now=now)
    provider.omitted_sessions.clear()
    repaired = service.execute(stream, now=now)

    assert incomplete.status == "succeeded"
    assert catalog.resolve(incomplete.snapshot_id).coverage_status == "incomplete"
    assert repaired.status == "succeeded"
    assert repaired.missing_rows == 0
    assert catalog.resolve(repaired.snapshot_id).coverage_status == "accepted"
    assert len(provider.calls) == 2


@pytest.mark.parametrize(
    ("gap_policy", "expected_status", "coverage_status"),
    [
        ("reject", "failed", None),
        ("record", "succeeded", "incomplete"),
        ("allow", "succeeded", "accepted"),
    ],
)
def test_gap_policy_controls_publication(gap_policy, expected_status, coverage_status, tmp_path):
    stream = _calendar_stream(Timeframe.D1, gap_policy=gap_policy)
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    provider = CalendarProvider(stream, omitted_sessions=("2024-01-02",))
    service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "batches"),
        catalog=catalog,
        providers={"fake": provider},
    )

    result = service.execute(stream, now=datetime(2024, 1, 3, 22, tzinfo=timezone.utc))

    assert result.status == expected_status
    if coverage_status is None:
        assert catalog.list() == []
    else:
        assert catalog.resolve(result.snapshot_id).coverage_status == coverage_status


def test_duplicate_bars_are_rejected_and_revisions_are_recorded(tmp_path):
    stream = _calendar_stream(Timeframe.D1)
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    duplicate_provider = CalendarProvider(stream, duplicate_first=True)
    duplicate_service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "duplicate-batches"),
        catalog=catalog,
        providers={"fake": duplicate_provider},
    )
    rejected = duplicate_service.execute(stream, now=datetime(2024, 1, 2, 22, tzinfo=timezone.utc))
    assert rejected.status == "failed"
    assert "overlap" in rejected.error.lower()

    revised_provider = CalendarProvider(stream, correction_by_call={2: {"2024-01-02": 5.0}})
    revised_service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "revised-batches"),
        catalog=catalog,
        providers={"fake": revised_provider},
    )
    first = revised_service.execute(stream, now=datetime(2024, 1, 2, 22, tzinfo=timezone.utc))
    revised = revised_service.execute(stream, now=datetime(2024, 1, 3, 22, tzinfo=timezone.utc))

    assert first.status == revised.status == "succeeded"
    assert revised.revised_rows == 1
    bars = catalog.load(revised.snapshot_id).bars
    assert bars.filter(pl.col("session_id") == "2024-01-02")["close"].item() == 106.0
