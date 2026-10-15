from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import yaml
from dagster import DagsterInstance, materialize

from lab.core.config import IngestionStreamConfig, Timeframe
from lab.platform.dagster import assets as dagster_assets
from lab.platform.dagster.resources import PipelineConfigResource
from lab.platform.ingestion import IngestionService
from lab.platform.market_store import ProviderBatchStore, SnapshotCatalog
from lab.quant.timing import expected_bar_keys


class _CalendarFixtureProvider:
    """Return only expected calendar bars, with no network/provider access."""

    def __init__(self, stream: IngestionStreamConfig) -> None:
        self.stream = stream
        self.calls = 0

    def timestamp_metadata(self, timeframe: Timeframe) -> dict[str, str]:
        return {"timestamp_role": self.stream.timestamp_role, "timezone": self.stream.provider_timezone}

    def fetch(self, tickers, *, timeframe, start, end):
        self.calls += 1
        expected = expected_bar_keys(
            self.stream.calendar,
            timeframe,
            start,
            end,
            now=end + timedelta(minutes=1),
        )
        rows = []
        for item in expected.to_dicts():
            if timeframe == Timeframe.D1:
                timestamp = datetime.fromisoformat(item["session_id"])
            else:
                timestamp = item["bar_open_time"].astimezone(ZoneInfo(self.stream.provider_timezone)).replace(tzinfo=None)
            for ticker in tickers:
                rows.append({
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 500.0,
                })
        return pl.DataFrame(rows)


def test_active_ingestion_asset_materializes_and_passes_coverage_check_with_fixture_provider(tmp_path, monkeypatch):
    start = (datetime.now(timezone.utc).date() - timedelta(days=10)).isoformat()
    stream = IngestionStreamConfig(
        name="dagster-interface-fixture",
        provider="fixture",
        tickers=("AAA",),
        calendar="XNYS",
        timeframe=Timeframe.D1,
        timestamp_role="session_date",
        provider_timezone="America/New_York",
        start_boundary=f"{start}T00:00:00+00:00",
        refresh_overlap_bars=1,
        completion_delay_minutes=0,
        gap_policy="reject",
    )
    ingestion_config = tmp_path / "ingestion.yaml"
    ingestion_config.write_text(yaml.safe_dump({"streams": [stream.model_dump(mode="json")]}, sort_keys=False))
    provider = _CalendarFixtureProvider(stream)
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    service = IngestionService(
        batch_store=ProviderBatchStore(tmp_path / "batches"),
        catalog=catalog,
        providers={"fixture": provider},
    )
    monkeypatch.setattr(dagster_assets, "create_ingestion_service", lambda _config: service)

    result = materialize(
        [dagster_assets.ingestion_publication, dagster_assets.ingestion_calendar_coverage],
        instance=DagsterInstance.ephemeral(),
        resources={
            "config_py": PipelineConfigResource(ingestion_config_path=str(ingestion_config)),
        },
    )

    assert result.success
    assert provider.calls == 1
    materializations = result.get_asset_materialization_events()
    assert [event.asset_key.to_user_string() for event in materializations] == ["ingestion_publication"]
    evaluations = result.get_asset_check_evaluations()
    assert len(evaluations) == 1
    assert evaluations[0].check_name == "ingestion_calendar_coverage"
    assert evaluations[0].passed is True
    published = catalog.list()
    assert len(published) == 1
    assert published[0].coverage_status == "accepted"
