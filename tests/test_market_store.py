from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Barrier

import polars as pl
import pytest

from lab.core.config import Timeframe
from lab.platform.data_access import build_market_snapshot
from lab.platform.market_store import ProviderBatchStore, SnapshotCatalog
from lab.quant.timing import bar_times_for_session


def _snapshot():
    opened, closed, session_id = bar_times_for_session("XNYS", "2024-01-02", Timeframe.D1)[0]
    bars = pl.DataFrame(
        {
            "ticker": ["AAA"],
            "raw_bar_index": [0],
            "bar_open_time": [opened],
            "bar_close_time": [closed],
            "timestamp": [closed],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [500.0],
            "session_id": [session_id],
        }
    )
    return build_market_snapshot([bars], calendar="XNYS", timeframe=Timeframe.D1, provenance={"source": "test"})


def test_provider_batch_is_idempotent_and_checksum_verified(tmp_path):
    store = ProviderBatchStore(tmp_path / "batches")
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"],
            "timestamp": [date(2024, 1, 2)],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        }
    )
    first = store.store(frame, provider="test", stream_id="daily", request={"start": "2024-01-02"})
    repeated = store.store(frame, provider="test", stream_id="daily", request={"start": "2024-01-02"})

    assert first["batch_id"] == repeated["batch_id"]
    assert store.load(first["batch_id"]).height == 1
    (tmp_path / "batches" / "batches" / first["batch_id"] / "bars.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.load(first["batch_id"])


def test_snapshot_catalog_publishes_recovers_and_verifies(tmp_path):
    root = tmp_path / "snapshots"
    snapshot = _snapshot()
    catalog = SnapshotCatalog(root)
    entry = catalog.publish(snapshot, stream_id="daily", selected_batch_ids=("batch-1",))
    assert catalog.load(entry.snapshot_id).snapshot_hash == snapshot.snapshot_hash
    assert catalog.latest_accepted("daily").snapshot_id == entry.snapshot_id

    recovered = SnapshotCatalog(root)
    assert recovered.resolve(entry.snapshot_id).batch_ids == ("batch-1",)
    bars_path = root / entry.snapshot_id / "bars.parquet"
    bars_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        recovered.resolve(entry.snapshot_id)


def test_snapshot_id_collision_with_different_content_is_rejected(tmp_path):
    catalog = SnapshotCatalog(tmp_path / "snapshots")
    original = _snapshot()
    catalog.publish(original, stream_id="daily")
    altered_bars = original.bars.with_columns((pl.col("close") + 1).alias("close"))
    altered = original.model_copy(update={"bars": altered_bars})

    with pytest.raises(ValueError, match="identity collision"):
        catalog.publish(altered, stream_id="daily")


def test_concurrent_identical_snapshot_publication_is_create_only(tmp_path):
    root = tmp_path / "snapshots"
    snapshot = _snapshot()
    SnapshotCatalog(root)
    barrier = Barrier(2)

    def publish():
        catalog = SnapshotCatalog(root)
        barrier.wait()
        entry = catalog.publish(snapshot, stream_id="daily")
        catalog.connection.close()
        return entry.snapshot_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot_ids = list(executor.map(lambda _: publish(), range(2)))

    catalog = SnapshotCatalog(root)
    assert snapshot_ids == [snapshot.snapshot_id, snapshot.snapshot_id]
    assert [entry.snapshot_id for entry in catalog.list()] == [snapshot.snapshot_id]
