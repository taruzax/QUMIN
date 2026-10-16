"""Deterministic bounded-memory replay of recorded L2 messages."""

import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import zlib
from decimal import Decimal, InvalidOperation
from http.client import IncompleteRead
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lab.platform.orderbook.acquisition import CSV_HEADER
from lab.platform.orderbook.contracts import (
    OrderBookError,
    decimal_text,
    manifest_from_dict,
    timestamp_from_microseconds,
)

EVENT_SCHEMA = pa.schema(
    [
        ("source_row", pa.int64()),
        ("exchange", pa.string()),
        ("symbol", pa.string()),
        ("timestamp", pa.string()),
        ("local_timestamp", pa.string()),
        ("is_snapshot", pa.bool_()),
        ("side", pa.string()),
        ("price", pa.string()),
        ("amount", pa.string()),
        ("applied", pa.bool_()),
    ]
)
BUFFER_ROWS = 10_000


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_manifest(manifest_path: Path):
    try:
        manifest = manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrderBookError("manifest is missing or malformed") from exc
    try:
        input_path = (manifest_path.parent / manifest.input_filename).resolve(strict=True)
    except OSError as exc:
        raise OrderBookError("manifest source file is missing or inaccessible") from exc
    if input_path.parent != manifest_path.parent.resolve() or not input_path.is_file():
        raise OrderBookError("manifest input must be inside its artifact directory")
    try:
        size, digest = _sha256(input_path)
    except OSError as exc:
        raise OrderBookError("manifest source file is missing or inaccessible") from exc
    if size != manifest.byte_count or digest != manifest.sha256:
        raise OrderBookError("source byte count or SHA-256 does not match manifest")
    return manifest, input_path


def _decimal(value: str, field: str, positive: bool = False) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise OrderBookError(f"invalid {field}: {value!r}") from exc
    if not result.is_finite() or (result <= 0 if positive else result < 0):
        requirement = "positive and finite" if positive else "nonnegative and finite"
        raise OrderBookError(f"{field} must be {requirement}")
    return result


def validate_history(manifest_path: str | Path, output_dir: str | Path) -> Path:
    """Validate and replay one recording into an atomic local artifact."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest, source = _read_manifest(manifest_file)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise OrderBookError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    events_path = staging / "events.parquet"
    bids: dict[Decimal, Decimal] = {}
    asks: dict[Decimal, Decimal] = {}
    rows: list[dict] = []
    counters = {
        "source_rows": 0,
        "initial_rows_skipped": 0,
        "messages": 0,
        "snapshot_blocks": 0,
        "snapshot_resets": 0,
        "accepted_states": 0,
        "exchange_time_regressions": 0,
    }
    first_local = last_local = first_exchange = last_exchange = None
    previous_local = None
    initialized = False
    snapshot_mode = False
    message_rows: list[tuple[dict, bool]] = []
    message_local_key = None
    writer = None

    def flush_events() -> None:
        nonlocal writer, rows
        if rows:
            table = pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(events_path, EVENT_SCHEMA, compression="zstd")
            writer.write_table(table)
            rows = []

    def add_row(record: dict, applied: bool) -> None:
        rows.append({**record, "applied": applied})
        if len(rows) >= BUFFER_ROWS:
            flush_events()

    def check_book() -> None:
        if not bids or not asks:
            raise OrderBookError("reconstructed book is one-sided")
        if max(bids) >= min(asks):
            raise OrderBookError("reconstructed book is locked or crossed")

    def commit_message() -> None:
        nonlocal initialized, snapshot_mode
        if not message_rows:
            return
        counters["messages"] += 1
        flags = {is_snapshot for _, is_snapshot in message_rows}
        if len(flags) != 1:
            raise OrderBookError("received message mixes snapshot and delta rows")
        is_snapshot = flags.pop()
        if is_snapshot:
            if not snapshot_mode:
                if initialized:
                    counters["snapshot_resets"] += 1
                bids.clear()
                asks.clear()
                snapshot_mode = True
                counters["snapshot_blocks"] += 1
            for record, _ in message_rows:
                book = bids if record["side"] == "bid" else asks
                price, amount = Decimal(record["price"]), Decimal(record["amount"])
                if amount == 0:
                    book.pop(price, None)
                else:
                    book[price] = amount
            initialized = True
            for record, _ in message_rows:
                add_row(record, True)
            return
        if snapshot_mode:
            check_book()
            counters["accepted_states"] += 1
            snapshot_mode = False
        if not initialized:
            counters["initial_rows_skipped"] += len(message_rows)
            for record, _ in message_rows:
                add_row(record, False)
            return
        updates: list[tuple[dict, Decimal, Decimal]] = []
        for record, _ in message_rows:
            updates.append((record, Decimal(record["price"]), Decimal(record["amount"])))
        for record, price, amount in updates:
            book = bids if record["side"] == "bid" else asks
            if amount == 0:
                book.pop(price, None)
            else:
                book[price] = amount
        check_book()
        counters["accepted_states"] += 1
        for record, _, _ in updates:
            add_row(record, True)

    try:
        with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            if tuple(next(reader, []) or ()) != CSV_HEADER:
                raise OrderBookError("recording has an incorrect CSV header")
            for row_number, cells in enumerate(reader, start=2):
                counters["source_rows"] += 1
                if len(cells) != len(CSV_HEADER) or any(cell == "" for cell in cells):
                    raise OrderBookError(f"malformed required cells at source row {row_number}")
                exchange, symbol, exchange_time, local_time, snapshot_text, side, price_text, amount_text = cells
                if exchange != manifest.exchange or symbol != manifest.symbol:
                    raise OrderBookError(f"exchange or symbol mismatch at source row {row_number}")
                if snapshot_text not in {"true", "false"} or side not in {"bid", "ask"}:
                    raise OrderBookError(f"invalid snapshot flag or side at source row {row_number}")
                _decimal(price_text, "price", positive=True)
                _decimal(amount_text, "amount")
                exchange_dt = timestamp_from_microseconds(exchange_time)
                local_dt = timestamp_from_microseconds(local_time)
                if local_dt.date().isoformat() != manifest.requested_date:
                    raise OrderBookError(f"arrival UTC date mismatch at source row {row_number}")
                if previous_local is not None and local_dt < previous_local:
                    raise OrderBookError(f"local_timestamp regressed at source row {row_number}")
                if last_exchange is not None and exchange_dt < last_exchange:
                    counters["exchange_time_regressions"] += 1
                previous_local, last_local, last_exchange = local_dt, local_dt, exchange_dt
                first_local = first_local or local_dt
                first_exchange = first_exchange or exchange_dt
                record = {
                    "source_row": row_number,
                    "exchange": exchange,
                    "symbol": symbol,
                    "timestamp": exchange_dt.isoformat(),
                    "local_timestamp": local_dt.isoformat(),
                    "is_snapshot": snapshot_text == "true",
                    "side": side,
                    "price": price_text,
                    "amount": amount_text,
                }
                flag = snapshot_text == "true"
                if message_rows and local_time != message_local_key:
                    commit_message()
                    message_rows = []
                    message_local_key = None
                if not message_rows:
                    message_local_key = local_time
                message_rows.append((record, flag))
        commit_message()
        if not initialized:
            raise OrderBookError("recording contains no snapshot")
        if snapshot_mode:
            check_book()
            counters["accepted_states"] += 1
        if not bids or not asks:
            raise OrderBookError("recording ends without a valid two-sided book")
        flush_events()
        if writer is None:
            raise OrderBookError("recording contains no events")
        writer.close()
        writer = None
        final_book = {
            "bids": [[decimal_text(p), decimal_text(q)] for p, q in sorted(bids.items(), reverse=True)],
            "asks": [[decimal_text(p), decimal_text(q)] for p, q in sorted(asks.items())],
        }
        (staging / "final_book.json").write_text(json.dumps(final_book, indent=2) + "\n", encoding="utf-8")
        _, event_hash = _sha256(events_path)
        report = {
            "manifest_version": 1,
            "replay_schema_version": 1,
            "event_timestamp_format": "ISO-8601 UTC",
            "source": {
                "exchange": manifest.exchange,
                "data_type": manifest.data_type,
                "recording_vendor": manifest.recording_vendor,
                "symbol": manifest.symbol,
                "requested_date": manifest.requested_date,
                "source_sha256": manifest.sha256,
            },
            "quality": {
                **counters,
                "first_local_timestamp": first_local.isoformat() if first_local else None,
                "last_local_timestamp": last_local.isoformat() if last_local else None,
                "first_exchange_timestamp": first_exchange.isoformat() if first_exchange else None,
                "last_exchange_timestamp": last_exchange.isoformat() if last_exchange else None,
                "native_sequence_continuity": "unverified",
                "disconnect_visibility": "unavailable_in_csv",
                "exchange_time_gaps_are_not_interpreted_as_missing_events": True,
            },
            "events_sha256": event_hash,
            "events_file": "events.parquet",
        }
        (staging / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.rename(staging, output)
        except FileExistsError as exc:
            raise OrderBookError(f"output already exists: {output}") from exc
        return output
    except (OSError, EOFError, UnicodeError, csv.Error, gzip.BadGzipFile, IncompleteRead, zlib.error) as exc:
        if isinstance(exc, OrderBookError):
            raise
        raise OrderBookError("source gzip or artifact I/O failed") from exc
    finally:
        if writer is not None:
            writer.close()
        if staging.exists():
            shutil.rmtree(staging)
