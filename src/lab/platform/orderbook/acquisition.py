"""On-demand acquisition and immutable caching of daily recordings."""

import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import zlib
from datetime import datetime, timezone
from http.client import IncompleteRead
from itertools import chain
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lab.platform.orderbook.contracts import (
    DATA_TYPE,
    EXCHANGE,
    MANIFEST_VERSION,
    RECORDING_VENDOR,
    SCHEMA_VERSION,
    TIMESTAMP_UNIT,
    OrderBookError,
    OrderBookManifest,
    manifest_from_dict,
    validate_request,
)

CSV_HEADER = ("exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount")
TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1024 * 1024


def dataset_url(symbol: str, day) -> str:
    """Return the fixed public first-of-month Tardis sample URL."""
    return f"https://datasets.tardis.dev/v1/binance/incremental_book_L2/{day:%Y/%m}/01/{symbol}.csv.gz"


def _validate_recording(path: Path, compressed: bool) -> None:
    try:
        source = (
            gzip.open(path, "rt", encoding="utf-8", newline="") if compressed else path.open("rt", encoding="utf-8", newline="")
        )
        with source:
            reader = csv.reader(source, strict=True)
            header = next(reader, None)
            if tuple(header or ()) != CSV_HEADER:
                raise OrderBookError("recording has an empty or incorrect CSV header")
            first_row = next(reader, None)
            if first_row is None:
                raise OrderBookError("recording contains no data rows")
            for row in chain((first_row,), reader):
                if len(row) != len(CSV_HEADER) or any(cell == "" for cell in row):
                    raise OrderBookError("recording contains a row with missing or incorrect cells")
    except (OSError, EOFError, UnicodeError, csv.Error, gzip.BadGzipFile, zlib.error) as exc:
        raise OrderBookError("recording is unreadable or its gzip stream is truncated/corrupt") from exc


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_manifest(path: Path) -> OrderBookManifest:
    try:
        return manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrderBookError("cached manifest is missing or malformed") from exc


def _verify_target(target: Path, symbol: str, day) -> Path:
    manifest_path = target / "manifest.json"
    manifest = _read_manifest(manifest_path)
    expected_name = f"{symbol}-{day.isoformat()}.csv.gz"
    if (
        manifest.symbol != symbol
        or manifest.requested_date != day.isoformat()
        or manifest.input_filename != expected_name
        or not (target / expected_name).is_file()
    ):
        raise OrderBookError(f"existing artifact conflicts with request: {target}")
    size, sha256 = _digest(target / expected_name)
    if size != manifest.byte_count or sha256 != manifest.sha256:
        raise OrderBookError(f"existing artifact checksum mismatch: {target}")
    return manifest_path


def _publish(source: Path, symbol: str, day, root: Path, method: str, source_url: str | None) -> Path:
    filename = f"{symbol}-{day.isoformat()}.csv.gz"
    target = root / f"{symbol}-{day.isoformat()}"
    root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing_manifest = _verify_target(target, symbol, day)
        if method == "import":
            old_size, old_hash = _digest(target / filename)
            new_size, new_hash = _digest(source)
            if (old_size, old_hash) != (new_size, new_hash):
                raise OrderBookError(f"import conflicts with existing recording: {target}")
        return existing_manifest
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=root))
    try:
        shutil.copyfile(source, staging / filename)
        _validate_recording(staging / filename, compressed=True)
        size, sha256 = _digest(staging / filename)
        manifest = OrderBookManifest(
            MANIFEST_VERSION,
            SCHEMA_VERSION,
            EXCHANGE,
            DATA_TYPE,
            RECORDING_VENDOR,
            symbol,
            day.isoformat(),
            filename,
            size,
            sha256,
            TIMESTAMP_UNIT,
            method,
            source_url,
        )
        (staging / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.rename(staging, target)
        except FileExistsError:
            return _verify_target(target, symbol, day)
        return target / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def fetch_history(symbol: str, day, root: str | Path) -> Path:
    """Fetch or reuse a verified public monthly sample for one completed day."""
    request = validate_request(symbol, day)
    if request.day.day != 1:
        raise OrderBookError("free Tardis samples are available only for the first day of each month")
    if request.day >= datetime.now(timezone.utc).date():
        raise OrderBookError("only completed historical dates can be fetched")
    root_path = Path(root).expanduser().resolve()
    target = root_path / f"{symbol}-{request.day.isoformat()}"
    if target.exists():
        return _verify_target(target, symbol, request.day)
    url = dataset_url(symbol, request.day)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=root_path)) if root_path.exists() else None
    if staging is None:
        root_path.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=root_path))
    compressed = staging / f"{symbol}-{request.day.isoformat()}.csv.gz"
    try:
        try:
            with urlopen(Request(url, headers={"User-Agent": "quant-ml-lab-orderbook/1"}), timeout=TIMEOUT_SECONDS) as response:
                declared = response.headers.get("Content-Length")
                received = 0
                with compressed.open("wb") as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        received += len(chunk)
                if declared is not None:
                    try:
                        declared_length = int(declared)
                    except ValueError as exc:
                        raise OrderBookError("HTTP Content-Length is malformed") from exc
                    if received != declared_length:
                        raise OrderBookError("download length does not match HTTP Content-Length")
        except HTTPError as exc:
            raise OrderBookError(f"Tardis download failed with HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError, IncompleteRead, zlib.error) as exc:
            raise OrderBookError(f"Tardis download failed: {exc.reason if isinstance(exc, URLError) else exc}") from exc
        _validate_recording(compressed, compressed=True)
        # Publish from the already downloaded staging file without changing its bytes.
        size, sha256 = _digest(compressed)
        manifest = OrderBookManifest(
            MANIFEST_VERSION,
            SCHEMA_VERSION,
            EXCHANGE,
            DATA_TYPE,
            RECORDING_VENDOR,
            symbol,
            request.day.isoformat(),
            compressed.name,
            size,
            sha256,
            TIMESTAMP_UNIT,
            "download",
            url,
        )
        (staging / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.rename(staging, target)
        except FileExistsError:
            return _verify_target(target, symbol, request.day)
        return target / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def import_history(path: str | Path, symbol: str, day, root: str | Path) -> Path:
    """Copy and publish a local CSV.gz recording without modifying its source."""
    request = validate_request(symbol, day)
    try:
        source = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise OrderBookError("input recording is missing or inaccessible") from exc
    resolved_root = Path(root).expanduser().resolve()
    if source == resolved_root or resolved_root in source.parents:
        raise OrderBookError("input recording must not be inside the artifact root")
    if source.suffix.lower() != ".gz" or not source.name.lower().endswith(".csv.gz"):
        raise OrderBookError("import input must be a .csv.gz recording")
    return _publish(source, symbol, request.day, Path(root).expanduser().resolve(), "import", None)
