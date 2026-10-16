"""Small contracts for immutable historical L2 recordings."""

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

EXCHANGE = "binance"
DATA_TYPE = "incremental_book_L2"
RECORDING_VENDOR = "tardis.dev"
MANIFEST_VERSION = 1
SCHEMA_VERSION = 1
TIMESTAMP_UNIT = "unix_microseconds_utc"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,30}$")


class OrderBookError(ValueError):
    """Invalid request, recording, or order-book artifact."""


@dataclass(frozen=True)
class OrderBookRequest:
    exchange: str
    data_type: str
    symbol: str
    day: date


@dataclass(frozen=True)
class OrderBookManifest:
    manifest_version: int
    schema_version: int
    exchange: str
    data_type: str
    recording_vendor: str
    symbol: str
    requested_date: str
    input_filename: str
    byte_count: int
    sha256: str
    timestamp_unit: str
    acquisition_method: str
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_request(symbol: str, day: date | str, exchange: str = EXCHANGE, data_type: str = DATA_TYPE) -> OrderBookRequest:
    """Validate common venue, type, symbol and UTC calendar identity."""
    if exchange != EXCHANGE or data_type != DATA_TYPE:
        raise OrderBookError(f"unsupported source identity: {exchange}/{data_type}")
    if not isinstance(symbol, str) or not SYMBOL_PATTERN.fullmatch(symbol):
        raise OrderBookError("symbol must contain 2-30 uppercase letters or digits")
    if isinstance(day, str):
        try:
            day = date.fromisoformat(day)
        except ValueError as exc:
            raise OrderBookError("date must use YYYY-MM-DD") from exc
    if not isinstance(day, date) or isinstance(day, datetime):
        raise OrderBookError("date must be a calendar date")
    return OrderBookRequest(exchange, data_type, symbol, day)


def timestamp_from_microseconds(value: str) -> datetime:
    """Convert a recorded Unix microsecond timestamp to UTC."""
    try:
        micros = int(value)
        if str(micros) != value:
            raise ValueError
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=micros)
    except (ValueError, OverflowError, OSError) as exc:
        raise OrderBookError(f"invalid UTC microsecond timestamp: {value!r}") from exc


def manifest_from_dict(data: dict[str, Any]) -> OrderBookManifest:
    """Parse and strictly validate a versioned acquisition manifest."""
    if not isinstance(data, dict):
        raise OrderBookError("manifest must be a JSON object")
    required = {
        "manifest_version": int,
        "schema_version": int,
        "exchange": str,
        "data_type": str,
        "recording_vendor": str,
        "symbol": str,
        "requested_date": str,
        "input_filename": str,
        "byte_count": int,
        "sha256": str,
        "timestamp_unit": str,
        "acquisition_method": str,
    }
    for key, kind in required.items():
        if type(data.get(key)) is not kind:
            raise OrderBookError(f"manifest field {key} has an invalid type")
    if data.get("source_url") is not None and not isinstance(data["source_url"], str):
        raise OrderBookError("manifest field source_url has an invalid type")
    try:
        manifest = OrderBookManifest(**data)
    except (TypeError, ValueError) as exc:
        raise OrderBookError("malformed manifest fields") from exc
    if manifest.manifest_version != MANIFEST_VERSION or manifest.schema_version != SCHEMA_VERSION:
        raise OrderBookError("unsupported manifest or schema version")
    request = validate_request(manifest.symbol, manifest.requested_date, manifest.exchange, manifest.data_type)
    if request.day.isoformat() != manifest.requested_date:
        raise OrderBookError("manifest date must use YYYY-MM-DD")
    if (
        not manifest.input_filename
        or Path(manifest.input_filename).name != manifest.input_filename
        or manifest.input_filename in {".", ".."}
    ):
        raise OrderBookError("manifest input_filename must be a filename within its artifact directory")
    if manifest.byte_count <= 0 or not re.fullmatch(r"[0-9a-f]{64}", manifest.sha256):
        raise OrderBookError("manifest byte_count or SHA-256 is invalid")
    if manifest.timestamp_unit != TIMESTAMP_UNIT or manifest.recording_vendor != RECORDING_VENDOR:
        raise OrderBookError("unsupported timestamp unit or recording vendor")
    if manifest.acquisition_method not in {"download", "import"}:
        raise OrderBookError("unsupported acquisition method")
    if manifest.acquisition_method == "download" and not manifest.source_url:
        raise OrderBookError("download manifest requires source_url")
    return manifest


def decimal_text(value: Decimal) -> str:
    """Render a finite decimal without converting through binary float."""
    if not value.is_finite():
        raise OrderBookError("decimal values must be finite")
    return format(value, "f")
