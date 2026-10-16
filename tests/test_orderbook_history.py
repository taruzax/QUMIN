"""Offline tests for historical Binance L2 acquisition and replay."""

import csv
import gzip
import hashlib
import io
import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from lab.platform.orderbook.acquisition import CSV_HEADER, fetch_history, import_history
from lab.platform.orderbook.contracts import OrderBookError, validate_request
from lab.platform.orderbook.replay import validate_history


def recording(rows: list[list[str]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    writer.writerows(rows)
    return gzip.compress(text.getvalue().encode())


def row(exchange_time: int, local_time: int, snapshot: bool, side: str, price: str, amount: str) -> list[str]:
    return ["binance", "BTCUSDT", str(exchange_time), str(local_time), str(snapshot).lower(), side, price, amount]


def imported(tmp_path: Path, rows: list[list[str]]) -> Path:
    source = tmp_path / "input.csv.gz"
    source.write_bytes(recording(rows))
    return import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw")


def test_contract_rejects_invalid_identity_and_date():
    assert validate_request("BTCUSDT", date(2024, 1, 1)).day == date(2024, 1, 1)
    for symbol, day, exchange in [
        ("btc-usdt", "2024-01-01", "binance"),
        ("BTCUSDT", "2024-02-30", "binance"),
        ("BTCUSDT", "2024-01-01", "unknown"),
    ]:
        with pytest.raises(OrderBookError):
            validate_request(symbol, day, exchange)


def test_import_is_immutable_and_cache_is_verified(tmp_path: Path):
    raw = recording(
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "2"),
        ]
    )
    source = tmp_path / "input.csv.gz"
    source.write_bytes(raw)
    manifest = import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw")
    assert source.read_bytes() == raw
    assert import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw") == manifest
    artifact = manifest.parent / "BTCUSDT-2024-01-01.csv.gz"
    artifact.write_bytes(b"tampered")
    with pytest.raises(OrderBookError, match="checksum"):
        import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw")


def test_import_conflicting_recording_does_not_reuse_cache(tmp_path: Path):
    root = tmp_path / "raw"
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    first.write_bytes(
        recording(
            [
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
            ]
        )
    )
    second.write_bytes(
        recording(
            [
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "99", "1"),
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
            ]
        )
    )
    import_history(first, "BTCUSDT", "2024-01-01", root)
    with pytest.raises(OrderBookError, match="conflicts"):
        import_history(second, "BTCUSDT", "2024-01-01", root)


def test_fetch_uses_sample_url_stream_and_verified_cache(tmp_path: Path, monkeypatch):
    payload = recording(
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "2"),
        ]
    )
    calls = []

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_stub(request, timeout):
        calls.append(request.full_url)
        return Response(payload)

    monkeypatch.setattr("lab.platform.orderbook.acquisition.urlopen", open_stub)
    manifest = fetch_history("BTCUSDT", "2024-01-01", tmp_path / "cache")
    assert calls == ["https://datasets.tardis.dev/v1/binance/incremental_book_L2/2024/01/01/BTCUSDT.csv.gz"]
    assert fetch_history("BTCUSDT", "2024-01-01", tmp_path / "cache") == manifest
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["http", "length", "transport"])
def test_fetch_failures_publish_no_artifact(tmp_path: Path, monkeypatch, failure):
    from urllib.error import HTTPError, URLError

    class Response(io.BytesIO):
        headers = {"Content-Length": "99"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_stub(request, timeout):
        if failure == "http":
            raise HTTPError(request.full_url, 403, "denied", {}, None)
        if failure == "transport":
            raise URLError("offline")
        return Response(
            recording(
                [
                    row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
                    row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
                ]
            )
        )

    monkeypatch.setattr("lab.platform.orderbook.acquisition.urlopen", open_stub)
    root = tmp_path / "cache"
    with pytest.raises(OrderBookError):
        fetch_history("BTCUSDT", "2024-01-01", root)
    assert not list(root.glob("BTCUSDT-2024-01-01"))


def test_fetch_interrupted_read_publishes_no_artifact(tmp_path: Path, monkeypatch):
    from http.client import IncompleteRead

    class Response(io.BytesIO):
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def read(self, size=-1):
            raise IncompleteRead(b"partial", 50)

    monkeypatch.setattr("lab.platform.orderbook.acquisition.urlopen", lambda *args, **kwargs: Response(b""))
    root = tmp_path / "cache"
    with pytest.raises(OrderBookError):
        fetch_history("BTCUSDT", "2024-01-01", root)
    assert not list(root.glob("BTCUSDT-2024-01-01"))


def test_import_rejects_empty_or_wrong_header_without_publication(tmp_path: Path):
    source = tmp_path / "bad.csv.gz"
    source.write_bytes(gzip.compress(b"wrong,header\nrow\n"))
    root = tmp_path / "raw"
    with pytest.raises(OrderBookError):
        import_history(source, "BTCUSDT", "2024-01-01", root)
    assert not list(root.glob("BTCUSDT-2024-01-01")) if root.exists() else True


def test_import_rejects_truncated_gzip(tmp_path: Path):
    source = tmp_path / "truncated.csv.gz"
    source.write_bytes(recording([row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1")])[:-5])
    with pytest.raises(OrderBookError):
        import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw")


def test_import_rejects_corrupt_deflate_payload(tmp_path: Path):
    source = tmp_path / "corrupt.csv.gz"
    payload = bytearray(recording([row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1")]))
    payload[12] ^= 0xFF
    source.write_bytes(payload)
    with pytest.raises(OrderBookError):
        import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw")


def test_replay_snapshot_reset_skip_delete_and_exact_decimal(tmp_path: Path):
    t = 1_704_067_200_000_000
    manifest = imported(
        tmp_path,
        [
            row(t, t, False, "bid", "99", "1"),
            row(t + 2, t + 1, True, "bid", "100.00000000000000000001", "2.5"),
            row(t + 1, t + 2, True, "ask", "101", "3"),  # exchange clock regresses
            row(t + 3, t + 3, False, "bid", "100.00000000000000000001", "0"),
            row(t + 3, t + 3, False, "bid", "100", "4"),
        ],
    )
    result = validate_history(manifest, tmp_path / "replay")
    book = json.loads((result / "final_book.json").read_text())
    report = json.loads((result / "validation.json").read_text())
    assert book == {"bids": [["100", "4"]], "asks": [["101", "3"]]}
    assert report["quality"]["initial_rows_skipped"] == 1
    assert report["quality"]["exchange_time_regressions"] == 1
    assert report["quality"]["native_sequence_continuity"] == "unverified"
    events = pq.read_table(result / "events.parquet").to_pylist()
    assert events[0]["applied"] is False
    assert events[1]["price"] == "100.00000000000000000001"
    assert hashlib.sha256((result / "events.parquet").read_bytes()).hexdigest() == report["events_sha256"]


def test_replay_resets_only_on_false_to_true_snapshot_transition(tmp_path: Path):
    t = 1_704_067_200_000_000
    manifest = imported(
        tmp_path,
        [
            row(t, t, True, "bid", "100", "1"),
            row(t, t, True, "ask", "101", "1"),
            row(t + 1, t + 1, False, "bid", "99", "2"),
            row(t + 2, t + 2, True, "bid", "98", "3"),
            row(t + 3, t + 3, True, "ask", "99", "4"),
        ],
    )
    output = validate_history(manifest, tmp_path / "replay")
    report = json.loads((output / "validation.json").read_text())
    book = json.loads((output / "final_book.json").read_text())
    assert report["quality"]["snapshot_blocks"] == 2
    assert report["quality"]["snapshot_resets"] == 1
    assert book == {"bids": [["98", "3"]], "asks": [["99", "4"]]}


def test_replay_second_snapshot_block_clears_old_levels_and_counts_states(tmp_path: Path):
    t = 1_704_067_200_000_000
    manifest = imported(
        tmp_path,
        [
            row(t, t, True, "bid", "100", "1"),
            row(t, t, True, "bid", "99", "0"),
            row(t, t, True, "bid", "99", "1"),
            row(t, t, True, "bid", "99", "0"),
            row(t, t, True, "ask", "101", "1"),
            row(t + 1, t + 1, False, "bid", "100", "2"),
            row(t + 2, t + 2, True, "bid", "98", "3"),
            row(t + 3, t + 3, True, "ask", "99", "4"),
        ],
    )
    output = validate_history(manifest, tmp_path / "replay")
    report = json.loads((output / "validation.json").read_text())
    book = json.loads((output / "final_book.json").read_text())
    assert report["quality"]["snapshot_blocks"] == 2
    assert report["quality"]["snapshot_resets"] == 1
    assert report["quality"]["accepted_states"] == 3
    assert report["event_timestamp_format"] == "ISO-8601 UTC"
    assert book == {"bids": [["98", "3"]], "asks": [["99", "4"]]}


def test_replay_flushes_parquet_buffer_without_reordering(tmp_path: Path):
    t = 1_704_067_200_000_000
    rows = [row(t, t, True, "bid", "100", "1"), row(t, t, True, "ask", "101", "1")]
    rows.extend(row(t + index, t + index, False, "bid", "100", str(index)) for index in range(1, 10_001))
    output = validate_history(imported(tmp_path, rows), tmp_path / "replay")
    events = pq.read_table(output / "events.parquet").to_pylist()
    assert len(events) == 10_002
    assert [event["source_row"] for event in events[:3]] == [2, 3, 4]
    assert events[-1]["source_row"] == 10_003
    assert json.loads((output / "final_book.json").read_text())["bids"] == [["100", "10000"]]


def test_replay_applies_crossing_resolved_within_one_message(tmp_path: Path):
    t = 1_704_067_200_000_000
    manifest = imported(
        tmp_path,
        [
            row(t, t, True, "bid", "100", "1"),
            row(t, t, True, "ask", "101", "1"),
            row(t + 1, t + 1, False, "bid", "102", "1"),
            row(t + 1, t + 1, False, "ask", "101", "0"),
            row(t + 1, t + 1, False, "ask", "103", "1"),
        ],
    )
    assert validate_history(manifest, tmp_path / "replay").exists()


@pytest.mark.parametrize(
    "rows",
    [
        [row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1")],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "102", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
    ],
)
def test_invalid_final_book_leaves_no_output(tmp_path: Path, rows):
    manifest = imported(tmp_path, rows)
    output = tmp_path / "failed-replay"
    with pytest.raises(OrderBookError):
        validate_history(manifest, output)
    assert not output.exists()


def test_corrupt_source_checksum_and_path_traversal_rejected(tmp_path: Path):
    manifest_path = imported(
        tmp_path,
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
    )
    metadata = json.loads(manifest_path.read_text())
    metadata["input_filename"] = "../outside.csv.gz"
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(OrderBookError):
        validate_history(manifest_path, tmp_path / "replay")


def test_manifest_wrong_types_fail_as_domain_error(tmp_path: Path):
    manifest_path = imported(
        tmp_path,
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
    )
    metadata = json.loads(manifest_path.read_text())
    metadata["byte_count"] = None
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(OrderBookError):
        validate_history(manifest_path, tmp_path / "replay")


def test_cache_malformed_manifest_and_utf8_fail_as_domain_errors(tmp_path: Path):
    source = tmp_path / "input.csv.gz"
    source.write_bytes(
        recording(
            [
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
            ]
        )
    )
    manifest_path = import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw")
    manifest_path.write_bytes(b"\xff\xfe")
    with pytest.raises(OrderBookError):
        import_history(source, "BTCUSDT", "2024-01-01", tmp_path / "raw")
    with pytest.raises(OrderBookError):
        validate_history(manifest_path, tmp_path / "replay")


@pytest.mark.parametrize(
    "rows",
    [
        [row(1_704_067_200_000_000, 1_704_067_200_000_000, False, "bid", "100", "1")],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
            row(1_704_067_200_000_000, 1_704_067_199_999_999, False, "bid", "99", "1"),
        ],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "NaN", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
            row(1_704_067_200_000_000, 1_704_153_600_000_000, False, "bid", "99", "1"),
        ],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "-1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "Infinity"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, "yes", "bid", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
        [
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "buy", "100", "1"),
            row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
        ],
    ],
)
def test_replay_rejects_missing_snapshot_bad_time_or_malformed_decimal(tmp_path: Path, rows):
    manifest = imported(tmp_path, rows)
    with pytest.raises(OrderBookError):
        validate_history(manifest, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_replay_rejects_local_time_regression(tmp_path: Path):
    t = 1_704_067_200_000_000
    manifest = imported(
        tmp_path,
        [
            row(t, t, True, "bid", "100", "1"),
            row(t, t, True, "ask", "101", "1"),
            row(t + 1, t + 2, False, "bid", "99", "1"),
            row(t + 2, t + 1, False, "bid", "98", "1"),
        ],
    )
    with pytest.raises(OrderBookError, match="regressed"):
        validate_history(manifest, tmp_path / "replay")


def test_cli_help_and_import_validate_round_trip(tmp_path: Path, monkeypatch, capsys):
    from lab.platform.orderbook import __main__ as cli

    source = tmp_path / "input.csv.gz"
    source.write_bytes(
        recording(
            [
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "bid", "100", "1"),
                row(1_704_067_200_000_000, 1_704_067_200_000_000, True, "ask", "101", "1"),
            ]
        )
    )
    assert (
        cli.main(
            ["import", "--input", str(source), "--symbol", "BTCUSDT", "--date", "2024-01-01", "--root", str(tmp_path / "raw")]
        )
        == 0
    )
    manifest = Path(json.loads(capsys.readouterr().out)["manifest"])
    assert cli.main(["validate", "--manifest", str(manifest), "--output", str(tmp_path / "validated")]) == 0
    assert json.loads(capsys.readouterr().out)["output"] == str(tmp_path / "validated")
    monkeypatch.setattr("sys.argv", ["orderbook", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert cli.main(["validate", "--manifest", str(tmp_path / "missing.json"), "--output", str(tmp_path / "bad")]) == 2
    assert "orderbook:" in capsys.readouterr().err
