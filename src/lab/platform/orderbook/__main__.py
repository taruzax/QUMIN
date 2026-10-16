"""Command line entry point for historical order-book recordings."""

import argparse
import json
import sys

from lab.platform.orderbook.acquisition import fetch_history, import_history
from lab.platform.orderbook.contracts import OrderBookError
from lab.platform.orderbook.replay import validate_history


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab.platform.orderbook")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="fetch a public first-of-month sample")
    fetch.add_argument("--symbol", required=True)
    fetch.add_argument("--date", required=True)
    fetch.add_argument("--root", required=True)
    local_import = commands.add_parser("import", help="import a local CSV.GZ recording")
    local_import.add_argument("--input", required=True)
    local_import.add_argument("--symbol", required=True)
    local_import.add_argument("--date", required=True)
    local_import.add_argument("--root", required=True)
    validate = commands.add_parser("validate", help="replay a manifest into a validated book artifact")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one explicit acquisition or replay request."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "fetch":
            path = fetch_history(args.symbol, args.date, args.root)
            result = {"manifest": str(path)}
        elif args.command == "import":
            path = import_history(args.input, args.symbol, args.date, args.root)
            result = {"manifest": str(path)}
        else:
            path = validate_history(args.manifest, args.output)
            result = {"output": str(path)}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OrderBookError, OSError) as exc:
        print(f"orderbook: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
