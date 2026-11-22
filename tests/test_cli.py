import json

import lab.cli as cli
from lab.cli import build_parser, main
from lab.research.holdout_gate import HoldoutRejected


def test_cli_parser_exposes_stable_stage_commands():
    parser = build_parser()
    for command in (
        "ingest",
        "snapshot-list",
        "snapshot-inspect",
        "prepare",
        "dataset-inspect",
        "experiment",
        "campaign-plan",
        "campaign-run",
        "campaign-resume",
        "campaign-status",
        "backtest",
        "report",
        "report-inspect",
        "development",
        "holdout",
    ):
        assert command in parser._subparsers._group_actions[0].choices


def test_holdout_requires_explicit_acknowledgement_before_loading_run(capsys):
    assert main(["holdout", "run-placeholder", "--model", "xgboost"]) == 1
    assert "--acknowledge-holdout" in capsys.readouterr().err


def test_machine_errors_are_structured_json(capsys):
    assert main(["--json", "holdout", "run-placeholder", "--model", "xgboost"]) == 1
    assert '"type": "ValueError"' in capsys.readouterr().out


def test_pending_holdout_is_retryable_and_exits_one(monkeypatch, capsys):
    def pending(_args):
        raise HoldoutRejected(
            {
                "verdict": "REJECTED_HOLDOUT_SEALED",
                "execution_status": "running",
                "evidence": {"reason": "HOLDOUT_RUNNING"},
            }
        )

    monkeypatch.setattr(cli, "_dispatch", pending)
    assert main(["--json", "holdout", "run-placeholder", "--acknowledge-holdout"]) == 1
    assert json.loads(capsys.readouterr().out)["retryable"] is True


def test_report_command_refuses_to_overwrite_existing_report(tmp_path, capsys):
    report = tmp_path / "existing.html"
    report.write_text("immutable")

    assert main(["report", "run-placeholder", "--output", str(report)]) == 1
    assert "choose a new path" in capsys.readouterr().err
    assert report.read_text() == "immutable"
