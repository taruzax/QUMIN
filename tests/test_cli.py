from lab.cli import build_parser, main


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


def test_report_command_refuses_to_overwrite_existing_report(tmp_path, capsys):
    report = tmp_path / "existing.html"
    report.write_text("immutable")

    assert main(["report", "run-placeholder", "--output", str(report)]) == 1
    assert "choose a new path" in capsys.readouterr().err
    assert report.read_text() == "immutable"
