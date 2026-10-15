import json
from pathlib import Path

import nbformat


def test_research_walkthrough_has_statistical_checkpoint():
    notebook = json.loads(open("notebooks/research_walkthrough.ipynb").read())
    nbformat.validate(notebook)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "campaign_evidence" in source
    assert "sharpe_evidence_metrics" in source
    assert "SNAPSHOT_ID" in source
    assert "SnapshotCatalog().load(SNAPSHOT_ID)" in source
    assert "snapshot.provenance" in source
    assert "prepare_dataset(config, snapshot=selected_snapshot)" in source
    assert "plan_campaign(config, campaign_spec)" in source
    assert "execute_campaign(config, campaign_plan)" in source
    assert "generate_run_report(report_path, result=run, config=config, evidence=current_evidence)" in source
    assert "current_evidence['included_logical_trial_ids']" in source
    assert "current_evidence['evidence_sidecar']['path']" in source
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None or isinstance(cell.get("execution_count"), int) for cell in code_cells)
    assert all(isinstance(cell.get("outputs", []), list) for cell in code_cells)
    assert all(
        isinstance(output, dict)
        and output.get("output_type") in {"stream", "display_data", "execute_result", "error"}
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_research_walkthrough_is_supported_manual_interface():
    readme = Path("README.md").read_text().lower()

    assert "research_walkthrough.ipynb" in readme
    assert "supported manual research interface" in readme
    assert "make notebook" in readme
