from lab.research.plots import plot_learning_curves, plot_trial_evidence
from lab.quant.metrics import unavailable_metric


def test_plot_helpers_preserve_available_and_unavailable_evidence():
    figure, axis = plot_learning_curves([{"epoch": 1.0, "loss": 0.2}], None)
    evidence_figure, evidence_axis = plot_trial_evidence({"psr": unavailable_metric("psr", "cash")})

    assert axis.lines
    assert evidence_axis.patches
    figure.clf()
    evidence_figure.clf()
