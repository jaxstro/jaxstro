"""Visual and executable contracts for the Foundations argument."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "10-foundations"

FIGURES = {
    "foundations.md": "two-channel-measurement-overview.png",
    "mathematical-objects/functions-units-scales.md": "units-residual-space.svg",
    "mathematical-objects/linear-algebra-language-of-change.md": "linear-weak-direction.svg",
    "mathematical-objects/what-is-a-derivative.md": "derivative-chain.svg",
    "mathematical-objects/probability-and-distributions.md": "probability-covariance.svg",
    "models-and-computation/what-is-a-model.md": "model-measurement-chain.svg",
    "models-and-computation/models-inference-information.md": "inference-replication.svg",
    "models-and-computation/sensitivity-conditioning-identifiability.md": "sensitivity-diagnostics.svg",
    "models-and-computation/from-relations-to-differentiable-programs.md": "executed-program-map.svg",
}


def test_each_foundation_page_has_an_accessible_argument_figure() -> None:
    for relative_path, filename in FIGURES.items():
        text = (DOCS / relative_path).read_text(encoding="utf-8")
        assert ":::{figure}" in text, relative_path
        assert filename in text, relative_path
        assert ":alt:" in text, relative_path
        asset = DOCS / "figures" / filename
        assert asset.is_file(), asset
        assert asset.stat().st_size > 0, asset


def test_running_case_is_an_executable_not_only_a_story() -> None:
    script = ROOT / "examples/onboarding/two_channel_measurement.py"
    landing = (DOCS / "foundations.md").read_text(encoding="utf-8")

    assert script.is_file()
    assert "--calibration-sigma" in landing
    assert "--separation" in landing
