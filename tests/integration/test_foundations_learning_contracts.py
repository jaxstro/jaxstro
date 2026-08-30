"""Learner-facing contracts for the Foundations throughline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONS = ROOT / "docs/10-foundations"

LEAVES = (
    "mathematical-objects/functions-units-scales.md",
    "mathematical-objects/linear-algebra-language-of-change.md",
    "mathematical-objects/what-is-a-derivative.md",
    "mathematical-objects/probability-and-distributions.md",
    "models-and-computation/what-is-a-model.md",
    "models-and-computation/models-inference-information.md",
    "models-and-computation/sensitivity-conditioning-identifiability.md",
    "models-and-computation/from-relations-to-differentiable-programs.md",
)


def test_every_foundation_leaf_has_a_shared_action_and_a_worked_audit() -> None:
    for relative_path in LEAVES:
        text = (FOUNDATIONS / relative_path).read_text(encoding="utf-8")
        assert "## Try the running case" in text, relative_path
        assert "## Worked audit" in text, relative_path
        assert "two-channel measurement" in text.lower(), relative_path


def test_statistical_and_programming_foundations_teach_their_core_relations() -> None:
    expected_equations = {
        "mathematical-objects/probability-and-distributions.md": (
            r"\int_{\mathcal{X}} p(x)\,dx = 1",
            r"\mathbb{E}[g(X)] = \int_{\mathcal{X}} g(x)p(x)\,dx",
        ),
        "models-and-computation/what-is-a-model.md": (
            r"z = f(\theta, s)",
            r"d = h(z, \eta) + \varepsilon",
        ),
        "models-and-computation/models-inference-information.md": (
            r"p(\theta, \eta \mid d) \propto p(d \mid \theta, \eta)p(\theta, \eta)",
            r"p(d_{\mathrm{rep}} \mid d)",
        ),
        "models-and-computation/sensitivity-conditioning-identifiability.md": (
            r"\delta d \approx J\,\delta\theta",
            r"F = J^{\mathsf{T}}C^{-1}J",
        ),
        "models-and-computation/from-relations-to-differentiable-programs.md": (
            r"x_K(\theta) = \Phi_K(\theta)",
            r"\frac{d x_K}{d\theta}",
        ),
    }
    for relative_path, equations in expected_equations.items():
        text = (FOUNDATIONS / relative_path).read_text(encoding="utf-8")
        for equation in equations:
            assert equation in text, f"{relative_path}: {equation}"


def test_landing_route_and_implicit_derivative_evidence_are_explicit() -> None:
    landing = (FOUNDATIONS / "foundations.md").read_text(encoding="utf-8")
    assert "## Recommended route through the foundations" in landing
    for relative_path in LEAVES:
        assert relative_path in landing

    program_page = (
        FOUNDATIONS / "models-and-computation/from-relations-to-differentiable-programs.md"
    ).read_text(encoding="utf-8")
    for target in (
        "rootfinding.md",
        "50-api/change-constraints/rootfinding.md",
        "60-validation/numerical/implicit-root-gradients.md",
    ):
        assert target in program_page
