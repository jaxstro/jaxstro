"""Pedagogical contracts for the linear-algebra foundations page."""

import re
from pathlib import Path

PAGE = (
    Path(__file__).resolve().parents[2]
    / "docs/10-foundations/mathematical-objects/linear-algebra-language-of-change.md"
)


def test_linear_algebra_page_teaches_the_core_geometric_equations() -> None:
    text = PAGE.read_text(encoding="utf-8")

    for equation in (
        r"\langle u, v \rangle = u^{\mathsf{T}}v",
        r"\lVert v \rVert_2 = \sqrt{v^{\mathsf{T}}v}",
        r"P_{\mathcal{S}} = QQ^{\mathsf{T}}",
        r"A = U\Sigma V^{\mathsf{T}}",
        r"\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}",
        r"X^{\mathsf{T}}W\widehat r = 0",
        r"\delta y \approx J\,\delta\theta + \varepsilon",
        r"C_y \approx J C_\theta J^{\mathsf{T}} + C_\varepsilon",
    ):
        assert equation in text


def test_linear_algebra_page_states_the_limits_of_its_local_tools() -> None:
    text = re.sub(r"\s+", " ", PAGE.read_text(encoding="utf-8")).lower()

    for phrase in (
        "do not form the inverse",
        "local statement",
        "not global identifiability",
        "positive-definite",
        "rank tolerance",
    ):
        assert phrase in text
