"""Currency contracts for the active repository agent guide."""

from pathlib import Path

from jaxstro._public import PUBLIC_MODULES

GUIDE = Path(__file__).resolve().parents[2] / "CLAUDE.md"


def test_guide_names_current_architecture_and_derivative_targets() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    missing = [m for m in PUBLIC_MODULES if f"`jaxstro.{m}`" not in text]
    assert missing == [], f"package map omits public modules: {missing}"
    assert "finite executed iteration" in text
    assert "certified implicit derivative" in text
    assert "Use `newton` / `newton_with_grad` / `newton_ppf`" not in text


def test_guide_is_not_a_historical_status_log() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for stale in (
        "Phase B working decisions",
        "T7b",
        "feature/consolidate-harden-release",
    ):
        assert stale not in text


def test_guide_preserves_parameter_reconstruction_and_brain_invariants() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert "does not rerun model initialization" in text
    assert "fit the leaf the observable reads" in text
    assert "## Brain status updates" in text
    assert "federate.py" in text
    assert 'brain "…"' in text
