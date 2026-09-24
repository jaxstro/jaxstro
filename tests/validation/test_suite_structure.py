"""Regression guard: all tests live in a tier dir and carry a tier marker."""

import pathlib

from tests.conftest import _tier_for

TIERS = ("unit", "integration", "validation")
TESTS_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_flat_test_modules():
    """No test_*.py may sit directly under tests/ — every test belongs to a tier."""
    flat = sorted(p.name for p in TESTS_ROOT.glob("test_*.py"))
    assert flat == [], f"flat test modules must move into a tier dir: {flat}"


def test_tier_dirs_populated():
    for tier in TIERS:
        d = TESTS_ROOT / tier
        assert d.is_dir(), f"missing tier dir: {d}"
        assert list(d.glob("test_*.py")), f"tier {tier} has no test modules"


def test_this_module_is_marked_validation(request):
    """The path->marker conftest hook applied this module's tier marker.

    No explicit decorator here: one would make this pass without the hook.
    """
    own = {m.name for m in request.node.iter_markers()}
    assert "validation" in own


def test_tier_comes_from_the_path_below_tests_only():
    """A directory named like a tier above the repo must not decide the tier."""
    checkout = pathlib.Path("/home/u/unit/jaxstro/tests")
    assert _tier_for(checkout / "validation" / "test_x.py", checkout) == "validation"
    assert _tier_for(checkout / "unit" / "quad" / "test_x.py", checkout) == "unit"
    assert _tier_for(checkout / "helpers.py", checkout) is None
