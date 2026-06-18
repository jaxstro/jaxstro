"""Regression guard: all tests live in a tier dir and carry a tier marker."""

import pathlib

import pytest

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


@pytest.mark.validation
def test_this_module_is_marked_validation(request):
    """Sanity: the path→marker conftest hook actually applied a tier marker."""
    own = {m.name for m in request.node.iter_markers()}
    assert "validation" in own
