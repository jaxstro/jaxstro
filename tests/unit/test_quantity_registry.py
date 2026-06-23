"""Tests for layered quantity unit registries."""

import pytest

from jaxstro import quantity as q
from jaxstro.quantity.errors import UnitRegistryError
from jaxstro.quantity.registry import UnitRegistry


def test_core_and_astro_registry_lookup():
    assert q.units.CORE_REGISTRY.lookup("cm") is q.cm
    assert q.astro.ASTRO_REGISTRY.lookup("Msun") is q.Msun
    assert q.DEFAULT_REGISTRY.lookup("pc") is q.pc


def test_parent_registry_lookup():
    child = UnitRegistry(
        "child", units={"code_length": q.pc}, parent=q.units.CORE_REGISTRY
    )

    assert child.lookup("code_length") is q.pc
    assert child.lookup("cm") is q.cm


def test_scoped_extension_registry_keeps_parent_immutable():
    scoped = q.DEFAULT_REGISTRY.scoped(
        "experiment",
        units={"code_mass": q.Msun},
        aliases={"mcode": "code_mass"},
    )

    assert scoped.lookup("mcode") is q.Msun
    assert scoped.lookup("km") is q.km
    with pytest.raises(UnitRegistryError):
        q.DEFAULT_REGISTRY.lookup("code_mass")


def test_strict_aliases_do_not_normalize_unknown_spellings():
    assert q.DEFAULT_REGISTRY.lookup("msun") is q.Msun
    with pytest.raises(UnitRegistryError) as exc:
        q.DEFAULT_REGISTRY.lookup("M_sun")

    assert exc.value.actual == "M_sun"


def test_close_match_suggestions_are_structured_and_friendly():
    with pytest.raises(UnitRegistryError) as exc:
        q.DEFAULT_REGISTRY.lookup("Msum")

    assert "Msun" in exc.value.expected
    assert "Did you mean" in str(exc.value)


def test_global_registration_api_is_explicitly_interactive():
    unit = q.Unit("code_time", q.yr.scale_to_cgs, q.yr.dimensions)

    assert "interactive" in (q.register_global_unit.__doc__ or "")
    q.register_global_unit(unit, aliases=("tcode",))
    assert q.GLOBAL_REGISTRY.lookup("tcode") is unit
