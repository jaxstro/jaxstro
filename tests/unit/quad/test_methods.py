"""Static adaptive-method configuration contracts."""

import inspect
from dataclasses import FrozenInstanceError

import jax
import pytest

from jaxstro.quad import (
    AdaptiveClenshawCurtis,
    AdaptiveTanhSinh,
    GaussKronrod,
    Romberg,
    RombergTanhSinh,
)


@pytest.mark.parametrize("pair", [15, 21, 31, 41, 51, 61])
def test_gauss_kronrod_accepts_every_canonical_pair_as_static_metadata(pair) -> None:
    method = GaussKronrod(pair=pair)
    leaves, tree = jax.tree.flatten(method)
    assert leaves == []
    assert jax.tree.unflatten(tree, leaves) == method


@pytest.mark.parametrize("pair", [True, 0, 7, 10, 20, 63, 21.0])
def test_gauss_kronrod_rejects_noncanonical_pairs(pair) -> None:
    with pytest.raises(ValueError, match="15, 21, 31, 41, 51, or 61"):
        GaussKronrod(pair=pair)


@pytest.mark.parametrize("order", [5, 9, 17, 33, 65])
def test_adaptive_clenshaw_curtis_accepts_nested_orders(order) -> None:
    assert AdaptiveClenshawCurtis(initial_order=order).initial_order == order


@pytest.mark.parametrize("order", [True, 0, 1, 3, 6, 16, 18, 33.0])
def test_adaptive_clenshaw_curtis_rejects_non_nested_orders(order) -> None:
    with pytest.raises(ValueError, match=r"2\^k \+ 1"):
        AdaptiveClenshawCurtis(initial_order=order)


@pytest.mark.parametrize(
    ("factory", "field", "accepted", "rejected", "message"),
    [
        (AdaptiveTanhSinh, "initial_level", 0, -1, "nonnegative integer"),
        (Romberg, "initial_level", 1, 0, "positive integer"),
        (RombergTanhSinh, "initial_level", 1, 0, "positive integer"),
    ],
)
def test_adaptive_level_methods_validate_static_integer_metadata(
    factory, field, accepted, rejected, message
) -> None:
    method = factory(**{field: accepted})
    leaves, tree = jax.tree.flatten(method)
    assert leaves == []
    assert jax.tree.unflatten(tree, leaves) == method
    with pytest.raises(ValueError, match=message):
        factory(**{field: rejected})
    with pytest.raises(ValueError, match=message):
        factory(**{field: True})


@pytest.mark.parametrize(
    ("factory", "field", "default"),
    [
        (GaussKronrod, "pair", 21),
        (AdaptiveClenshawCurtis, "initial_order", 17),
        (AdaptiveTanhSinh, "initial_level", 3),
        (Romberg, "initial_level", 1),
        (RombergTanhSinh, "initial_level", 1),
    ],
)
def test_adaptive_method_defaults_signatures_and_immutability(
    factory, field, default
) -> None:
    method = factory()
    parameter = inspect.signature(factory).parameters[field]
    assert parameter.default == default
    assert getattr(method, field) == default
    with pytest.raises(FrozenInstanceError):
        setattr(method, field, default)
