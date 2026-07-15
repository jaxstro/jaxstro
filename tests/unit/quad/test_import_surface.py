import inspect

import jaxstro
from jaxstro.numerics import integration, quadrature


def test_quad_is_a_lazy_public_top_level_module() -> None:
    assert jaxstro.quad.__name__ == "jaxstro.quad"
    assert "quad" in jaxstro.__all__


def test_sampled_facade_is_exact_legacy_identity() -> None:
    assert jaxstro.quad.trapezoid is integration.trapz
    assert jaxstro.quad.cumulative_trapezoid is integration.cumulative_trapz
    assert jaxstro.quad.simpson is integration.simpson
    assert jaxstro.quad.cumulative_simpson is integration.cumulative_simpson


def test_fixed_helper_facade_is_exact_legacy_identity() -> None:
    names = (
        "gauss_legendre_nodes",
        "gauss_laguerre_nodes",
        "gauss_hermite_nodes",
        "clenshaw_curtis_nodes",
        "hermite_e_basis",
        "hermite_coefficients",
    )
    for name in names:
        assert getattr(jaxstro.quad, name) is getattr(quadrature, name)
        assert getattr(jaxstro.numerics, name) is getattr(jaxstro.quad, name)


def test_a0_facade_does_not_change_signatures() -> None:
    assert inspect.signature(jaxstro.quad.trapezoid) == inspect.signature(
        integration.trapz
    )
    assert inspect.signature(jaxstro.quad.simpson) == inspect.signature(
        integration.simpson
    )
