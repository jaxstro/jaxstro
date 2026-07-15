import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad


@pytest.mark.parametrize(
    "norm,expected",
    (
        (quad.MaxNorm(), 5.0),
        (quad.L1Norm(), 9.0),
        (quad.L2Norm(), jnp.sqrt(35.0)),
    ),
)
def test_error_norm_reduces_payload_to_one_scalar(norm, expected) -> None:
    error = jnp.array([1.0, 3.0, 5.0])
    assert jnp.allclose(quad.error_norm(error, norm), expected)


def test_complex_error_uses_magnitude() -> None:
    error = jnp.array([3.0 + 4.0j, 0.0 + 2.0j])
    assert quad.error_norm(error, quad.MaxNorm()) == 5.0


def test_tolerance_is_absolute_or_relative_maximum() -> None:
    value = jnp.array([3.0, 4.0])
    tolerance = quad.tolerance_threshold(
        value,
        epsabs=1e-3,
        epsrel=1e-2,
        norm=quad.L2Norm(),
    )
    assert jnp.allclose(tolerance, 5e-2)


def test_norm_configuration_is_static_under_jit() -> None:
    evaluate = jax.jit(
        lambda value, norm: quad.tolerance_threshold(
            value,
            epsabs=1e-6,
            epsrel=1e-3,
            norm=norm,
        )
    )
    assert jnp.allclose(
        evaluate(jnp.array([2.0, -3.0]), quad.MaxNorm()),
        3e-3,
    )
