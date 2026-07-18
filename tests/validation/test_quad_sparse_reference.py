from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxstro import quad


def orthogonal_matrix(dimension: int) -> Array:
    indices = jnp.arange(dimension, dtype=jnp.float64)
    raw = jnp.cos(jnp.pi * (indices[:, None] + 0.5) * indices[None, :] / dimension)
    return raw / jnp.linalg.norm(raw, axis=0, keepdims=True)


def localized_gaussian(x: Array) -> Array:
    center = jnp.asarray(0.37, dtype=x.dtype)
    beta = jnp.asarray(24.0, dtype=x.dtype)
    return jnp.exp(-beta * jnp.sum((x - center) ** 2, axis=-1))


def localized_gaussian_truth(dimension: int) -> Array:
    center = jnp.asarray(0.37, dtype=jnp.float64)
    beta = jnp.asarray(24.0, dtype=jnp.float64)
    factor = (
        jnp.sqrt(jnp.pi)
        / (2.0 * jnp.sqrt(beta))
        * (
            jax.scipy.special.erf(jnp.sqrt(beta) * (1.0 - center))
            + jax.scipy.special.erf(jnp.sqrt(beta) * center)
        )
    )
    return factor**dimension


@dataclass(frozen=True)
class SparseTruthCase:
    case_id: str
    dimensions: tuple[int, ...]
    integrand: Callable[[Array], Array]
    truth: Callable[[int], Array]
    fixed_levels: tuple[int, ...]
    fixed_work: tuple[int, ...]
    fixed_atol: tuple[float, ...]
    adaptive_dimensions: tuple[int, ...]
    adaptive_max_indices: int
    adaptive_atol: float


SPARSE_TRUTH_CASES = (
    SparseTruthCase(
        "product_quadratic",
        (2, 4, 8, 16),
        lambda x: jnp.prod(1.0 + x**2, axis=-1),
        lambda d: jnp.asarray((4.0 / 3.0) ** d),
        (5, 5, 5, 4),
        (65, 401, 3937, 6049),
        (1.0e-12, 1.0e-12, 5.0e-4, 1.6),
        (2,),
        8,
        1.0e-12,
    ),
    SparseTruthCase(
        "separable_exponential",
        (2, 4, 8, 16),
        lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
        lambda d: jnp.asarray((1.0 - jnp.exp(-1.0)) ** d),
        (5, 5, 5, 4),
        (65, 401, 3937, 6049),
        (1.0e-9, 1.0e-8, 3.0e-8, 2.0e-6),
        (2,),
        16,
        1.0e-8,
    ),
    SparseTruthCase(
        "rotated_quadratic",
        (2, 4, 8),
        lambda x: jnp.sum(
            (x @ orthogonal_matrix(x.shape[-1])) ** 2,
            axis=-1,
        ),
        lambda d: jnp.asarray(d / 3.0),
        (3, 3, 3),
        (13, 41, 145),
        (1.0e-12, 1.0e-12, 1.0e-11),
        (2,),
        8,
        1.0e-12,
    ),
    SparseTruthCase(
        "localized_gaussian",
        (2, 4),
        localized_gaussian,
        localized_gaussian_truth,
        (7, 7),
        (321, 2929),
        (5.0e-6, 9.0e-4),
        (2,),
        28,
        2.0e-8,
    ),
    SparseTruthCase(
        "axis_zero_anisotropy",
        (2, 4, 8, 16),
        lambda x: jnp.exp(-8.0 * x[..., 0]),
        lambda _d: jnp.asarray((1.0 - jnp.exp(-8.0)) / 8.0),
        (5, 5, 5, 4),
        (65, 401, 3937, 6049),
        (1.0e-12, 1.0e-12, 1.0e-12, 4.0e-7),
        (2, 4, 8, 16),
        8,
        1.0e-12,
    ),
)


def _fixed_parameters():
    for case in SPARSE_TRUTH_CASES:
        for dimension, level, work, atol in zip(
            case.dimensions,
            case.fixed_levels,
            case.fixed_work,
            case.fixed_atol,
            strict=True,
        ):
            yield pytest.param(
                case,
                dimension,
                level,
                work,
                atol,
                id=f"{case.case_id}-d{dimension}",
            )


def _adaptive_parameters():
    for case in SPARSE_TRUTH_CASES:
        for dimension in case.adaptive_dimensions:
            yield pytest.param(
                case,
                dimension,
                id=f"{case.case_id}-d{dimension}",
            )


@pytest.mark.parametrize(
    ("case", "dimension", "level", "expected_work", "atol"),
    tuple(_fixed_parameters()),
)
def test_fixed_sparse_grid_against_analytic_truth(
    case: SparseTruthCase,
    dimension: int,
    level: int,
    expected_work: int,
    atol: float,
):
    result = quad.integrate(
        case.integrand,
        quad.Hyperrectangle(jnp.zeros(dimension), jnp.ones(dimension)),
        method=quad.Smolyak(level),
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=7000,
        max_indices=6000,
        max_frontier=6000,
        max_nodes=7000,
        gradient="stop",
    )
    truth = case.truth(dimension)
    assert jnp.abs(result.value - truth) <= atol
    assert result.work.evaluations == expected_work
    assert result.error.kind == quad.ErrorKind.SPARSE_GRID_SURPLUS
    assert jnp.isfinite(result.error.norm)


def test_dimension_16_fixed_control_improves_truth_without_level_5_memory_spike():
    case = SPARSE_TRUTH_CASES[0]
    domain = quad.Hyperrectangle(jnp.zeros(16), jnp.ones(16))

    def solve(level, max_work):
        return quad.integrate(
            case.integrand,
            domain,
            method=quad.Smolyak(level),
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=max_work,
            max_indices=max_work,
            max_frontier=max_work,
            max_nodes=max_work,
            gradient="stop",
        )

    coarse = solve(3, 1000)
    certified = solve(4, 7000)
    truth = case.truth(16)
    assert jnp.abs(certified.value - truth) < jnp.abs(coarse.value - truth)
    assert certified.work.evaluations == 6049


@pytest.mark.parametrize(("case", "dimension"), tuple(_adaptive_parameters()))
def test_adaptive_sparse_grid_against_analytic_truth(
    case: SparseTruthCase,
    dimension: int,
):
    max_indices = case.adaptive_max_indices
    result = quad.integrate(
        case.integrand,
        quad.Hyperrectangle(jnp.zeros(dimension), jnp.ones(dimension)),
        method=quad.AdaptiveSmolyak(initial_level=1),
        epsabs=1.0e-8,
        epsrel=1.0e-8,
        max_evaluations=2048,
        max_indices=max_indices,
        max_frontier=1 + dimension * max_indices,
        max_nodes=2048,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.CONVERGED
    assert jnp.abs(result.value - case.truth(dimension)) <= case.adaptive_atol
    assert result.work.evaluations <= 2048
    assert result.work.refinements < max_indices
    assert result.error.kind == quad.ErrorKind.SPARSE_GRID_SURPLUS
    assert result.error.norm <= result.tolerance
