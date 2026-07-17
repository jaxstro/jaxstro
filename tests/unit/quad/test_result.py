import jax
import jax.numpy as jnp

from jaxstro import quad
from jaxstro.quad.result import QuadStatus


def _result(value):
    error = quad.QuadError(
        estimate=jnp.abs(value) * 0.0,
        norm=jnp.asarray(0.0),
        kind=jnp.asarray(quad.ErrorKind.EMBEDDED_RULE, dtype=jnp.int32),
        confidence_level=jnp.asarray(jnp.nan),
    )
    work = quad.QuadWork(
        evaluations=jnp.asarray(15, dtype=jnp.int32),
        refinements=jnp.asarray(0, dtype=jnp.int32),
        active_regions=jnp.asarray(1, dtype=jnp.int32),
        levels=jnp.asarray(0, dtype=jnp.int32),
        replicates=jnp.asarray(0, dtype=jnp.int32),
    )
    return quad.QuadResult(
        value=value,
        error=error,
        tolerance=jnp.asarray(1e-8),
        status=jnp.asarray(quad.QuadStatus.CONVERGED, dtype=jnp.int32),
        work=work,
    )


def test_result_fields_are_checkpoint_stable() -> None:
    assert quad.QuadError._fields == (
        "estimate",
        "norm",
        "kind",
        "confidence_level",
    )
    assert quad.QuadWork._fields == (
        "evaluations",
        "refinements",
        "active_regions",
        "levels",
        "replicates",
    )
    assert quad.QuadResult._fields == (
        "value",
        "error",
        "tolerance",
        "status",
        "work",
    )


def test_result_is_a_fixed_shape_jax_pytree() -> None:
    result = _result(jnp.array([1.0, 2.0]))
    leaves, structure = jax.tree.flatten(result)
    rebuilt = jax.tree.unflatten(structure, leaves)
    assert jax.tree.structure(rebuilt) == jax.tree.structure(result)
    assert jnp.array_equal(rebuilt.value, result.value)


def test_status_and_error_codes_are_stable() -> None:
    assert int(quad.QuadStatus.CONVERGED) == 0
    assert int(quad.QuadStatus.MAX_EVALUATIONS) == 1
    assert int(quad.QuadStatus.MAX_REGIONS) == 2
    assert int(quad.QuadStatus.NONFINITE_INTEGRAND) == 3
    assert int(quad.QuadStatus.ROUNDOFF_LIMITED) == 4
    assert int(quad.QuadStatus.DIVERGENCE_SUSPECTED) == 5
    assert int(quad.QuadStatus.INVALID_INPUT) == 6
    assert int(quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE) == 7
    assert int(quad.ErrorKind.EMBEDDED_RULE) == 0
    assert int(quad.ErrorKind.REFINEMENT_DIFFERENCE) == 1
    assert int(quad.ErrorKind.SPARSE_GRID_SURPLUS) == 2
    assert int(quad.ErrorKind.REPLICATE_STANDARD_ERROR) == 3
    assert int(quad.ErrorKind.CONFIDENCE_INTERVAL_HALF_WIDTH) == 4
    assert int(quad.ErrorKind.UNAVAILABLE) == 5


def test_max_indices_appends_without_renumbering_statuses():
    assert QuadStatus.ERROR_ESTIMATE_UNAVAILABLE == 7
    assert QuadStatus.MAX_INDICES == 8
