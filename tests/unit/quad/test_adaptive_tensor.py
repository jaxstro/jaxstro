import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad import _tensor, tensor
from jaxstro.quantity import units as q_units


def _options(*, max_evaluations=512, epsabs=1e-10, epsrel=1e-10):
    return dict(
        method=quad.AdaptiveTensorClenshawCurtis(initial_level=2),
        epsabs=epsabs,
        epsrel=epsrel,
        max_evaluations=max_evaluations,
        gradient="stop",
    )


@pytest.mark.parametrize("initial_level", [True, 1, 2.0, "2"])
def test_adaptive_tensor_rejects_invalid_initial_level(initial_level):
    with pytest.raises(
        ValueError,
        match="initial_level must be an integer at least 2",
    ):
        quad.AdaptiveTensorClenshawCurtis(initial_level=initial_level)


def test_adaptive_tensor_declaration_is_static_pytree_metadata():
    method = quad.AdaptiveTensorClenshawCurtis(initial_level=3)

    leaves, structure = jax.tree.flatten(method)
    rebuilt = jax.tree.unflatten(structure, leaves)

    assert leaves == []
    assert rebuilt == method


def test_canonical_axis_ids_reduce_nested_dyadic_angles_exactly():
    assert jnp.array_equal(
        _tensor.canonical_cc_axis_ids(2),
        jnp.array(
            [
                [0, 0],
                [1, 2],
                [1, 1],
                [3, 2],
                [1, 0],
            ],
            dtype=jnp.int32,
        ),
    )


def test_nested_reuse_counts_only_new_coordinate_tuples():
    coarse = _tensor.canonical_tensor_ids(jnp.array([2, 2]))
    fine = _tensor.canonical_tensor_ids(jnp.array([3, 2]))
    coarse_ids = set(map(tuple, coarse.tolist()))
    fine_ids = set(map(tuple, fine.tolist()))

    assert len(coarse_ids) == coarse.shape[0]
    assert len(fine_ids) == fine.shape[0]
    assert coarse_ids.issubset(fine_ids)
    assert fine.shape[0] - coarse.shape[0] == 20


def test_frontier_selection_uses_directional_profit_and_sums_evidence():
    axis, evidence = _tensor.choose_tensor_axis(
        jnp.array([3.0, 2.0]),
        jnp.array([3, 1]),
    )

    assert axis == 1
    assert evidence == 5.0


def test_frontier_profit_ties_choose_the_lowest_axis():
    axis, evidence = _tensor.choose_tensor_axis(
        jnp.array([3.0, 1.0]),
        jnp.array([3, 1]),
    )

    assert axis == 0
    assert evidence == 4.0


def test_representable_new_node_count_detects_collapsed_coordinates():
    base = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    collapsed = jnp.array([[0.0, 0.0], [0.5, 0.5], [0.5, 0.5], [1.0, 1.0]])

    assert _tensor.count_representable_new_nodes(base, collapsed) == 0


def test_adaptive_tensor_refines_under_directional_frontier_evidence():
    result = quad.integrate(
        lambda x: jnp.exp(8.0 * x[:, 0]) + x[:, 1],
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(epsabs=1e-5, epsrel=1e-5),
    )

    expected = (jnp.exp(8.0) - 1.0) / 8.0 + 0.5
    assert jnp.allclose(result.value, expected, rtol=0.0, atol=2e-3)
    assert result.error.kind == quad.ErrorKind.REFINEMENT_DIFFERENCE
    assert result.work.refinements == 1
    assert result.work.evaluations == 121
    assert result.work.levels == 3
    assert result.work.active_regions == 0
    assert result.work.replicates == 0


def test_initial_frontier_reuses_every_shared_coordinate_tuple():
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(max_evaluations=65),
    )

    assert jnp.allclose(
        result.value, 1.0, rtol=0.0, atol=4.0 * jnp.finfo(jnp.float64).eps
    )
    assert result.error.estimate <= 4.0 * jnp.finfo(jnp.float64).eps
    assert result.error.norm <= 4.0 * jnp.finfo(jnp.float64).eps
    assert result.status == quad.QuadStatus.CONVERGED
    assert result.work.evaluations == 65
    assert result.work.refinements == 0
    assert result.work.levels == 2


def test_initial_frontier_capacity_is_validated_before_payload_inference(monkeypatch):
    def fail_payload_inference(*_args, **_kwargs):
        raise AssertionError("payload inference must follow capacity validation")

    monkeypatch.setattr(
        tensor,
        "infer_multidim_payload_zero",
        fail_payload_inference,
    )

    with pytest.raises(
        ValueError,
        match="initial adaptive tensor frontier requires 65 evaluations",
    ):
        quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            **_options(max_evaluations=64),
        )


def test_convergence_precedes_capacity_at_the_exact_initial_frontier_budget():
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(max_evaluations=65),
    )

    assert result.status == quad.QuadStatus.CONVERGED


def test_capacity_status_follows_nonconverged_initial_frontier():
    result = quad.integrate(
        lambda x: jnp.exp(8.0 * x[:, 0]) + x[:, 1],
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=65,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )

    assert result.status == quad.QuadStatus.MAX_EVALUATIONS
    assert result.work.evaluations == 65
    assert result.work.refinements == 0


def test_roundoff_precedes_capacity_when_selected_refinement_collapses(monkeypatch):
    monkeypatch.setattr(
        _tensor,
        "_selected_representable_new_count",
        lambda *_args, **_kwargs: jnp.asarray(0, dtype=jnp.int32),
    )

    result = quad.integrate(
        lambda x: jnp.exp(8.0 * x[:, 0]) + x[:, 1],
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=65,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )

    assert result.status == quad.QuadStatus.ROUNDOFF_LIMITED
    assert result.work.evaluations == 65
    assert result.work.refinements == 0


def test_traced_invalid_tolerance_returns_fail_closed_without_work():
    @jax.jit
    def solve(epsabs):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            **_options(max_evaluations=65, epsabs=epsabs),
        )

    result = solve(jnp.asarray(-1.0))

    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 0


def test_invalid_bound_precedes_zero_volume_without_frontier_work():
    @jax.jit
    def solve(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            **_options(max_evaluations=65),
        )

    result = solve(jnp.array([0.0, jnp.inf]))

    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 0


def test_nonfinite_integrand_precedes_capacity_and_fails_closed():
    result = quad.integrate(
        lambda x: jnp.where(x[:, 0] == 0.0, jnp.nan, 1.0),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=65,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )

    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)
    assert 0 < result.work.evaluations <= 65


def test_nonfinite_density_returns_nonfinite_integrand():
    measure = quad.WeightedMeasure(
        lambda x, _args: jnp.where(x[:, 0] == 0.0, jnp.inf, 1.0),
        density_unit=q_units.dimensionless,
    )
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        measure=measure,
        **_options(max_evaluations=65),
    )

    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)


def test_zero_volume_returns_exact_payload_zero_without_point_work():
    result = quad.integrate(
        lambda x: jnp.stack((x[:, 0], 1j * x[:, 1]), axis=-1),
        quad.Hyperrectangle(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
        **_options(max_evaluations=65),
    )

    assert jnp.array_equal(result.value, jnp.zeros(2, dtype=jnp.complex128))
    assert result.error.kind == quad.ErrorKind.UNAVAILABLE
    assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
    assert result.work.evaluations == 0
    assert result.work.refinements == 0


def test_array_and_complex_payloads_preserve_value_and_evidence_contracts():
    result = quad.integrate(
        lambda x: jnp.stack(
            (
                x[:, 0] + x[:, 1],
                1j * (x[:, 0] - x[:, 1]),
            ),
            axis=-1,
        ),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(max_evaluations=65),
    )

    assert jnp.allclose(result.value, jnp.array([1.0 + 0.0j, 0.0 + 0.0j]))
    assert result.error.estimate.shape == result.value.shape
    assert result.error.norm.shape == ()
    assert result.error.kind == quad.ErrorKind.REFINEMENT_DIFFERENCE
    assert result.status == quad.QuadStatus.CONVERGED


@pytest.mark.parametrize("dimension", [1, 9])
def test_adaptive_tensor_rejects_dimensions_outside_b1_envelope(dimension):
    with pytest.raises(
        ValueError,
        match="Phase B1 deterministic methods require dimension 2 through 8",
    ):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(dimension), jnp.ones(dimension)),
            **_options(max_evaluations=4096),
        )


@pytest.mark.parametrize("gradient", ["replay", "through", "invalid"])
def test_adaptive_tensor_rejects_every_gradient_mode_except_stop(gradient):
    with pytest.raises(
        ValueError,
        match='AdaptiveTensorClenshawCurtis requires gradient="stop"',
    ):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.AdaptiveTensorClenshawCurtis(),
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=65,
            gradient=gradient,
        )


def test_adaptive_tensor_stop_mode_has_exact_zero_grad_and_jvp_tangents():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def parameterized(scale):
        return quad.integrate(
            lambda x, factor: factor * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            **_options(max_evaluations=65),
        ).value

    derivative = jax.grad(parameterized)(jnp.asarray(2.0))
    _, tangent = jax.jvp(
        parameterized,
        (jnp.asarray(2.0),),
        (jnp.asarray(1.0),),
    )

    assert jnp.array_equal(derivative, 0.0)
    assert jnp.array_equal(tangent, 0.0)


def test_adaptive_tensor_composes_with_jit_and_vmap_in_stop_mode():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    @jax.jit
    def solve(scale):
        return quad.integrate(
            lambda x, factor: factor * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            **_options(max_evaluations=65),
        ).value

    values = jax.vmap(solve)(jnp.array([1.0, 2.0, 3.0]))

    assert jnp.allclose(values, jnp.array([1.0, 2.0, 3.0]))
