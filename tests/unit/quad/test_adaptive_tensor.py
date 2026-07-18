import inspect

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


def test_float32_level_13_cost_uses_real_represented_cc_nodes():
    float32_counts = _tensor.represented_cc_axis_counts(
        initial_level=12,
        max_level=13,
        dtype=jnp.float32,
    )
    float64_counts = _tensor.represented_cc_axis_counts(
        initial_level=12,
        max_level=13,
        dtype=jnp.float64,
    )

    assert jnp.array_equal(float32_counts, jnp.array([4097, 8192]))
    assert jnp.array_equal(float64_counts, jnp.array([4097, 8193]))
    assert float32_counts[1] - float32_counts[0] == 4095
    assert float64_counts[1] - float64_counts[0] == 4096


def test_one_shared_represented_cardinality_owner_drives_frontier_cost():
    represented_counts = _tensor.represented_cc_axis_counts(
        initial_level=12,
        max_level=13,
        dtype=jnp.float32,
    )
    levels = jnp.array([12], dtype=jnp.int32)

    accepted, directional = _tensor._represented_formula_cardinalities(
        levels,
        represented_counts=represented_counts,
        initial_level=12,
        max_level=13,
    )

    assert accepted == 4097
    assert jnp.array_equal(directional, jnp.array([4095], dtype=jnp.int32))
    assert _tensor._represented_frontier_cardinality(
        levels,
        represented_counts=represented_counts,
        initial_level=12,
        max_level=13,
    ) == (4097 + 4095)


def test_formula_membership_is_active_only_and_not_capacity_scanned():
    evaluation_source = inspect.getsource(_tensor._evaluate_formula_with_cache)
    lookup_source = inspect.getsource(_tensor._cache_lookup)

    assert "_cache_lookup" in evaluation_source
    assert "lax.fori_loop" in evaluation_source
    assert "lax.scan" not in evaluation_source
    assert "canonical_ids ==" not in evaluation_source
    assert "current.points ==" not in evaluation_source
    assert "hash_slots" in lookup_source
    assert "point_keys[safe_index]" in lookup_source
    assert "lax.while_loop" in lookup_source
    assert not hasattr(_tensor, "_masked_representable_new_count")
    assert not hasattr(_tensor, "_frontier_missing_count")


def test_dormant_quadratic_representable_count_helper_is_removed():
    assert not hasattr(_tensor, "count_representable_new_nodes")
    assert "count_representable_new_nodes" not in _tensor.__all__


@pytest.mark.parametrize("max_evaluations", [65, 512, 2048])
def test_declared_padding_does_not_change_initial_logical_work(max_evaluations):
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(max_evaluations=max_evaluations),
    )

    assert result.status == quad.QuadStatus.CONVERGED
    assert result.work.evaluations == 65
    assert result.work.refinements == 0


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


def test_sharp_axis_end_to_end_level_vector_is_anisotropic():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    capacity = _tensor.validate_adaptive_tensor_capacity(
        initial_level=2,
        dimension=2,
        max_evaluations=512,
        dtype=jnp.float64,
    )
    controller = _tensor.adaptive_tensor_controller(
        lambda x: jnp.exp(8.0 * x[:, 0]) + x[:, 1],
        domain,
        args=(),
        measure=quad.LebesgueMeasure(),
        initial_level=2,
        epsabs=1e-5,
        epsrel=1e-5,
        max_evaluations=512,
        error_norm=quad.MaxNorm(),
        zero=jnp.asarray(0.0),
        capacity=capacity,
    )

    assert jnp.array_equal(controller.levels, jnp.array([3, 2]))
    assert controller.refinements == 1
    assert controller.evaluations == 121


def _run_unequal_cost_controller():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    capacity = _tensor.validate_adaptive_tensor_capacity(
        initial_level=2,
        dimension=2,
        max_evaluations=230,
        dtype=jnp.float64,
    )
    return _tensor.adaptive_tensor_controller(
        lambda x: jnp.exp(12.0 * x[:, 0]) + 0.15 * jnp.exp(8.0 * x[:, 1]),
        domain,
        args=(),
        measure=quad.LebesgueMeasure(),
        initial_level=2,
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=230,
        error_norm=quad.MaxNorm(),
        zero=jnp.asarray(0.0),
        capacity=capacity,
    )


def test_controller_uses_unequal_cost_profit_not_raw_directional_error(monkeypatch):
    profit_selected = _run_unequal_cost_controller()

    assert jnp.array_equal(profit_selected.levels, jnp.array([3, 3]))
    assert profit_selected.refinements == 2
    assert profit_selected.evaluations == 225

    monkeypatch.setattr(
        _tensor,
        "choose_tensor_axis",
        lambda directional_error, _new_cost: (
            jnp.argmax(directional_error),
            jnp.sum(directional_error),
        ),
    )
    raw_error_selected = _run_unequal_cost_controller()

    assert jnp.array_equal(raw_error_selected.levels, jnp.array([3, 2]))
    assert raw_error_selected.refinements == 1
    assert raw_error_selected.evaluations == 121


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


def test_oversized_initial_level_rejects_from_a_low_nested_bound(monkeypatch):
    visited_levels = []
    original_cc_unit_nodes = _tensor._cc_unit_nodes

    def bounded_cc_unit_nodes(level, dtype):
        visited_levels.append(level)
        if level >= 4:
            raise AssertionError("oversized CC metadata must not be constructed")
        return original_cc_unit_nodes(level, dtype)

    def fail_payload_inference(*_args, **_kwargs):
        raise AssertionError("payload inference must follow capacity validation")

    _tensor._adaptive_tensor_capacity_cached.cache_clear()
    _tensor._represented_cc_axis_metadata_cached.cache_clear()
    monkeypatch.setattr(_tensor, "_cc_unit_nodes", bounded_cc_unit_nodes)
    monkeypatch.setattr(
        tensor,
        "infer_multidim_payload_zero",
        fail_payload_inference,
    )

    with pytest.raises(ValueError) as exc_info:
        quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.AdaptiveTensorClenshawCurtis(initial_level=18),
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=65,
            gradient="stop",
        )

    assert str(exc_info.value) == (
        "initial adaptive tensor frontier requires at least 81 evaluations "
        "by level 3, exceeding max_evaluations=65"
    )
    assert visited_levels == [0, 1, 2, 3]


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
    def zero_represented_growth(
        levels,
        *,
        represented_counts,
        initial_level,
        max_level,
    ):
        del represented_counts, initial_level, max_level
        return (
            jnp.prod(jnp.left_shift(1, levels) + 1),
            jnp.zeros_like(levels),
        )

    monkeypatch.setattr(
        _tensor,
        "_represented_formula_cardinalities",
        zero_represented_growth,
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


@pytest.mark.parametrize("gradient", ["through", "invalid"])
def test_adaptive_tensor_rejects_unknown_gradient_modes(gradient):
    with pytest.raises(ValueError) as exc_info:
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.AdaptiveTensorClenshawCurtis(),
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=65,
            gradient=gradient,
        )
    assert str(exc_info.value) == 'gradient must be "replay" or "stop"'


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
