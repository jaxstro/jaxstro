import inspect

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

from jaxstro import quad
from jaxstro.quad import _cubature, cubature
from jaxstro.quantity import units as q_units


def _options(
    *,
    max_evaluations=5_000,
    max_regions=64,
    epsabs=1e-10,
    epsrel=1e-10,
):
    return dict(
        method=quad.AdaptiveCubature(),
        epsabs=epsabs,
        epsrel=epsrel,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
        gradient="stop",
    )


def test_adaptive_cubature_declaration_is_static_and_requires_genz_malik():
    method = quad.AdaptiveCubature()
    leaves, structure = jax.tree.flatten(method)
    rebuilt = jax.tree.unflatten(structure, leaves)

    assert leaves == []
    assert rebuilt == method
    assert isinstance(method.rule, quad.GenzMalik)
    with pytest.raises(
        TypeError,
        match="AdaptiveCubature requires GenzMalik in Phase B1",
    ):
        quad.AdaptiveCubature(rule=object())


def test_adaptive_cubature_and_rule_are_public_but_replay_evidence_is_private():
    assert "AdaptiveCubature" in quad.__all__
    assert "GenzMalik" in quad.__all__
    assert quad.AdaptiveCubature.__module__ == "jaxstro.quad.cubature"
    assert quad.GenzMalik.__module__ == "jaxstro.quad.cubature"
    assert not hasattr(quad, "CubatureReplayEvidence")
    assert "CubatureReplayEvidence" not in _cubature.__all__


def test_adaptive_cubature_documents_scalar_vmap_and_lax_map_cost_contracts():
    documentation = " ".join(
        (
            inspect.getdoc(quad.AdaptiveCubature) or "",
            inspect.getdoc(quad.integrate) or "",
        )
    ).lower()

    assert "scalar" in documentation
    assert "vmap" in documentation
    assert "logical work" in documentation
    assert "physical" in documentation
    assert "lax.map" in documentation


@pytest.mark.parametrize("dimension", range(2, 9))
def test_cubature_initial_rule_count_and_dimension_envelope(dimension):
    count = 2**dimension + 2 * dimension**2 + 2 * dimension + 1
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(dimension), jnp.ones(dimension)),
        **_options(
            max_evaluations=count,
            max_regions=1,
            epsabs=1e-10,
            epsrel=0.0,
        ),
    )

    assert _cubature.genz_malik_point_count(dimension) == count
    assert jnp.allclose(result.value, 1.0, rtol=0.0, atol=2e-13)
    assert result.status == quad.QuadStatus.CONVERGED
    assert result.error.kind == quad.ErrorKind.EMBEDDED_RULE
    assert result.work.evaluations == count
    assert result.work.refinements == 0
    assert result.work.active_regions == 1
    assert result.work.levels == 0
    assert result.work.replicates == 0


@pytest.mark.parametrize("dimension", [1, 9])
def test_cubature_rejects_dimensions_outside_b1_envelope(dimension):
    with pytest.raises(
        ValueError,
        match="Phase B1 deterministic methods require dimension 2 through 8",
    ):
        quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(dimension), jnp.ones(dimension)),
            **_options(max_evaluations=10_000),
        )


def test_initial_rule_capacity_precedes_materialization_and_payload_inference(
    monkeypatch,
):
    point_count = _cubature.genz_malik_point_count(2)

    def fail_rule(*_args, **_kwargs):
        raise AssertionError("rule materialization must follow capacity validation")

    def fail_payload(*_args, **_kwargs):
        raise AssertionError("payload inference must follow capacity validation")

    monkeypatch.setattr(cubature, "genz_malik_data", fail_rule)
    monkeypatch.setattr(cubature, "infer_multidim_payload_zero", fail_payload)

    with pytest.raises(
        ValueError,
        match=rf"initial Genz-Malik rule requires {point_count} evaluations",
    ):
        quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(2), jnp.zeros(2)),
            **_options(max_evaluations=point_count - 1),
        )


@pytest.mark.parametrize("max_regions", [None, 0, -1, True, 2.5, "4"])
def test_max_regions_is_validated_eagerly(max_regions):
    with pytest.raises(ValueError, match="max_regions must be a positive integer"):
        quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            **_options(max_regions=max_regions),
        )


def test_reachable_region_store_has_no_arbitrary_private_row_ceiling():
    point_count = _cubature.genz_malik_point_count(2)
    requested_regions = 100_001
    capacity = _cubature.validate_cubature_capacity(
        dimension=2,
        max_evaluations=point_count * (1 + 2 * (requested_regions - 1)),
        max_regions=requested_regions,
    )

    assert capacity.store_capacity == requested_regions
    assert capacity.max_refinements == requested_regions - 1


def test_reachable_region_store_rejects_only_derived_int32_shape_overflow():
    point_count = _cubature.genz_malik_point_count(2)
    max_int32 = 2**31 - 1

    with pytest.raises(
        ValueError,
        match="reachable cubature work exceeds JAX int32 indexing",
    ):
        _cubature.validate_cubature_capacity(
            dimension=2,
            max_evaluations=point_count * (1 + 2 * max_int32),
            max_regions=max_int32 + 1,
        )


def test_huge_declared_regions_do_not_allocate_when_evaluation_budget_is_small():
    point_count = _cubature.genz_malik_point_count(2)
    capacity = _cubature.validate_cubature_capacity(
        dimension=2,
        max_evaluations=point_count,
        max_regions=10**12,
    )

    assert capacity.store_capacity == 1
    assert capacity.max_refinements == 0
    assert capacity.evaluation_refinement_limit == 0
    assert capacity.region_refinement_limit == 1


def test_cubature_integrates_array_payload_and_counts_points():
    result = quad.integrate(
        lambda x: jnp.stack(
            (jnp.ones(x.shape[0]), x[:, 0] * x[:, 1]),
            axis=-1,
        ),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(max_evaluations=5_000, max_regions=64, epsabs=1e-9, epsrel=1e-9),
    )

    assert jnp.allclose(result.value, jnp.array([1.0, 0.25]), atol=1e-9)
    assert result.error.kind == quad.ErrorKind.EMBEDDED_RULE
    assert result.status == quad.QuadStatus.CONVERGED
    assert result.work.active_regions == result.work.refinements + 1


def test_signed_orientation_density_and_complex_payload_are_preserved():
    measure = quad.WeightedMeasure(
        lambda x, _args: 2.0 * x[:, 0],
        density_unit=q_units.dimensionless,
    )
    result = quad.integrate(
        lambda x: jnp.stack(
            (jnp.ones(x.shape[0]), 1j * x[:, 1]),
            axis=-1,
        ),
        quad.Hyperrectangle(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])),
        measure=measure,
        **_options(epsabs=1e-9, epsrel=1e-9),
    )

    assert jnp.allclose(
        result.value,
        jnp.array([-1.0 + 0.0j, -0.5j]),
        rtol=0.0,
        atol=2e-12,
    )
    assert result.error.estimate.shape == (2,)
    assert result.error.norm.shape == ()
    assert result.status == quad.QuadStatus.CONVERGED


def test_cubature_matches_analytic_genz_family_anchors():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    options = _options(
        max_evaluations=20_000,
        max_regions=512,
        epsabs=1e-8,
        epsrel=1e-8,
    )

    oscillatory_scale = jnp.array([2.0, 3.0])
    phase = jnp.asarray(0.3)
    oscillatory = quad.integrate(
        lambda x: jnp.cos(phase + x @ oscillatory_scale),
        domain,
        **options,
    )
    complex_factors = (jnp.exp(1j * oscillatory_scale) - 1.0) / (1j * oscillatory_scale)
    expected_oscillatory = jnp.real(jnp.exp(1j * phase) * jnp.prod(complex_factors))

    gaussian_scale = jnp.array([3.0, 5.0])
    gaussian_center = jnp.array([0.3, 0.7])
    gaussian = quad.integrate(
        lambda x: jnp.exp(
            -jnp.sum(
                gaussian_scale**2 * (x - gaussian_center) ** 2,
                axis=-1,
            )
        ),
        domain,
        **options,
    )
    gaussian_factors = (
        jnp.sqrt(jnp.pi)
        / (2.0 * gaussian_scale)
        * (
            jsp.special.erf(gaussian_scale * (1.0 - gaussian_center))
            + jsp.special.erf(gaussian_scale * gaussian_center)
        )
    )
    expected_gaussian = jnp.prod(gaussian_factors)

    peak_scale = jnp.array([4.0, 6.0])
    peak_center = jnp.array([0.25, 0.65])
    product_peak = quad.integrate(
        lambda x: jnp.prod(
            1.0 / (peak_scale**-2 + (x - peak_center) ** 2),
            axis=-1,
        ),
        domain,
        **options,
    )
    peak_factors = peak_scale * (
        jnp.arctan(peak_scale * (1.0 - peak_center))
        + jnp.arctan(peak_scale * peak_center)
    )
    expected_product_peak = jnp.prod(peak_factors)

    for result, expected in (
        (oscillatory, expected_oscillatory),
        (gaussian, expected_gaussian),
        (product_peak, expected_product_peak),
    ):
        assert result.status == quad.QuadStatus.CONVERGED
        assert jnp.allclose(result.value, expected, rtol=2e-8, atol=2e-9)
        assert jnp.abs(result.value - expected) <= 2.0 * result.tolerance


def test_region_and_axis_ties_choose_the_lowest_index():
    assert (
        _cubature.select_region(
            jnp.array([True, True, False]),
            jnp.array([4.0, 4.0, 100.0]),
        )
        == 0
    )
    assert (
        _cubature.select_region(
            jnp.array([True, True, True]),
            jnp.array([1.0, 4.0, 4.0]),
        )
        == 1
    )
    assert _cubature.select_split_axis(jnp.array([3.0, 3.0])) == 0


def test_global_value_and_error_equal_the_final_active_leaf_sum():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    point_count = _cubature.genz_malik_point_count(2)
    capacity = _cubature.validate_cubature_capacity(
        dimension=2,
        max_evaluations=5 * point_count,
        max_regions=3,
    )
    data = _cubature.genz_malik_data(2, jnp.float64)
    controller = _cubature.cubature_controller(
        lambda x: jnp.exp(20.0 * jnp.sum(x, axis=-1)),
        domain,
        args=(),
        measure=quad.LebesgueMeasure(),
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=5 * point_count,
        max_regions=3,
        error_norm=quad.MaxNorm(),
        zero=jnp.asarray(0.0),
        data=data,
        capacity=capacity,
    )
    active_count = int(controller.active_regions)
    leaf_estimates = _cubature.evaluate_cubature_regions(
        lambda x: jnp.exp(20.0 * jnp.sum(x, axis=-1)),
        domain,
        controller.evidence.lower[:active_count],
        controller.evidence.upper[:active_count],
        args=(),
        measure=quad.LebesgueMeasure(),
        error_norm=quad.MaxNorm(),
        data=data,
    )

    assert controller.status == quad.QuadStatus.MAX_EVALUATIONS
    assert jnp.all(controller.evidence.active[:active_count])
    assert not jnp.any(controller.evidence.active[active_count:])
    assert jnp.allclose(
        controller.value,
        jnp.sum(leaf_estimates.value, axis=0),
        rtol=2e-15,
        atol=0.0,
    )
    assert jnp.allclose(
        controller.error,
        jnp.sum(leaf_estimates.error, axis=0),
        rtol=2e-15,
        atol=0.0,
    )


def test_float32_five_leaf_reduction_cannot_false_converge(monkeypatch):
    """A signed parent delta goes negative while five leaf errors stay positive."""
    point_count = _cubature.genz_malik_point_count(2)
    capacity = _cubature.validate_cubature_capacity(
        dimension=2,
        max_evaluations=point_count * 11,
        max_regions=5,
    )
    data = _cubature.genz_malik_data(2, jnp.float32)
    half_ulp_at_one = jnp.asarray(2.0**-24, dtype=jnp.float32)

    def scripted_regions(
        _fun,
        _domain,
        lower,
        upper,
        *,
        args,
        measure,
        error_norm,
        data,
    ):
        del args, measure, error_norm, data
        lower = jnp.asarray(lower, dtype=jnp.float32)
        upper = jnp.asarray(upper, dtype=jnp.float32)
        start = lower[:, 0]
        width = upper[:, 0] - start
        one = jnp.asarray(1.0, dtype=jnp.float32)
        large = (start == 0.0) & ((width == 1.0) | (width == 0.5) | (width == 0.25))
        tiny = ((start == 0.5) & (width == 0.5)) | ((start == 0.25) & (width == 0.25))
        quarter_tiny = ((start == 0.0) | (start == 0.125)) & (width == 0.125)
        eighth_tiny = ((start == 0.5) | (start == 0.75)) & (width == 0.25)
        local = jnp.where(
            large,
            one,
            jnp.where(
                tiny,
                half_ulp_at_one,
                jnp.where(
                    quarter_tiny,
                    half_ulp_at_one / 4.0,
                    jnp.where(eighth_tiny, half_ulp_at_one / 8.0, 0.0),
                ),
            ),
        )
        return _cubature.CubatureRegionEstimate(
            value=local,
            error=local,
            error_norm=local,
            split_axis=jnp.zeros(local.shape, dtype=jnp.int32),
            nonfinite=jnp.zeros(local.shape, dtype=jnp.bool_),
        )

    monkeypatch.setattr(
        _cubature,
        "evaluate_cubature_regions",
        scripted_regions,
    )
    controller = _cubature.cubature_controller(
        lambda x: jnp.zeros(x.shape[0], dtype=jnp.float32),
        quad.Hyperrectangle(
            jnp.zeros(2, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32),
        ),
        args=(),
        measure=quad.LebesgueMeasure(),
        epsabs=half_ulp_at_one / 4.0,
        epsrel=0.0,
        max_evaluations=point_count * 11,
        max_regions=5,
        error_norm=quad.MaxNorm(),
        zero=jnp.asarray(0.0, dtype=jnp.float32),
        data=data,
        capacity=capacity,
    )
    leaves = scripted_regions(
        None,
        None,
        controller.evidence.lower,
        controller.evidence.upper,
        args=(),
        measure=None,
        error_norm=quad.MaxNorm(),
        data=data,
    )
    active_error = jnp.sum(jnp.where(controller.evidence.active, leaves.error, 0.0))
    active_value = jnp.sum(jnp.where(controller.evidence.active, leaves.value, 0.0))

    assert controller.active_regions == 5
    assert controller.status == quad.QuadStatus.MAX_REGIONS
    assert active_error == 1.75 * half_ulp_at_one
    assert controller.error == active_error
    assert controller.error >= 0.0
    assert controller.error_norm == active_error
    assert controller.error_norm > controller.tolerance
    assert controller.value == active_value


def test_exact_work_active_regions_and_deepest_depth_after_two_splits():
    point_count = _cubature.genz_malik_point_count(2)
    result = quad.integrate(
        lambda x: jnp.exp(20.0 * jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=10_000,
            max_regions=3,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )

    assert result.status == quad.QuadStatus.MAX_REGIONS
    assert result.work.evaluations == 5 * point_count
    assert result.work.refinements == 2
    assert result.work.active_regions == 3
    assert result.work.active_regions == result.work.refinements + 1
    assert result.work.levels == 2


def _record_region_batch_sizes(monkeypatch):
    batch_sizes = []
    original = _cubature.evaluate_cubature_regions

    def recording(*args, **kwargs):
        lower = jnp.asarray(args[2])
        jax.debug.callback(
            lambda size: batch_sizes.append(int(size)),
            jnp.asarray(lower.shape[0], dtype=jnp.int32),
            ordered=True,
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(_cubature, "evaluate_cubature_regions", recording)
    return batch_sizes


def test_evaluation_capacity_exhaustion_evaluates_no_child(monkeypatch):
    point_count = _cubature.genz_malik_point_count(2)
    batches = _record_region_batch_sizes(monkeypatch)
    result = quad.integrate(
        lambda x: jnp.exp(20.0 * jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=2 * point_count,
            max_regions=3,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )
    jax.block_until_ready(result.value)

    assert result.status == quad.QuadStatus.MAX_EVALUATIONS
    assert result.work.evaluations == point_count
    assert result.work.refinements == 0
    assert batches == [1]


def test_region_capacity_exhaustion_evaluates_no_child(monkeypatch):
    point_count = _cubature.genz_malik_point_count(2)
    batches = _record_region_batch_sizes(monkeypatch)
    result = quad.integrate(
        lambda x: jnp.exp(20.0 * jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=3 * point_count,
            max_regions=1,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )
    jax.block_until_ready(result.value)

    assert result.status == quad.QuadStatus.MAX_REGIONS
    assert result.work.evaluations == point_count
    assert result.work.refinements == 0
    assert batches == [1]


def test_jit_convergence_physically_evaluates_no_child(monkeypatch):
    point_count = _cubature.genz_malik_point_count(2)
    batches = _record_region_batch_sizes(monkeypatch)

    @jax.jit
    def solve():
        return quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            **_options(
                max_evaluations=3 * point_count,
                max_regions=2,
                epsabs=1e-10,
                epsrel=0.0,
            ),
        )

    result = solve()
    jax.block_until_ready(result.value)

    assert result.status == quad.QuadStatus.CONVERGED
    assert result.work.evaluations == point_count
    assert batches == [1]


def _heterogeneous_cubature_solve(kind):
    point_count = _cubature.genz_malik_point_count(2)
    return quad.integrate(
        lambda x, selected: jnp.where(
            selected == 0,
            jnp.ones(x.shape[0]),
            jnp.exp(20.0 * jnp.sum(x, axis=-1)),
        ),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        args=kind,
        **_options(
            max_evaluations=5 * point_count,
            max_regions=2,
            epsabs=1e-10,
            epsrel=0.0,
        ),
    )


def test_heterogeneous_vmap_preserves_result_semantics_and_logical_work():
    kinds = jnp.asarray([0, 1], dtype=jnp.int32)
    mapped = jax.jit(jax.vmap(_heterogeneous_cubature_solve))(kinds)
    scalar = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[_heterogeneous_cubature_solve(kind) for kind in kinds],
    )
    equal = jax.tree.map(
        lambda actual, expected: jnp.array_equal(
            actual,
            expected,
            equal_nan=True,
        ),
        mapped,
        scalar,
    )
    point_count = _cubature.genz_malik_point_count(2)

    assert all(bool(value) for value in jax.tree.leaves(equal))
    assert jnp.array_equal(
        mapped.status,
        jnp.asarray(
            [quad.QuadStatus.CONVERGED, quad.QuadStatus.MAX_REGIONS],
            dtype=jnp.int32,
        ),
    )
    assert jnp.array_equal(
        mapped.work.evaluations,
        jnp.asarray([point_count, 3 * point_count], dtype=jnp.int32),
    )


def test_lax_map_heterogeneous_batch_physically_skips_converged_child(
    monkeypatch,
):
    batches = _record_region_batch_sizes(monkeypatch)
    kinds = jnp.asarray([0, 1], dtype=jnp.int32)

    mapped = jax.lax.map(_heterogeneous_cubature_solve, kinds)
    jax.block_until_ready(mapped.value)

    assert mapped.status[0] == quad.QuadStatus.CONVERGED
    assert mapped.status[1] == quad.QuadStatus.MAX_REGIONS
    assert batches.count(1) == 2
    assert batches.count(2) == 1


def test_max_evaluations_precedes_max_regions_when_both_are_exhausted():
    point_count = _cubature.genz_malik_point_count(2)
    result = quad.integrate(
        lambda x: jnp.exp(20.0 * jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=point_count,
            max_regions=1,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )

    assert result.status == quad.QuadStatus.MAX_EVALUATIONS
    assert result.work.evaluations == point_count


def test_convergence_precedes_both_capacity_statuses():
    point_count = _cubature.genz_malik_point_count(2)
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=point_count,
            max_regions=1,
            epsabs=1e-10,
            epsrel=0.0,
        ),
    )

    assert result.status == quad.QuadStatus.CONVERGED


def test_roundoff_status_is_only_the_selected_midpoint_endpoint_collapse():
    running = _cubature.cubature_termination_status(
        nonfinite=False,
        converged=False,
        midpoint_collapsed=False,
        has_evaluation_capacity=True,
        has_region_capacity=True,
    )
    collapsed = _cubature.cubature_termination_status(
        nonfinite=False,
        converged=False,
        midpoint_collapsed=True,
        has_evaluation_capacity=True,
        has_region_capacity=True,
    )
    converged = _cubature.cubature_termination_status(
        nonfinite=False,
        converged=True,
        midpoint_collapsed=True,
        has_evaluation_capacity=False,
        has_region_capacity=False,
    )

    assert running == _cubature.RUNNING
    assert collapsed == quad.QuadStatus.ROUNDOFF_LIMITED
    assert converged == quad.QuadStatus.CONVERGED
    source = inspect.getsource(_cubature.cubature_controller)
    assert "DIVERGENCE_SUSPECTED" not in source
    assert "no_improvement" not in source
    assert "growth_count" not in source


def test_midpoint_collapse_precedes_capacity_but_follows_nonfinite():
    roundoff = _cubature.cubature_termination_status(
        nonfinite=False,
        converged=False,
        midpoint_collapsed=True,
        has_evaluation_capacity=False,
        has_region_capacity=False,
    )
    nonfinite = _cubature.cubature_termination_status(
        nonfinite=True,
        converged=True,
        midpoint_collapsed=True,
        has_evaluation_capacity=False,
        has_region_capacity=False,
    )

    assert roundoff == quad.QuadStatus.ROUNDOFF_LIMITED
    assert nonfinite == quad.QuadStatus.NONFINITE_INTEGRAND


def test_invalid_tolerance_precedes_zero_volume_without_point_work():
    point_count = _cubature.genz_malik_point_count(2)

    @jax.jit
    def solve(epsabs):
        return quad.integrate(
            lambda x: jnp.stack((x[:, 0], 1j * x[:, 1]), axis=-1),
            quad.Hyperrectangle(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
            **_options(
                max_evaluations=point_count,
                max_regions=1,
                epsabs=epsabs,
            ),
        )

    result = solve(jnp.asarray(-1.0))

    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.all(jnp.isnan(result.value))
    assert result.work.evaluations == 0


def test_invalid_bound_precedes_zero_volume_without_point_work():
    point_count = _cubature.genz_malik_point_count(2)

    @jax.jit
    def solve(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            **_options(max_evaluations=point_count, max_regions=1),
        )

    result = solve(jnp.array([0.0, jnp.inf]))

    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 0


def test_zero_volume_returns_exact_complex_payload_zero_without_rule_work():
    point_count = _cubature.genz_malik_point_count(2)
    result = quad.integrate(
        lambda x: jnp.stack((x[:, 0], 1j * x[:, 1]), axis=-1),
        quad.Hyperrectangle(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
        **_options(max_evaluations=point_count, max_regions=1),
    )

    assert jnp.array_equal(result.value, jnp.zeros(2, dtype=jnp.complex128))
    assert result.error.kind == quad.ErrorKind.UNAVAILABLE
    assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
    assert result.work.evaluations == 0
    assert result.work.refinements == 0


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_integrand_fails_closed_before_capacity(bad):
    point_count = _cubature.genz_malik_point_count(2)
    result = quad.integrate(
        lambda x: jnp.full((x.shape[0], 2), bad),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            max_evaluations=point_count,
            max_regions=1,
            epsabs=0.0,
            epsrel=0.0,
        ),
    )

    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.all(jnp.isnan(result.value))
    assert jnp.all(jnp.isnan(result.error.estimate))
    assert jnp.isnan(result.error.norm)
    assert result.work.evaluations == point_count


def test_nonfinite_density_returns_nonfinite_integrand():
    point_count = _cubature.genz_malik_point_count(2)
    measure = quad.WeightedMeasure(
        lambda x, _args: jnp.full(x.shape[0], jnp.inf),
        density_unit=q_units.dimensionless,
    )
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        measure=measure,
        **_options(max_evaluations=point_count, max_regions=1),
    )

    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)


def test_finite_payload_derived_reduction_overflow_fails_closed():
    dimension = 8
    point_count = _cubature.genz_malik_point_count(dimension)
    scale = jnp.asarray(0.6 * jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    result = quad.integrate(
        lambda x: jnp.full((x.shape[0],), scale, dtype=jnp.float32),
        quad.Hyperrectangle(
            jnp.zeros(dimension, dtype=jnp.float32),
            jnp.ones(dimension, dtype=jnp.float32),
        ),
        **_options(max_evaluations=point_count, max_regions=1),
    )

    assert jnp.isfinite(scale)
    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)
    assert jnp.isnan(result.error.estimate)


def test_dimension_eight_float32_embedded_floor_does_not_false_converge():
    dimension = 8
    point_count = _cubature.genz_malik_point_count(dimension)
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0], dtype=jnp.float32),
        quad.Hyperrectangle(
            jnp.zeros(dimension, dtype=jnp.float32),
            jnp.ones(dimension, dtype=jnp.float32),
        ),
        **_options(
            max_evaluations=3 * point_count,
            max_regions=1,
            epsabs=1e-8,
            epsrel=0.0,
        ),
    )

    assert result.error.norm >= jnp.asarray(1.0e-6, dtype=jnp.float32)
    assert result.error.norm < jnp.asarray(1.2e-6, dtype=jnp.float32)
    assert result.tolerance == jnp.asarray(1e-8, dtype=jnp.float32)
    assert result.status == quad.QuadStatus.MAX_REGIONS
    assert result.status != quad.QuadStatus.ROUNDOFF_LIMITED
    assert result.work.evaluations == point_count


def test_stop_mode_composes_with_jit_vmap_grad_and_jvp():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    point_count = _cubature.genz_malik_point_count(2)

    @jax.jit
    def solve(scale):
        return quad.integrate(
            lambda x, factor: factor * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            **_options(max_evaluations=point_count, max_regions=1),
        ).value

    values = jax.vmap(solve)(jnp.array([1.0, 2.0, 3.0]))
    derivative = jax.grad(solve)(jnp.asarray(2.0))
    _, tangent = jax.jvp(
        solve,
        (jnp.asarray(2.0),),
        (jnp.asarray(1.0),),
    )

    assert jnp.allclose(values, jnp.array([1.0, 2.0, 3.0]))
    assert jnp.array_equal(derivative, 0.0)
    assert jnp.array_equal(tangent, 0.0)


@pytest.mark.parametrize("gradient", ["replay", "through", "invalid"])
def test_adaptive_cubature_rejects_every_gradient_mode_except_stop(gradient):
    with pytest.raises(
        ValueError,
        match='AdaptiveCubature requires gradient="stop"',
    ):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.AdaptiveCubature(),
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=100,
            max_regions=2,
            gradient=gradient,
        )


@pytest.mark.parametrize("value", [jnp.array([1.0]), 1.0 + 0.0j])
def test_cubature_tolerances_must_be_real_scalars(value):
    expected = "scalar" if jnp.asarray(value).ndim else "real dtype"
    exception = ValueError if expected == "scalar" else TypeError
    with pytest.raises(exception, match=expected):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            **_options(epsabs=value),
        )
