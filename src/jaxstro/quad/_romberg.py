"""Fixed-capacity global Romberg refinement engines."""

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._tanh_sinh import _host_lattice
from .result import QuadStatus
from .tolerance import ErrorNorm
from .tolerance import error_norm as reduce_error_norm


class GlobalRefinementResult(NamedTuple):
    value: Array
    error: Array
    tolerance: Array
    status: Array
    evaluations: Array
    refinements: Array
    levels: Array


def _validate_global_inputs(initial_level, max_evaluations, max_regions) -> None:
    for name, value in (
        ("initial_level", initial_level),
        ("max_evaluations", max_evaluations),
        ("max_regions", max_regions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"global {name} must be a positive integer")


def validate_global_capacities(
    *,
    initial_level: int,
    max_evaluations: int,
    max_regions: int,
    tanh_sinh: bool,
    dtype,
) -> None:
    """Reject impossible global workspaces before tracing user code."""
    _validate_global_inputs(initial_level, max_evaluations, max_regions)
    if tanh_sinh:
        initial_cost = len(
            _host_lattice(initial_level, np.dtype(dtype).name).compact_nodes
        )
        message = "initial tanh-sinh level"
    else:
        initial_cost = 2**initial_level + 1
        message = "initial Romberg grid"
    if max_evaluations < initial_cost:
        raise ValueError(f"max_evaluations is smaller than the {message}")


def _tolerance_valid(epsabs, epsrel) -> Array:
    absolute = jnp.asarray(epsabs)
    relative = jnp.asarray(epsrel)
    return (
        jnp.isfinite(absolute)
        & jnp.isfinite(relative)
        & (absolute >= 0.0)
        & (relative >= 0.0)
    )


def _tolerance(value, epsabs, epsrel, norm: ErrorNorm) -> Array:
    value_norm = reduce_error_norm(value, norm)
    dtype = jnp.result_type(value_norm, epsabs, epsrel, 0.0)
    return jnp.maximum(
        jnp.asarray(epsabs, dtype=dtype),
        jnp.asarray(epsrel, dtype=dtype) * value_norm,
    )


def _gamma(count, dtype) -> Array:
    scaled = jnp.asarray(count, dtype=dtype) * jnp.finfo(dtype).eps
    return scaled / (1.0 - scaled)


def _richardson_error(value, previous, value_floor, previous_floor) -> Array:
    """Combine successive-diagonal change with both propagated floors."""
    return jnp.abs(value - previous) + value_floor + previous_floor


def _masked_evaluate(
    evaluate_one: Callable[[Array], tuple[Array, Array, Array]],
    nodes: Array,
    active: Array,
    zero: Array,
) -> tuple[Array, Array, Array]:
    def body(inputs):
        node, lane_active = inputs
        return jax.lax.cond(
            lane_active,
            evaluate_one,
            lambda _node: (zero, jnp.asarray(False), jnp.asarray(False)),
            node,
        )

    return jax.lax.map(body, (nodes, active))


def _richardson_row(table, floors, level, base, base_floor, max_level):
    table = table.at[level, 0].set(base)
    floors = floors.at[level, 0].set(base_floor)

    def column(column, state):
        current_table, current_floors = state

        def update(operand):
            update_table, update_floors = operand
            q = jnp.asarray(4.0**column, dtype=jnp.real(base).dtype)
            denominator = q - 1.0
            left = update_table[level, column - 1]
            above = update_table[level - 1, column - 1]
            value = (q * left - above) / denominator
            propagated = (
                q * update_floors[level, column - 1]
                + update_floors[level - 1, column - 1]
            ) / denominator
            arithmetic = (
                _gamma(3, jnp.real(base).dtype)
                * (q * jnp.abs(left) + jnp.abs(above))
                / denominator
            )
            return (
                update_table.at[level, column].set(value),
                update_floors.at[level, column].set(propagated + arithmetic),
            )

        return jax.lax.cond(column <= level, update, lambda operand: operand, state)

    return jax.lax.fori_loop(1, max_level + 1, column, (table, floors))


def romberg_refine(
    evaluate_one: Callable[[Array], tuple[Array, Array, Array]],
    zero: Array,
    *,
    initial_level: int,
    max_evaluations: int,
    max_regions: int,
    epsabs,
    epsrel,
    error_norm: ErrorNorm,
    dtype,
    input_valid=True,
) -> GlobalRefinementResult:
    """Run classical trapezoid/Richardson refinement with exact node reuse."""
    validate_global_capacities(
        initial_level=initial_level,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
        tanh_sinh=False,
        dtype=dtype,
    )
    max_level = (max_evaluations - 1).bit_length() - 1
    lane_count = 2 ** max(max_level - 1, 0)
    payload_shape = zero.shape
    value_dtype = zero.dtype
    real_dtype = jnp.real(zero).dtype
    table = jnp.zeros((max_level + 1, max_level + 1) + payload_shape, dtype=value_dtype)
    floors = jnp.zeros((max_level + 1, max_level + 1) + payload_shape, dtype=real_dtype)

    fine_count = 2**initial_level + 1
    lane = jnp.arange(max(lane_count, fine_count), dtype=jnp.int32)
    initial_active = lane < fine_count
    initial_nodes = -1.0 + 2.0 * lane.astype(dtype) / (fine_count - 1)
    initial_values, initial_nonfinite, initial_roundoff = _masked_evaluate(
        evaluate_one, initial_nodes, initial_active, zero
    )

    def initialize(level, state):
        current_table, current_floors = state
        stride = 2 ** (initial_level - level)
        selected = initial_active & (lane % stride == 0)
        endpoint = (lane == 0) | (lane == fine_count - 1)
        coefficients = jnp.where(selected, jnp.where(endpoint, 0.5, 1.0), 0.0)
        shape = coefficients.shape + (1,) * len(payload_shape)
        step = jnp.asarray(2.0 / 2**level, dtype=dtype)
        base = step * jnp.sum(initial_values * jnp.reshape(coefficients, shape), axis=0)
        resabs = step * jnp.sum(
            jnp.abs(initial_values) * jnp.reshape(coefficients, shape), axis=0
        )
        base_floor = _gamma(2**level + 1, real_dtype) * resabs
        return _richardson_row(
            current_table,
            current_floors,
            level,
            base,
            base_floor,
            max_level,
        )

    table, floors = jax.lax.fori_loop(0, initial_level + 1, initialize, (table, floors))
    value = table[initial_level, initial_level]
    previous = table[initial_level - 1, initial_level - 1]
    error = _richardson_error(
        value,
        previous,
        floors[initial_level, initial_level],
        floors[initial_level - 1, initial_level - 1],
    )
    tolerance = _tolerance(value, epsabs, epsrel, error_norm)
    invalid_input = ~(jnp.asarray(input_valid) & _tolerance_valid(epsabs, epsrel))
    nonfinite = (
        jnp.any(initial_nonfinite)
        | ~jnp.all(jnp.isfinite(value))
        | ~jnp.all(jnp.isfinite(error))
    )
    converged = reduce_error_norm(error, error_norm) <= tolerance
    running = jnp.asarray(-1, dtype=jnp.int32)
    status = jnp.where(
        invalid_input,
        jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
        jnp.where(
            nonfinite,
            jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
            jnp.where(
                converged,
                jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                jnp.where(
                    jnp.any(initial_roundoff),
                    jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                    jnp.where(
                        initial_level == max_level,
                        jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
                        running,
                    ),
                ),
            ),
        ),
    )

    class State(NamedTuple):
        table: Array
        floors: Array
        value: Array
        error: Array
        tolerance: Array
        status: Array
        level: Array

    state = State(
        table,
        floors,
        value,
        error,
        tolerance,
        status,
        jnp.asarray(initial_level, dtype=jnp.int32),
    )

    def condition(current):
        return current.status == running

    def body(current):
        level = current.level + 1
        new_count = 2 ** (level - 1)
        new_active = jnp.arange(lane_count) < new_count
        odd = 2 * jnp.arange(lane_count) + 1
        nodes = -1.0 + 2.0 * odd.astype(dtype) / (2**level)
        values, lane_nonfinite, lane_roundoff = _masked_evaluate(
            evaluate_one, nodes, new_active, zero
        )
        shape = new_active.shape + (1,) * len(payload_shape)
        selected_values = jnp.where(jnp.reshape(new_active, shape), values, 0.0)
        step = jnp.asarray(2.0, dtype=dtype) / (2**level)
        base = 0.5 * current.table[level - 1, 0] + step * jnp.sum(
            selected_values, axis=0
        )
        resabs = 0.5 * (
            current.floors[level - 1, 0] / _gamma(2 ** (level - 1) + 1, real_dtype)
        ) + step * jnp.sum(jnp.abs(selected_values), axis=0)
        base_floor = _gamma(2**level + 1, real_dtype) * resabs
        new_table, new_floors = _richardson_row(
            current.table,
            current.floors,
            level,
            base,
            base_floor,
            max_level,
        )
        new_value = new_table[level, level]
        new_error = _richardson_error(
            new_value,
            current.value,
            new_floors[level, level],
            current.floors[level - 1, level - 1],
        )
        new_tolerance = _tolerance(new_value, epsabs, epsrel, error_norm)
        new_nonfinite = (
            jnp.any(lane_nonfinite)
            | ~jnp.all(jnp.isfinite(new_value))
            | ~jnp.all(jnp.isfinite(new_error))
        )
        new_converged = reduce_error_norm(new_error, error_norm) <= new_tolerance
        new_status = jnp.where(
            new_nonfinite,
            jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
            jnp.where(
                new_converged,
                jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                jnp.where(
                    jnp.any(lane_roundoff),
                    jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                    jnp.where(
                        level == max_level,
                        jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
                        running,
                    ),
                ),
            ),
        )
        return State(
            new_table,
            new_floors,
            new_value,
            new_error,
            new_tolerance,
            new_status,
            level,
        )

    state = jax.lax.while_loop(condition, body, state)
    evaluations = 2**state.level + 1
    return GlobalRefinementResult(
        value=state.value,
        error=state.error,
        tolerance=state.tolerance,
        status=state.status,
        evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
        refinements=state.level,
        levels=state.level + 1,
    )


def romberg_replay_value(
    evaluate_one,
    zero,
    *,
    initial_level: int,
    accepted_level,
    max_evaluations: int,
    dtype,
):
    """Reconstruct one stopped classical Romberg diagonal."""
    max_level = (max_evaluations - 1).bit_length() - 1
    accepted_level = jax.lax.stop_gradient(jnp.asarray(accepted_level, dtype=jnp.int32))
    lane_count = 2 ** max(max_level - 1, 0)
    payload_shape = zero.shape
    value_dtype = zero.dtype
    real_dtype = jnp.real(zero).dtype
    table = jnp.zeros(
        (max_level + 1, max_level + 1) + payload_shape,
        dtype=value_dtype,
    )
    floors = jnp.zeros(
        (max_level + 1, max_level + 1) + payload_shape,
        dtype=real_dtype,
    )

    fine_count = 2**initial_level + 1
    lane = jnp.arange(max(lane_count, fine_count), dtype=jnp.int32)
    initial_active = lane < fine_count
    initial_nodes = -1.0 + 2.0 * lane.astype(dtype) / (fine_count - 1)
    initial_values, _, _ = _masked_evaluate(
        evaluate_one,
        initial_nodes,
        initial_active,
        zero,
    )

    def initialize(level, state):
        current_table, current_floors = state
        stride = 2 ** (initial_level - level)
        selected = initial_active & (lane % stride == 0)
        endpoint = (lane == 0) | (lane == fine_count - 1)
        coefficients = jnp.where(
            selected,
            jnp.where(endpoint, 0.5, 1.0),
            0.0,
        )
        shape = coefficients.shape + (1,) * len(payload_shape)
        step = jnp.asarray(2.0 / 2**level, dtype=dtype)
        weighted = initial_values * jnp.reshape(coefficients, shape)
        base = step * jnp.sum(weighted, axis=0)
        resabs = step * jnp.sum(jnp.abs(weighted), axis=0)
        base_floor = _gamma(2**level + 1, real_dtype) * resabs
        return _richardson_row(
            current_table,
            current_floors,
            level,
            base,
            base_floor,
            max_level,
        )

    table, floors = jax.lax.fori_loop(
        0,
        initial_level + 1,
        initialize,
        (table, floors),
    )

    def maybe_advance(level, state):
        def advance(current):
            current_table, current_floors = current
            new_count = 2 ** (level - 1)
            lane = jnp.arange(lane_count, dtype=jnp.int32)
            new_active = lane < new_count
            odd = 2 * lane + 1
            nodes = -1.0 + 2.0 * odd.astype(dtype) / (2**level)
            values, _, _ = _masked_evaluate(
                evaluate_one,
                nodes,
                new_active,
                zero,
            )
            shape = new_active.shape + (1,) * len(payload_shape)
            selected_values = jnp.where(
                jnp.reshape(new_active, shape),
                values,
                0.0,
            )
            step = jnp.asarray(2.0, dtype=dtype) / (2**level)
            base = 0.5 * current_table[level - 1, 0] + step * jnp.sum(
                selected_values,
                axis=0,
            )
            previous_resabs = current_floors[level - 1, 0] / _gamma(
                2 ** (level - 1) + 1,
                real_dtype,
            )
            resabs = 0.5 * previous_resabs + step * jnp.sum(
                jnp.abs(selected_values),
                axis=0,
            )
            base_floor = _gamma(2**level + 1, real_dtype) * resabs
            return _richardson_row(
                current_table,
                current_floors,
                level,
                base,
                base_floor,
                max_level,
            )

        return jax.lax.cond(
            level <= accepted_level,
            advance,
            lambda current: current,
            state,
        )

    table, _ = jax.lax.fori_loop(
        initial_level + 1,
        max_level + 1,
        maybe_advance,
        (table, floors),
    )
    return table[accepted_level, accepted_level]


@lru_cache(maxsize=None)
def _tanh_sinh_tables(initial_level: int, max_evaluations: int, dtype_name: str):
    max_level = initial_level
    while True:
        candidate = _host_lattice(max_level + 1, dtype_name)
        if len(candidate.compact_nodes) > max_evaluations:
            break
        max_level += 1
    finest = _host_lattice(max_level, dtype_name)
    max_indices = tuple(finest.compact_indices)
    position = {index: offset for offset, index in enumerate(max_indices)}
    rows = []
    densities = []
    active_rows = []
    terminal_rows = []
    counts = []
    exhausted = []
    for level in range(max_level + 1):
        host = _host_lattice(level, dtype_name)
        scale = 2 ** (max_level - level)
        row = np.zeros(len(max_indices), dtype=dtype_name)
        density = np.zeros(len(max_indices), dtype=dtype_name)
        active = np.zeros(len(max_indices), dtype=np.bool_)
        terminal = np.zeros(len(max_indices), dtype=np.bool_)
        for index, weight, density_weight in zip(
            host.compact_indices,
            host.compact_weights,
            host.compact_density_weights,
            strict=True,
        ):
            offset = position[index * scale]
            row[offset] = weight
            density[offset] = density_weight
            active[offset] = True
            terminal[offset] = abs(index) == host.terminal_index
        rows.append(row)
        densities.append(density)
        active_rows.append(active)
        terminal_rows.append(terminal)
        counts.append(len(host.compact_nodes))
        exhausted.append(host.dtype_exhausted)
    return (
        max_level,
        finest.compact_nodes,
        np.stack(rows),
        np.stack(densities),
        np.stack(active_rows),
        np.stack(terminal_rows),
        tuple(counts),
        tuple(exhausted),
    )


def romberg_tanh_sinh_refine(
    evaluate_one: Callable[[Array], tuple[Array, Array, Array]],
    zero: Array,
    *,
    initial_level: int,
    max_evaluations: int,
    max_regions: int,
    epsabs,
    epsrel,
    error_norm: ErrorNorm,
    dtype,
    input_valid=True,
) -> GlobalRefinementResult:
    """Run nested global tanh-sinh levels without Richardson extrapolation."""
    validate_global_capacities(
        initial_level=initial_level,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
        tanh_sinh=True,
        dtype=dtype,
    )
    dtype_name = np.dtype(dtype).name
    (
        max_level,
        nodes_host,
        weights_host,
        density_host,
        active_host,
        terminal_host,
        counts,
        exhausted_host,
    ) = _tanh_sinh_tables(initial_level, max_evaluations, dtype_name)
    nodes = jnp.asarray(nodes_host, dtype=dtype)
    weights = jnp.asarray(weights_host, dtype=dtype)
    densities = jnp.asarray(density_host, dtype=dtype)
    active = jnp.asarray(active_host)
    terminal = jnp.asarray(terminal_host)
    count_array = jnp.asarray(counts, dtype=jnp.int32)
    exhausted = jnp.asarray(exhausted_host)
    payload_shape = zero.shape
    real_dtype = jnp.real(zero).dtype
    value_shape = (nodes.shape[0],) + payload_shape
    cached = jnp.zeros(value_shape, dtype=zero.dtype)
    evaluated = jnp.zeros((nodes.shape[0],), dtype=jnp.bool_)

    def add_level(cached_values, evaluated_mask, level):
        new_mask = active[level] & ~evaluated_mask
        values, nonfinite, roundoff = _masked_evaluate(
            evaluate_one, nodes, new_mask, zero
        )
        shape = new_mask.shape + (1,) * len(payload_shape)
        updated = jnp.where(jnp.reshape(new_mask, shape), values, cached_values)
        return updated, evaluated_mask | new_mask, nonfinite, roundoff

    cached, evaluated, nonfinite_lanes, roundoff_lanes = add_level(
        cached, evaluated, initial_level
    )

    def estimate(level, cached_values):
        shape = (nodes.shape[0],) + (1,) * len(payload_shape)
        level_weights = jnp.reshape(weights[level], shape)
        previous_weights = jnp.reshape(weights[level - 1], shape)
        value = jnp.sum(cached_values * level_weights, axis=0)
        previous = jnp.sum(cached_values * previous_weights, axis=0)
        resabs = jnp.sum(jnp.abs(cached_values) * level_weights, axis=0)
        previous_resabs = jnp.sum(jnp.abs(cached_values) * previous_weights, axis=0)
        summation = (
            _gamma(count_array[level], real_dtype) * resabs
            + _gamma(count_array[level - 1], real_dtype) * previous_resabs
        )
        terminal_shape = terminal[level].shape + (1,) * len(payload_shape)
        terminal_values = cached_values * jnp.reshape(densities[level], shape)
        tail = jnp.sum(
            jnp.where(
                jnp.reshape(terminal[level], terminal_shape),
                jnp.abs(terminal_values),
                0.0,
            ),
            axis=0,
        )
        core_error = jnp.abs(value - previous) + summation
        return value, core_error + tail, core_error, tail

    value, error, core_error, tail_error = estimate(initial_level, cached)
    tolerance = _tolerance(value, epsabs, epsrel, error_norm)
    nonfinite = (
        jnp.any(nonfinite_lanes)
        | ~jnp.all(jnp.isfinite(value))
        | ~jnp.all(jnp.isfinite(error))
    )
    converged = reduce_error_norm(error, error_norm) <= tolerance
    exhausted_roundoff = (
        exhausted[initial_level]
        & (reduce_error_norm(core_error, error_norm) <= tolerance)
        & (reduce_error_norm(tail_error, error_norm) > tolerance)
    )
    running = jnp.asarray(-1, dtype=jnp.int32)
    status = jnp.where(
        ~(jnp.asarray(input_valid) & _tolerance_valid(epsabs, epsrel)),
        jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
        jnp.where(
            nonfinite,
            jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
            jnp.where(
                converged,
                jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                jnp.where(
                    jnp.any(roundoff_lanes) | exhausted_roundoff,
                    jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                    jnp.where(
                        initial_level == max_level,
                        jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
                        running,
                    ),
                ),
            ),
        ),
    )

    class State(NamedTuple):
        cached: Array
        evaluated: Array
        value: Array
        error: Array
        tolerance: Array
        status: Array
        level: Array

    state = State(
        cached,
        evaluated,
        value,
        error,
        tolerance,
        status,
        jnp.asarray(initial_level, dtype=jnp.int32),
    )

    def condition(current):
        return current.status == running

    def body(current):
        level = current.level + 1
        new_cached, new_evaluated, lane_nonfinite, lane_roundoff = add_level(
            current.cached, current.evaluated, level
        )
        new_value, new_error, new_core_error, new_tail_error = estimate(
            level, new_cached
        )
        new_tolerance = _tolerance(new_value, epsabs, epsrel, error_norm)
        new_nonfinite = (
            jnp.any(lane_nonfinite)
            | ~jnp.all(jnp.isfinite(new_value))
            | ~jnp.all(jnp.isfinite(new_error))
        )
        new_converged = reduce_error_norm(new_error, error_norm) <= new_tolerance
        exhausted_roundoff = (
            exhausted[level]
            & (reduce_error_norm(new_core_error, error_norm) <= new_tolerance)
            & (reduce_error_norm(new_tail_error, error_norm) > new_tolerance)
        )
        new_status = jnp.where(
            new_nonfinite,
            jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
            jnp.where(
                new_converged,
                jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                jnp.where(
                    jnp.any(lane_roundoff),
                    jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                    jnp.where(
                        exhausted_roundoff,
                        jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                        jnp.where(
                            level == max_level,
                            jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
                            running,
                        ),
                    ),
                ),
            ),
        )
        return State(
            new_cached,
            new_evaluated,
            new_value,
            new_error,
            new_tolerance,
            new_status,
            level,
        )

    state = jax.lax.while_loop(condition, body, state)
    return GlobalRefinementResult(
        value=state.value,
        error=state.error,
        tolerance=state.tolerance,
        status=state.status,
        evaluations=count_array[state.level],
        refinements=state.level,
        levels=state.level + 1,
    )


def romberg_tanh_sinh_replay_value(
    evaluate_one,
    zero,
    *,
    initial_level: int,
    accepted_level,
    max_evaluations: int,
    dtype,
):
    """Reconstruct one stopped global tanh-sinh weight row."""
    dtype_name = np.dtype(dtype).name
    _, nodes_host, weights_host, _, active_host, _, _, _ = _tanh_sinh_tables(
        initial_level,
        max_evaluations,
        dtype_name,
    )
    nodes = jnp.asarray(nodes_host, dtype=dtype)
    weights = jnp.asarray(weights_host, dtype=dtype)
    active = jnp.asarray(active_host)
    level = jax.lax.stop_gradient(jnp.asarray(accepted_level, dtype=jnp.int32))
    values, _, _ = _masked_evaluate(evaluate_one, nodes, active[level], zero)
    shape = (nodes.shape[0],) + (1,) * zero.ndim
    return jnp.sum(values * jnp.reshape(weights[level], shape), axis=0)


__all__ = [
    "GlobalRefinementResult",
    "romberg_refine",
    "romberg_replay_value",
    "romberg_tanh_sinh_refine",
    "romberg_tanh_sinh_replay_value",
    "validate_global_capacities",
]
