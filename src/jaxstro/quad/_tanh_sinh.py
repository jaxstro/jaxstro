"""Nested representability-aware tanh-sinh formulas."""

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .rules import FixedRuleData, TanhSinhRule


class TanhSinhLatticeData(NamedTuple):
    """Padded adaptive lattice plus its compact fixed-rule representation."""

    candidate_indices: Array
    parameters: Array
    nodes: Array
    density_weights: Array
    weights: Array
    active: Array
    compact_indices: Array
    compact_nodes: Array
    compact_density_weights: Array
    compact_weights: Array
    coarse_to_fine: Array
    terminal_index: Array
    terminal_parameter: Array
    terminal_node: Array
    terminal_density_weight: Array
    dtype_exhausted: Array


@dataclass(frozen=True)
class _HostLattice:
    candidate_indices: tuple[int, ...]
    parameters: tuple[float, ...]
    nodes: tuple[float, ...]
    density_weights: tuple[float, ...]
    weights: tuple[float, ...]
    active: tuple[bool, ...]
    compact_indices: tuple[int, ...]
    compact_nodes: tuple[float, ...]
    compact_density_weights: tuple[float, ...]
    compact_weights: tuple[float, ...]
    coarse_to_fine: tuple[int, ...]
    terminal_index: int
    terminal_parameter: float
    terminal_node: float
    terminal_density_weight: float
    dtype_exhausted: bool


def _positive_candidates(level: int, dtype: np.dtype, cap: int):
    scalar = dtype.type
    step = scalar(2.0**-level)
    index = np.arange(cap + 1, dtype=np.int64)
    parameter = np.asarray(index, dtype=dtype) * step
    pi = scalar(np.pi)
    transformed = scalar(0.5) * pi * np.sinh(parameter)
    nodes = np.asarray(np.tanh(transformed), dtype=dtype)
    density_weights = np.asarray(
        scalar(0.5) * pi * np.cosh(parameter) / np.cosh(transformed) ** scalar(2),
        dtype=dtype,
    )
    return step, parameter, nodes, density_weights


def _raw_candidate_cap(level: int, dtype: np.dtype) -> int:
    scalar = dtype.type
    endpoint = np.nextafter(scalar(1.0), scalar(0.0), dtype=dtype)
    pi = scalar(np.pi)
    parameter_cap = np.arcsinh((scalar(2.0) / pi) * np.arctanh(endpoint))
    return int(np.ceil(parameter_cap / scalar(2.0**-level))) + 1


def _positive_candidate(index: int, level: int, dtype: np.dtype):
    """Evaluate one host candidate without constructing a candidate array."""
    scalar = dtype.type
    step = scalar(2.0**-level)
    parameter = scalar(index) * step
    pi = scalar(np.pi)
    transformed = scalar(0.5) * pi * np.sinh(parameter)
    node = scalar(np.tanh(transformed))
    density = scalar(
        scalar(0.5) * pi * np.cosh(parameter) / np.cosh(transformed) ** scalar(2)
    )
    return node, density


@lru_cache(maxsize=None)
def _retained_positive_indices(level: int, dtype_name: str) -> tuple[int, ...]:
    """Return representable positive indices without materializing rule arrays."""
    dtype = np.dtype(dtype_name)
    scalar = dtype.type
    previous = _retained_positive_indices(level - 1, dtype_name) if level > 0 else None
    raw_cap = _raw_candidate_cap(level, dtype)
    cap = raw_cap if previous is None else max(raw_cap, 2 * previous[-1])

    if previous is None:
        retained = [0]
        reserved = {0}
    else:
        reserved = {2 * index for index in previous}
        retained = sorted(reserved)
    retained_nodes = {
        index: _positive_candidate(index, level, dtype)[0] for index in retained
    }

    for index in range(1, cap + 1):
        if index in reserved:
            continue
        node, density = _positive_candidate(index, level, dtype)
        valid = (
            np.isfinite(node)
            and node >= scalar(0.0)
            and node < scalar(1.0)
            and np.isfinite(density)
            and density > scalar(0.0)
        )
        if not bool(valid):
            continue
        position = bisect_left(retained, index)
        if position > 0 and not bool(node > retained_nodes[retained[position - 1]]):
            continue
        if position < len(retained) and not bool(
            node < retained_nodes[retained[position]]
        ):
            continue
        retained.insert(position, index)
        retained_nodes[index] = node
    return tuple(retained)


@lru_cache(maxsize=None)
def _host_lattice(level: int, dtype_name: str) -> _HostLattice:
    dtype = np.dtype(dtype_name)
    scalar = dtype.type
    previous = _host_lattice(level - 1, dtype_name) if level > 0 else None
    raw_cap = _raw_candidate_cap(level, dtype)
    cap = raw_cap if previous is None else max(raw_cap, 2 * previous.terminal_index)
    step, positive_parameters, positive_nodes, positive_density = _positive_candidates(
        level, dtype, cap
    )
    endpoint = np.nextafter(scalar(1.0), scalar(0.0), dtype=dtype)

    if previous is None:
        retained = [0]
        reserved = {0}
    else:
        previous_positive = [index for index in previous.compact_indices if index >= 0]
        reserved = {2 * index for index in previous_positive}
        retained = sorted(reserved)

    valid = (
        np.isfinite(positive_nodes)
        & (positive_nodes >= scalar(0.0))
        & (positive_nodes < scalar(1.0))
        & np.isfinite(positive_density)
        & (positive_density > scalar(0.0))
    )
    for index in range(1, cap + 1):
        if index in reserved or not bool(valid[index]):
            continue
        position = bisect_left(retained, index)
        if position > 0:
            left = retained[position - 1]
            if not bool(positive_nodes[index] > positive_nodes[left]):
                continue
        if position < len(retained):
            right = retained[position]
            if not bool(positive_nodes[index] < positive_nodes[right]):
                continue
        retained.insert(position, index)

    positive_nonzero = retained[1:]
    compact_indices = tuple([-index for index in reversed(positive_nonzero)] + retained)
    compact_nodes = tuple(
        float(-positive_nodes[-index]) if index < 0 else float(positive_nodes[index])
        for index in compact_indices
    )
    compact_density = tuple(
        float(positive_density[abs(index)]) for index in compact_indices
    )
    compact_weights = tuple(float(step) * weight for weight in compact_density)
    compact_position = {
        index: position for position, index in enumerate(compact_indices)
    }
    if previous is None:
        coarse_to_fine: tuple[int, ...] = ()
    else:
        coarse_to_fine = tuple(
            compact_position[2 * index] for index in previous.compact_indices
        )

    retained_set = set(retained)
    inactive_sentinel = float(scalar(0.25))
    candidate_indices = tuple(range(-cap, cap + 1))
    active = tuple(abs(index) in retained_set for index in candidate_indices)
    parameters = tuple(float(step) * index for index in candidate_indices)
    nodes = tuple(
        (float(-positive_nodes[-index]) if index < 0 else float(positive_nodes[index]))
        if is_active
        else inactive_sentinel
        for index, is_active in zip(candidate_indices, active, strict=True)
    )
    density_weights = tuple(
        float(positive_density[abs(index)]) if is_active else 0.0
        for index, is_active in zip(candidate_indices, active, strict=True)
    )
    weights = tuple(float(step) * weight for weight in density_weights)
    terminal_index = retained[-1]

    return _HostLattice(
        candidate_indices=candidate_indices,
        parameters=parameters,
        nodes=nodes,
        density_weights=density_weights,
        weights=weights,
        active=active,
        compact_indices=compact_indices,
        compact_nodes=compact_nodes,
        compact_density_weights=compact_density,
        compact_weights=compact_weights,
        coarse_to_fine=coarse_to_fine,
        terminal_index=terminal_index,
        terminal_parameter=float(positive_parameters[terminal_index]),
        terminal_node=float(positive_nodes[terminal_index]),
        terminal_density_weight=float(positive_density[terminal_index]),
        dtype_exhausted=bool(positive_nodes[terminal_index] == endpoint),
    )


def _selected_dtype(dtype):
    selected_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
    if selected_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("tanh-sinh lattice dtype must be float32 or float64")
    return selected_dtype


def _open_unit_interval_trim(host: _HostLattice, dtype: np.dtype) -> int:
    """Count symmetric outer pairs lost under the affine unit-interval map."""
    scalar = dtype.type
    half = scalar(0.5)
    zero = scalar(0.0)
    one = scalar(1.0)
    mapped = tuple(half * (scalar(node) + one) for node in host.compact_nodes)
    pair_count = (len(mapped) - 1) // 2
    trim = 0
    while trim < pair_count and not (
        mapped[trim] > zero
        and mapped[-1 - trim] < one
        and mapped[trim] < mapped[trim + 1]
        and mapped[-2 - trim] < mapped[-1 - trim]
    ):
        trim += 1
    retained = mapped[trim : len(mapped) - trim if trim else None]
    if not all(zero < point < one for point in retained) or not all(
        left < right for left, right in pairwise(retained)
    ):
        raise RuntimeError(
            "tanh-sinh unit-interval nodes are not strictly representable"
        )
    return trim


def _tanh_sinh_lattice_data(level: int, *, dtype=None) -> TanhSinhLatticeData:
    """Return static lattice data for a level and real precision policy."""
    selected_dtype = _selected_dtype(dtype)
    host = _host_lattice(level, np.dtype(selected_dtype).name)
    return TanhSinhLatticeData(
        candidate_indices=jnp.asarray(host.candidate_indices, dtype=jnp.int32),
        parameters=jnp.asarray(host.parameters, dtype=selected_dtype),
        nodes=jnp.asarray(host.nodes, dtype=selected_dtype),
        density_weights=jnp.asarray(host.density_weights, dtype=selected_dtype),
        weights=jnp.asarray(host.weights, dtype=selected_dtype),
        active=jnp.asarray(host.active, dtype=jnp.bool_),
        compact_indices=jnp.asarray(host.compact_indices, dtype=jnp.int32),
        compact_nodes=jnp.asarray(host.compact_nodes, dtype=selected_dtype),
        compact_density_weights=jnp.asarray(
            host.compact_density_weights, dtype=selected_dtype
        ),
        compact_weights=jnp.asarray(host.compact_weights, dtype=selected_dtype),
        coarse_to_fine=jnp.asarray(host.coarse_to_fine, dtype=jnp.int32),
        terminal_index=jnp.asarray(host.terminal_index, dtype=jnp.int32),
        terminal_parameter=jnp.asarray(host.terminal_parameter, dtype=selected_dtype),
        terminal_node=jnp.asarray(host.terminal_node, dtype=selected_dtype),
        terminal_density_weight=jnp.asarray(
            host.terminal_density_weight, dtype=selected_dtype
        ),
        dtype_exhausted=jnp.asarray(host.dtype_exhausted),
    )


def tanh_sinh_rule_point_count(
    rule: TanhSinhRule,
    *,
    dtype=None,
    open_unit_interval: bool = False,
) -> int:
    """Return a compact rule count without constructing JAX rule arrays."""
    selected_dtype = _selected_dtype(dtype)
    host_dtype = np.dtype(selected_dtype)
    retained = _retained_positive_indices(rule.level, host_dtype.name)
    if not open_unit_interval:
        return 2 * len(retained) - 1

    scalar = host_dtype.type
    half = scalar(0.5)
    zero = scalar(0.0)
    one = scalar(1.0)
    point_count = 1
    for index in retained[1:]:
        node, _density = _positive_candidate(index, rule.level, host_dtype)
        left = half * (-node + one)
        right = half * (node + one)
        if zero < left < right < one:
            point_count += 2
    return point_count


def tanh_sinh_rule_data(
    rule: TanhSinhRule,
    *,
    dtype=None,
    open_unit_interval: bool = False,
) -> FixedRuleData:
    """Construct a compact nested double-exponential formula on ``(-1, 1)``."""
    selected_dtype = _selected_dtype(dtype)
    lattice = _tanh_sinh_lattice_data(rule.level, dtype=selected_dtype)
    if open_unit_interval:
        host = _host_lattice(rule.level, np.dtype(selected_dtype).name)
        trim = _open_unit_interval_trim(host, np.dtype(selected_dtype))
        retained = slice(trim, -trim if trim else None)
        nodes = lattice.compact_nodes[retained]
        weights = lattice.compact_weights[retained]
    else:
        nodes = lattice.compact_nodes
        weights = lattice.compact_weights
    return FixedRuleData(
        nodes=nodes,
        weights=weights,
        degree=-1,
        nested=True,
    )


__all__ = ["tanh_sinh_rule_data", "tanh_sinh_rule_point_count"]
