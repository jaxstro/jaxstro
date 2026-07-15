"""Static declarations for one-dimensional adaptive quadrature methods."""

from dataclasses import dataclass

import jax


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_level(name: str, value: int, *, minimum: int) -> None:
    if not _is_integer(value) or value < minimum:
        requirement = "positive integer" if minimum == 1 else "nonnegative integer"
        raise ValueError(f"{name} initial_level must be a {requirement}")


class _StaticMethod:
    _field_name: str

    def tree_flatten(self):
        return (), getattr(self, self._field_name)

    @classmethod
    def tree_unflatten(cls, value, _children):
        return cls(**{cls._field_name: value})


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaussKronrod(_StaticMethod):
    """Embedded Gauss-Kronrod pair selected by its Kronrod node count."""

    pair: int = 21
    _field_name = "pair"

    def __post_init__(self) -> None:
        if not _is_integer(self.pair) or self.pair not in {15, 21, 31, 41, 51, 61}:
            raise ValueError("GaussKronrod pair must be 15, 21, 31, 41, 51, or 61")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveClenshawCurtis(_StaticMethod):
    """Nested regional Clenshaw-Curtis refinement."""

    initial_order: int = 17
    _field_name = "initial_order"

    def __post_init__(self) -> None:
        intervals = self.initial_order - 1 if _is_integer(self.initial_order) else 0
        if (
            not _is_integer(self.initial_order)
            or self.initial_order < 5
            or intervals & (intervals - 1) != 0
        ):
            raise ValueError(
                "AdaptiveClenshawCurtis initial_order must have form 2^k + 1 "
                "with k >= 2"
            )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveTanhSinh(_StaticMethod):
    """Regional double-exponential refinement with explicit tail evidence."""

    initial_level: int = 3
    _field_name = "initial_level"

    def __post_init__(self) -> None:
        _require_level(type(self).__name__, self.initial_level, minimum=0)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Romberg(_StaticMethod):
    """Global trapezoid refinement with Richardson extrapolation."""

    initial_level: int = 1
    _field_name = "initial_level"

    def __post_init__(self) -> None:
        _require_level(type(self).__name__, self.initial_level, minimum=1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RombergTanhSinh(_StaticMethod):
    """Global tanh-sinh level refinement without Richardson extrapolation."""

    initial_level: int = 1
    _field_name = "initial_level"

    def __post_init__(self) -> None:
        _require_level(type(self).__name__, self.initial_level, minimum=1)


__all__ = [
    "AdaptiveClenshawCurtis",
    "AdaptiveTanhSinh",
    "GaussKronrod",
    "Romberg",
    "RombergTanhSinh",
]
