"""Freshness comparison for committed JSON evidence artifacts.

A committed artifact is fresh when a new run of its generator reproduces it.
On the platform that generated the artifact (its origin) floats must match
bit-for-bit. On any other platform floats must satisfy
``|x - y| <= OFF_ORIGIN_RTOL * max(|x|, |y|) + OFF_ORIGIN_ATOL``; XLA on
Linux x86_64 and macOS arm64 differs in the last bits. Keys, strings,
integers, booleans, and list lengths must match exactly everywhere.

The tolerance was approved on 2026-09-23 from the full-gate run 35584140364
(2026-09-21), Linux x86_64 against the macOS arm64 multidimensional-quadrature
artifact: computed values differed by at most 3.8e-15 relative, and error
fields (differences from truth, near zero) by at most 2.8e-15 absolute. The
bound is 26x and 3.6x those. Off the origin, a field near 1e-14 is therefore
checked only to order unity.

The absolute floor does not scale with the magnitude of the quantities an
error field is computed from: in full-gate run 35950495028 (2026-09-24) the
absolute error of a value near 525.66 was 0.0 on macOS and 3.41e-13 on
Linux. Off the origin, fields named in ``derived`` (errors computed as
differences of other recorded fields, and finite-difference estimates, whose
last-bit noise is amplified by the inverse step) are therefore not compared;
the fields they are checked against still are, and pass/fail flags stay
exact. Approved 2026-09-24.
"""

from __future__ import annotations

import math
import platform
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

OFF_ORIGIN_RTOL = 1.0e-13
OFF_ORIGIN_ATOL = 1.0e-14


@dataclass(frozen=True)
class Origin:
    """Platform identity that decides bit-exact versus tolerant comparison."""

    system: str
    machine: str

    @classmethod
    def of_runner(cls) -> Origin:
        return cls(system=platform.platform().split("-")[0], machine=platform.machine())

    @classmethod
    def from_environment(cls, environment: dict[str, Any]) -> Origin:
        """Read the origin from an artifact's recorded ``platform``/``machine``."""
        return cls(
            system=environment["platform"].split("-")[0],
            machine=environment["machine"],
        )


def assert_fresh(
    recorded: Any,
    fresh: Any,
    *,
    origin: Origin,
    ignore: Iterable[str] = (),
    derived: Iterable[str] = (),
) -> None:
    """Assert that ``fresh`` reproduces ``recorded``; name the first differing path.

    ``ignore`` lists top-level keys that record provenance (for example the
    generating environment) rather than content. ``derived`` lists keys, at
    any depth, whose values are errors computed from other recorded fields;
    they are compared on the origin and skipped elsewhere.
    """
    ignored = frozenset(ignore)
    if ignored:
        recorded = {k: v for k, v in recorded.items() if k not in ignored}
        fresh = {k: v for k, v in fresh.items() if k not in ignored}
    exact = origin == Origin.of_runner()
    skip = frozenset() if exact else frozenset(derived)
    _compare(recorded, fresh, "$", exact=exact, skip=skip)


def _compare(
    recorded: Any, fresh: Any, path: str, *, exact: bool, skip: frozenset[str]
) -> None:
    if type(recorded) is not type(fresh):
        raise AssertionError(
            f"{path}: type {type(recorded).__name__} != {type(fresh).__name__}"
        )
    if isinstance(recorded, dict):
        if recorded.keys() != fresh.keys():
            raise AssertionError(
                f"{path}: keys differ: {sorted(recorded.keys() ^ fresh.keys())}"
            )
        for key in recorded:
            if key in skip:
                continue
            _compare(recorded[key], fresh[key], f"{path}.{key}", exact=exact, skip=skip)
    elif isinstance(recorded, list):
        if len(recorded) != len(fresh):
            raise AssertionError(f"{path}: length {len(recorded)} != {len(fresh)}")
        for index, (left, right) in enumerate(zip(recorded, fresh, strict=True)):
            _compare(left, right, f"{path}[{index}]", exact=exact, skip=skip)
    elif isinstance(recorded, float):
        if recorded == fresh or (math.isnan(recorded) and math.isnan(fresh)):
            return
        bound = OFF_ORIGIN_RTOL * max(abs(recorded), abs(fresh)) + OFF_ORIGIN_ATOL
        if exact or not abs(recorded - fresh) <= bound:
            raise AssertionError(
                f"{path}: {recorded!r} != {fresh!r} "
                f"(|diff| {abs(recorded - fresh):.3g}, off-origin bound {bound:.3g})"
            )
    elif recorded != fresh:
        raise AssertionError(f"{path}: {recorded!r} != {fresh!r}")
