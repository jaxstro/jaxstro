"""Contract for the cross-platform evidence-artifact freshness comparison."""

import pytest

from tests.validation._freshness import (
    OFF_ORIGIN_ATOL,
    OFF_ORIGIN_RTOL,
    Origin,
    assert_fresh,
)

HERE = Origin.of_runner()
ELSEWHERE = Origin(system="Plan9", machine="pdp11")


def test_origin_is_bit_exact_for_floats() -> None:
    assert_fresh({"x": 1.0}, {"x": 1.0}, origin=HERE)
    with pytest.raises(AssertionError, match=r"\$\.x"):
        assert_fresh({"x": 1.0}, {"x": 1.0 + 2.0**-52}, origin=HERE)


def test_off_origin_floats_use_the_approved_tolerance() -> None:
    assert (OFF_ORIGIN_RTOL, OFF_ORIGIN_ATOL) == (1.0e-13, 1.0e-14)
    # Measured Linux x86_64 vs macOS arm64 differences (full-gate run
    # 35584140364, 2026-09-21): value rel 3.8e-15, error-field abs 2.8e-15.
    assert_fresh(
        {"v": 0.0024999999896942308}, {"v": 0.0024999999896942403}, origin=ELSEWHERE
    )
    assert_fresh({"e": 5.2e-16}, {"e": 3.3e-15}, origin=ELSEWHERE)
    with pytest.raises(AssertionError, match=r"\$\.v"):
        assert_fresh({"v": 1.0}, {"v": 1.0 + 1.0e-12}, origin=ELSEWHERE)
    with pytest.raises(AssertionError, match=r"\$\.e"):
        assert_fresh({"e": 0.0}, {"e": 2.0e-14}, origin=ELSEWHERE)


@pytest.mark.parametrize(
    ("recorded", "fresh", "path"),
    (
        ({"n": 21}, {"n": 17}, r"\$\.n"),
        ({"s": "gk21"}, {"s": "cc17"}, r"\$\.s"),
        ({"a": 1}, {"b": 1}, r"\$: keys"),
        ({"r": [1.0, 2.0]}, {"r": [1.0]}, r"\$\.r: length"),
        ({"r": [{"k": True}]}, {"r": [{"k": False}]}, r"\$\.r\[0\]\.k"),
        ({"x": 1}, {"x": 1.0}, r"\$\.x: type"),
    ),
)
def test_non_float_content_is_exact_everywhere(recorded, fresh, path) -> None:
    for origin in (HERE, ELSEWHERE):
        with pytest.raises(AssertionError, match=path):
            assert_fresh(recorded, fresh, origin=origin)


def test_ignored_keys_are_provenance_not_content() -> None:
    recorded = {"environment": {"platform": "macOS-26.1"}, "x": 1.0}
    fresh = {"environment": {"platform": "Linux-6.17"}, "x": 1.0}
    assert_fresh(recorded, fresh, origin=HERE, ignore=("environment",))
    with pytest.raises(AssertionError, match="environment"):
        assert_fresh(recorded, fresh, origin=HERE)


def test_origin_reads_recorded_environment() -> None:
    origin = Origin.from_environment(
        {"platform": "macOS-26.1-arm64-arm-64bit-Mach-O", "machine": "arm64"}
    )
    assert origin == Origin(system="macOS", machine="arm64")
    linux = Origin.from_environment(
        {
            "platform": "Linux-6.17.0-1022-azure-x86_64-with-glibc2.39",
            "machine": "x86_64",
        }
    )
    assert linux == Origin(system="Linux", machine="x86_64")
