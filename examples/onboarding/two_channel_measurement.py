"""Inspect covariance and weak directions in the Foundations running case.

Run from the repository root, for example:
``uv run --no-sync python examples/onboarding/two_channel_measurement.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import jax.numpy as jnp


def two_channel_measurement(
    calibration_sigma: float = 0.2,
    separation: float = 0.01,
) -> Mapping[str, jnp.ndarray]:
    """Return the covariance and local sensitivity geometry of two channels.

    The calibration term moves both channels together. ``separation`` controls
    how distinctly the second parameter moves the second channel; zero makes
    the two parameter directions exactly degenerate in this local model.
    """
    if calibration_sigma < 0.0:
        raise ValueError("calibration_sigma must be nonnegative")
    if separation < 0.0:
        raise ValueError("separation must be nonnegative")

    shared_variance = calibration_sigma**2
    covariance = jnp.eye(2) + shared_variance * jnp.ones((2, 2))
    jacobian = jnp.array([[1.0, 1.0], [1.0, 1.0 + separation]])
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    correlation = covariance[0, 1] / jnp.sqrt(covariance[0, 0] * covariance[1, 1])
    return {
        "covariance": covariance,
        "jacobian": jacobian,
        "singular_values": singular_values,
        "correlation": correlation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-sigma", type=float, default=0.2)
    parser.add_argument("--separation", type=float, default=0.01)
    return parser


def main() -> None:
    args = _parser().parse_args()
    case = two_channel_measurement(args.calibration_sigma, args.separation)
    smallest = case["singular_values"][-1]
    largest = case["singular_values"][0]
    condition = jnp.where(smallest > 0.0, largest / smallest, jnp.inf)

    print("Two-channel measurement")
    print(f"shared calibration sigma = {args.calibration_sigma:g}")
    print(f"parameter-direction separation = {args.separation:g}")
    print("covariance =")
    print(case["covariance"])
    print(f"correlation = {float(case['correlation']):.6g}")
    print("Jacobian =")
    print(case["jacobian"])
    print(f"singular values = {case['singular_values']}")
    print(f"local condition number = {float(condition):.6g}")
    print(
        "Interpretation: covariance records the shared calibration assumption; "
        "the smaller singular value records how weakly this local measurement "
        "map separates one parameter combination."
    )


if __name__ == "__main__":
    main()
