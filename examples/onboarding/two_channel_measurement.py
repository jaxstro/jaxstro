"""Inspect covariance and weak directions in the Foundations running case.

Run from the repository root, for example:
``uv run --no-sync python examples/onboarding/two_channel_measurement.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax.numpy as jnp  # noqa: E402

from jaxstro.numerics.linear_algebra import condition_number  # noqa: E402


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
    _, singular_values, directions = jnp.linalg.svd(jacobian)
    correlation = covariance[0, 1] / jnp.sqrt(covariance[0, 0] * covariance[1, 1])
    return {
        "covariance": covariance,
        "jacobian": jacobian,
        "singular_values": singular_values,
        "parameter_directions": directions,
        "condition_number": condition_number(jacobian),
        "correlation": correlation,
    }


def _combination(direction: jnp.ndarray) -> str:
    a, b = (float(v) for v in direction)
    return f"({a:+.2f} theta_1 {b:+.2f} theta_2)"


def warranted_claim(case: Mapping[str, jnp.ndarray]) -> str:
    """State what this configuration's local geometry supports, from its numbers."""
    singular_values = case["singular_values"]
    strong = _combination(case["parameter_directions"][0])
    weak = _combination(case["parameter_directions"][-1])
    # Numerical rank as in numpy.linalg.matrix_rank: sigma_min <= sigma_max * n * eps.
    rank_floor = (
        float(singular_values[0]) * 2 * float(jnp.finfo(singular_values.dtype).eps)
    )
    if float(singular_values[-1]) <= rank_floor:
        geometry = f"This local map does not constrain the combination {weak} at all."
    else:
        ratio = float(case["condition_number"])
        geometry = (
            f"This local map constrains {strong} about {ratio:.3g} times more "
            f"tightly than {weak}."
        )
    rho = float(case["correlation"])
    if rho > 0.0:
        noise = (
            f"The shared calibration makes the channels correlated (rho = {rho:.3g}); "
            "two independent error bars would omit that correlation."
        )
    else:
        noise = "Under this covariance the channels are uncorrelated."
    return f"{geometry} {noise}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-sigma", type=float, default=0.2)
    parser.add_argument("--separation", type=float, default=0.01)
    return parser


def main() -> None:
    args = _parser().parse_args()
    case = two_channel_measurement(args.calibration_sigma, args.separation)

    print("Two-channel measurement")
    print(f"shared calibration sigma = {args.calibration_sigma:g}")
    print(f"parameter-direction separation = {args.separation:g}")
    print("covariance =")
    print(case["covariance"])
    print(f"correlation = {float(case['correlation']):.6g}")
    print("Jacobian =")
    print(case["jacobian"])
    print(f"singular values = {case['singular_values']}")
    print(f"local condition number = {float(case['condition_number']):.6g}")
    print(f"Warranted claim: {warranted_claim(case)}")


if __name__ == "__main__":
    main()
