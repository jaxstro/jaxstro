"""Evidence-bearing figures for safeguarded and implicit scalar roots."""

from __future__ import annotations

from typing import Any

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from jaxstro.numerics import (  # noqa: E402
    PROPOSAL_INVERSE_QUADRATIC,
    PROPOSAL_MIDPOINT,
    PROPOSAL_SECANT,
    ImplicitRootAssumptions,
    implicit_bracketed_root,
    safeguarded_bracketed_root,
)

from .style import (  # noqa: E402
    NEGATIVE,
    NEUTRAL,
    PALETTE,
    POSITIVE,
    polish_axes,
    setup_style,
)

PROPOSAL_MARKERS = {
    PROPOSAL_INVERSE_QUADRATIC: "o",
    PROPOSAL_SECANT: "s",
    PROPOSAL_MIDPOINT: "^",
}


def root_trace_results():
    """Return a public-API residual curve and safeguarded solver trace."""
    x = np.linspace(0.0, 2.0, 801, dtype=np.float64)
    residual = x * x - 2.0
    result = safeguarded_bracketed_root(
        lambda value: value * value - 2.0,
        0.0,
        2.0,
        max_steps=96,
        atol=1.0e-12,
        rtol=1.0e-12,
        safeguard_fraction=0.1,
    )
    return x, residual, result


def implicit_comparison_results() -> dict[str, Any]:
    """Return certified and rejected implicit-root evidence for one lesson."""
    assumptions = ImplicitRootAssumptions(True, True)

    def quadratic(x, theta):
        return x * x - theta

    def solve(theta):
        return implicit_bracketed_root(
            quadratic,
            theta,
            0.0,
            4.0,
            assumptions=assumptions,
            max_steps=96,
            atol=1.0e-14,
            rtol=1.0e-14,
            safeguard_fraction=0.1,
            derivative_residual_atol=1.0e-12,
            derivative_width_atol=1.0e-12,
            derivative_slope_floor=1.0e-8,
        )

    theta = jnp.asarray(2.0, dtype=jnp.float64)
    certified = solve(theta)
    ad = jax.grad(lambda parameter: solve(parameter).root)(theta)
    step = jnp.asarray(1.0e-5, dtype=theta.dtype)
    fd = (solve(theta + step).root - solve(theta - step).root) / (2.0 * step)
    rejected = implicit_bracketed_root(
        lambda x, parameter: (x - parameter) ** 3,
        theta,
        0.0,
        4.0,
        assumptions=assumptions,
        max_steps=32,
        atol=1.0e-14,
        rtol=1.0e-14,
        derivative_residual_atol=1.0e-12,
        derivative_width_atol=1.0e-12,
        derivative_slope_floor=1.0e-8,
    )
    return {
        "certified": certified,
        "rejected": rejected,
        "analytic": float(1.0 / (2.0 * jnp.sqrt(theta))),
        "ad": float(ad),
        "fd": float(fd),
    }


def build_rootfinding_safeguards() -> Figure:
    """Plot proposal telemetry and verified bracket contraction."""
    setup_style()
    x, residual, result = root_trace_results()
    executed = np.asarray(result.trace.executed)
    proposal = np.asarray(result.trace.proposal)[executed]
    proposal_residual = np.asarray(result.trace.residual)[executed]
    kinds = np.asarray(result.trace.proposal_kind)[executed]
    lo = np.asarray(result.trace.lo)[executed]
    hi = np.asarray(result.trace.hi)[executed]
    iterations = np.arange(proposal.size)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    ax = axes[0]
    ax.plot(x, residual, color=NEUTRAL, label=r"$G(x)=x^2-2$")
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.8)
    kind_styles = {
        PROPOSAL_INVERSE_QUADRATIC: (POSITIVE, "IQI"),
        PROPOSAL_SECANT: (PALETTE[0], "Secant"),
        PROPOSAL_MIDPOINT: (PALETTE[4], "Midpoint fallback"),
    }
    for kind, (color, label) in kind_styles.items():
        mask = kinds == kind
        if np.any(mask):
            ax.scatter(
                proposal[mask],
                proposal_residual[mask],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                marker=PROPOSAL_MARKERS[kind],
                label=label,
                zorder=3,
            )
    ax.axvline(np.sqrt(2.0), color=POSITIVE, linestyle="--", linewidth=1.0)
    ax.set(
        xlabel="Coordinate x",
        ylabel="Signed residual G(x)",
        title="Safeguarded proposals",
    )
    ax.legend(frameon=False, loc="upper left")
    polish_axes(ax, grid_axis="y")

    ax = axes[1]
    ax.plot(iterations, lo, color=PALETTE[0], linestyle="-", label="Verified lo")
    ax.plot(iterations, hi, color=PALETTE[1], linestyle="--", label="Verified hi")
    ax.plot(
        iterations,
        hi - lo,
        color=POSITIVE,
        linestyle=":",
        label="Full bracket width",
    )
    ax.set_yscale("log")
    ax.set(
        xlabel="Executed iteration",
        ylabel="Coordinate / width",
        title="Bracket evidence contracts",
    )
    ax.text(
        0.03,
        0.04,
        f"terminal status = {int(result.status)}",
        transform=ax.transAxes,
        color=NEUTRAL,
        fontsize=7.5,
    )
    ax.legend(frameon=False, loc="upper right")
    polish_axes(ax, grid_axis="y")
    fig.tight_layout()
    return fig


def build_rootfinding_value_versus_ift() -> Figure:
    """Contrast branch-selected value evidence with certified IFT evidence."""
    setup_style()
    comparison = implicit_comparison_results()
    certified = comparison["certified"]
    rejected = comparison["rejected"]
    x, residual, value_result = root_trace_results()
    executed = np.asarray(value_result.trace.executed)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    ax = axes[0]
    ax.plot(x, residual, color=NEUTRAL)
    ax.scatter(
        np.asarray(value_result.trace.proposal)[executed],
        np.asarray(value_result.trace.residual)[executed],
        color=PALETTE[4],
        s=20,
        label="Executed branch history",
    )
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.8)
    ax.set(
        xlabel="Coordinate x",
        ylabel="Signed residual",
        title="Value-first: audit the executed map",
    )
    ax.legend(frameon=False)
    polish_axes(ax, grid_axis="y")

    ax = axes[1]
    labels = ["Analytic", "IFT AD", "Central FD"]
    values = [comparison["analytic"], comparison["ad"], comparison["fd"]]
    ax.bar(labels, values, color=[PALETTE[0], POSITIVE, PALETTE[2]], width=0.65)
    ax.axhline(comparison["analytic"], color=NEUTRAL, linewidth=0.8, linestyle="--")
    ax.set_ylim(0.0, 0.43)
    certificate_label = (
        f"certified={bool(certified.certified)}; assertions + all runtime gates pass"
    )
    ax.text(
        0.03,
        0.96,
        certificate_label,
        transform=ax.transAxes,
        va="top",
        color=POSITIVE,
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    ax.text(
        0.03,
        0.88,
        f"flat root rejected: certified = {bool(rejected.certified)}",
        transform=ax.transAxes,
        va="top",
        color=NEGATIVE,
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    ax.set(
        ylabel=r"$dx^\star/d\theta$",
        title="IFT: derivative of the certified relation",
    )
    polish_axes(ax, grid_axis="y")
    fig.suptitle(
        f"Same root question, different derivative claims (root = {float(certified.root):.6f})",
        fontsize=9.5,
    )
    fig.tight_layout()
    return fig
