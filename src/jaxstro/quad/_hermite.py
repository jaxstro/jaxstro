"""Probabilists' Gauss-Hermite rule and Hermite-e expansions for N(0, 1)."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._recurrence import gaussian_rule_data
from .measures import StandardNormalMeasure
from .rules import GaussianRule


def gauss_hermite_nodes(n: int) -> tuple[Array, Array]:
    r"""Probabilists' Gauss-Hermite nodes and weights for :math:`\mathcal{N}(0,1)`.

    Returns ``(nodes, weights)`` such that the standard-normal expectation
    :math:`\langle f \rangle = \int f(g)\,\phi(g)\,dg \approx \sum_i w_i f(g_i)`,
    where :math:`\phi` is the standard-normal density. Exact for polynomials up
    to degree :math:`2n - 1`; the weights sum to ``1`` and reproduce the
    Gaussian moments :math:`\langle g^{2m} \rangle = (2m-1)!!`.

    Parameters
    ----------
    n : int
        Number of quadrature points (static).

    Notes
    -----
    The rule comes from the shared Golub-Welsch engine for
    :class:`~jaxstro.quad.StandardNormalMeasure` (recurrence ``a_k = 0``,
    ``b_k = sqrt(k)``), with Newton-refined nodes and Christoffel-function
    weights that keep their relative accuracy in the tails. It replaced
    ``numpy.polynomial.hermite.hermgauss`` on 2026-09-24; against a 60-digit
    reference its tail weights are more accurate (n = 256: 328 eps against
    572 eps relative).
    """
    data = gaussian_rule_data(GaussianRule(n), StandardNormalMeasure())
    return data.nodes, data.weights


def hermite_e_basis(g: Float[Array, " q"], n_max: int) -> Float[Array, " n q"]:
    r"""Probabilists' Hermite polynomials :math:`He_0..He_{n_{max}}` at points ``g``.

    Returns an array of shape ``(n_max + 1, q)`` whose row ``n`` is
    :math:`He_n(g)`, built from the stable upward recurrence

    .. math::

        He_0 = 1, \quad He_1 = g, \quad
        He_{n+1}(g) = g\,He_n(g) - n\,He_{n-1}(g)

    (Abramowitz & Stegun 1964, 22.7). Pure JAX, differentiable in ``g``.
    """
    rows = [jnp.ones_like(g)]
    if n_max >= 1:
        rows.append(g)
    for n in range(1, n_max):
        rows.append(g * rows[n] - n * rows[n - 1])
    return jnp.stack(rows, axis=0)


def hermite_coefficients(
    map_fn: Callable[[Float[Array, " q"]], Float[Array, " q"]],
    n_max: int,
    n_quad: int = 256,
) -> Float[Array, " n"]:
    r"""Probabilists' Hermite-e expansion coefficients of ``map_fn``.

    Computes ``c_n = <map_fn(g) He_n(g)>`` for ``n = 0..n_max``, where the
    expectation is under :math:`\mathcal{N}(0, 1)` evaluated by an ``n_quad``-point
    Gauss-Hermite rule. Returns shape ``(n_max + 1,)``.

    ``c_0`` is the mean :math:`\langle map\_fn \rangle`; ``c_n`` for ``n >= 1``
    are the higher Hermite-e coefficients (e.g. for the Mehler bivariate-Hermite
    2-point series).

    Differentiability
    -----------------
    ``c_n`` is differentiable in any parameter that ``map_fn`` closes over: the
    nodes/weights are constants, and the gradient flows through the integrand
    values ``map_fn(g_nodes)``. The integration nodes are NOT differentiated.

    Parameters
    ----------
    map_fn : Callable[[Array], Array]
        The scalar map evaluated at the quadrature nodes; may close over
        differentiable parameters.
    n_max : int
        Highest Hermite-e order to compute (static).
    n_quad : int
        Number of Gauss-Hermite quadrature points (static; default 256).
    """
    g_nodes, weights = gauss_hermite_nodes(n_quad)
    values = map_fn(g_nodes)
    he = hermite_e_basis(g_nodes, n_max)
    return (he * (values * weights)[None, :]).sum(axis=1)
