# src/jaxstro/numerics/quadrature.py
r"""
Gaussian quadrature factory: Gauss-Legendre, Gauss-Hermite (probabilists'),
and the probabilists' Hermite-e polynomial basis / expansion coefficients.

Quadrature theory
-----------------
An ``n``-point Gaussian rule with nodes :math:`x_i` and weights :math:`w_i`
integrates polynomials **exactly up to degree** :math:`2n-1`:

.. math::

    \int_a^b f(x)\, \omega(x)\, dx \;\approx\; \sum_{i=1}^{n} w_i\, f(x_i).

- **Gauss-Legendre**: :math:`\omega(x) = 1` on :math:`[-1, 1]`.
- **Gauss-Hermite (probabilists')**: :math:`\omega(x) = e^{-x^2/2}/\sqrt{2\pi}`
  on :math:`(-\infty, \infty)`, i.e. the standard-normal density. The rule then
  computes expectations under :math:`\mathcal{N}(0, 1)`:
  :math:`\langle f \rangle = \sum_i w_i\, f(x_i)`, with :math:`\sum_i w_i = 1`.

Nodes/weights are the classical Gauss rule whose nodes are the roots of the
orthogonal polynomial and whose weights follow from the Golub & Welsch (1969)
eigenvalue construction (as implemented by NumPy's polynomial routines).

JAX-native exception (sanctioned, setup-only)
---------------------------------------------
Node/weight **generation** is a one-time, HOST-SIDE constant setup: we call
``numpy.polynomial`` (``leggauss`` / ``hermgauss``) to compute the nodes and
weights, then **freeze** them to ``jnp.asarray`` constants with a **static**
``n``. This does **not** violate the JAX-native rule:

- The nodes/weights are *constants*, computed once at call time (not in a hot
  loop, not traced). They carry no parameter dependence.
- Every downstream operation -- evaluating the integrand at the nodes, the
  weighted sum, the Hermite recurrence -- is pure ``jax.numpy`` and fully
  differentiable. ``jax.grad`` flows through the **integrand values**, never
  through the (constant) nodes. Generating nodes with numpy is therefore both
  correct and performant; this is the one place ``numpy`` appears, by design.

References
----------
- Golub, G. H. & Welsch, J. H. 1969, "Calculation of Gauss Quadrature Rules",
  Math. Comp. 23, 221 (the eigenvalue construction behind the NumPy routines).
- NumPy ``numpy.polynomial.legendre.leggauss`` (Gauss-Legendre) and
  ``numpy.polynomial.hermite.hermgauss`` (physicists' Gauss-Hermite); the
  probabilists' rule is obtained from the physicists' rule by the substitution
  :math:`g = \sqrt{2}\, x`, :math:`w \mapsto w / \sqrt{\pi}` (see below).
- Probabilists' Hermite ``He_n`` recurrence: Abramowitz & Stegun (1964),
  22.7; ``He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)``.
"""

from typing import Callable

import jax.numpy as jnp
import numpy as np  # constants only: quadrature node/weight generation at call time
from jaxtyping import Array, Float


def gauss_legendre_nodes(n: int) -> tuple[Array, Array]:
    r"""Gauss-Legendre nodes and weights on :math:`[-1, 1]`.

    Returns ``(nodes, weights)`` such that
    :math:`\int_{-1}^{1} f(x)\,dx \approx \sum_i w_i f(x_i)`, exact for
    polynomials up to degree :math:`2n - 1`. The weights sum to ``2`` (the
    length of the interval).

    Parameters
    ----------
    n : int
        Number of quadrature points (static; host-side node generation).

    Notes
    -----
    Nodes/weights are generated host-side via
    ``numpy.polynomial.legendre.leggauss`` (Golub & Welsch 1969 eigenvalue
    construction) and frozen to ``jnp`` constants. They are exact constants;
    differentiability of any quadrature built on top of them flows through the
    integrand values, not these nodes.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


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
        Number of quadrature points (static; host-side node generation).

    Notes
    -----
    The probabilists' rule (weight :math:`e^{-g^2/2}`, normalized to a
    probability density) is obtained from the **physicists'** rule
    (``numpy.polynomial.hermite.hermgauss``, weight :math:`e^{-x^2}`) by the
    classical substitution :math:`g = \sqrt{2}\,x`, :math:`w_i \mapsto
    w_i / \sqrt{\pi}`. This construction is chosen deliberately so the output is
    **byte-identical** to the progenax ``_gauss_hermite`` rule it consolidates.
    """
    # Physicists' Gauss-Hermite (weight e^{-x^2}); substitute g = sqrt(2) x to
    # obtain the probabilists' rule for expectations under N(0,1).
    x, w = np.polynomial.hermite.hermgauss(n)
    g_nodes = jnp.asarray(np.sqrt(2.0) * x)
    weights = jnp.asarray(w / np.sqrt(np.pi))
    return g_nodes, weights


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


__all__ = [
    "gauss_legendre_nodes",
    "gauss_hermite_nodes",
    "hermite_e_basis",
    "hermite_coefficients",
]
