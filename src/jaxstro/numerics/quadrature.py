# src/jaxstro/numerics/quadrature.py
r"""
Gaussian quadrature factory: Gauss-Legendre, Gauss-Laguerre,
Gauss-Hermite (probabilists'), Clenshaw-Curtis, and the probabilists'
Hermite-e polynomial basis / expansion coefficients.

Quadrature theory
-----------------
An ``n``-point Gaussian rule with nodes :math:`x_i` and weights :math:`w_i`
integrates polynomials **exactly up to degree** :math:`2n-1`:

.. math::

    \int_a^b f(x)\, \omega(x)\, dx \;\approx\; \sum_{i=1}^{n} w_i\, f(x_i).

- **Gauss-Legendre**: :math:`\omega(x) = 1` on :math:`[-1, 1]`.
- **Gauss-Laguerre**: :math:`\omega(x) = e^{-x}` on :math:`[0, \infty)`.
- **Gauss-Hermite (probabilists')**: :math:`\omega(x) = e^{-x^2/2}/\sqrt{2\pi}`
  on :math:`(-\infty, \infty)`, i.e. the standard-normal density. The rule then
  computes expectations under :math:`\mathcal{N}(0, 1)`:
  :math:`\langle f \rangle = \sum_i w_i\, f(x_i)`, with :math:`\sum_i w_i = 1`.
- **Clenshaw-Curtis**: interpolatory quadrature on Chebyshev-Lobatto nodes
  :math:`x_i = \cos(i\pi/(n-1))` over :math:`[-1, 1]`.

The Gaussian rules come from the shared Golub-Welsch engine and the
Clenshaw-Curtis rule from the Chebyshev module of :mod:`jaxstro.quad`. This
module is a compatibility import lane.

References
----------
- Golub, G. H. & Welsch, J. H. 1969, "Calculation of Gauss Quadrature Rules",
  Math. Comp. 23, 221 (the eigenvalue construction of the nodes).
- Trefethen, L. N. 2008, "Is Gauss quadrature better than Clenshaw-Curtis?",
  SIAM Review 50, 67 (Clenshaw-Curtis context and algorithmic comparison).
- Probabilists' Hermite ``He_n`` recurrence: Abramowitz & Stegun (1964),
  22.7; ``He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)``.
"""

from jaxstro.quad._chebyshev import clenshaw_curtis_nodes
from jaxstro.quad._hermite import (
    gauss_hermite_nodes,
    hermite_coefficients,
    hermite_e_basis,
)
from jaxstro.quad._recurrence import gauss_laguerre_nodes, gauss_legendre_nodes

__all__ = [
    "gauss_legendre_nodes",
    "gauss_hermite_nodes",
    "gauss_laguerre_nodes",
    "clenshaw_curtis_nodes",
    "hermite_e_basis",
    "hermite_coefficients",
]
