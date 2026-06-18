# src/jaxstro/params/transforms.py
"""
Bijector registry: differentiable maps between unconstrained :math:`\\mathbb{R}`
and constrained physical parameter spaces, with **analytic** log-Jacobians.

Convention (numpyro/TFP)
------------------------
``forward(u)`` maps an **unconstrained** real value ``u`` to the **constrained**
physical value ``x``; ``inverse(x)`` is its reciprocal. ``forward_log_det_jacobian(u)``
returns the log absolute determinant of the Jacobian of ``forward`` evaluated at
``u``, i.e.

.. math::

    \\log\\left| \\frac{\\partial\\, \\text{forward}(u)}{\\partial u} \\right| .

This is the change-of-variables term needed to push a density from constrained to
unconstrained space (so that, e.g., a numpyro model sampling in
unconstrained space evaluates the correct constrained posterior).

All log-Jacobians are written **analytically** (never autodiffed determinants),
in a float64-stable form using :func:`jax.nn.softplus` and
:func:`jax.nn.log_sigmoid`. Each method is ``jit``/``grad``/``vmap``-safe and
JAX-native (no numpy).

The shared base class :class:`AbstractBijector` lets callers treat a whole
bijector as a single PyTree leaf via
``is_leaf=lambda x: isinstance(x, AbstractBijector)``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["AbstractBijector", "Identity", "Exp", "Softplus", "Sigmoid"]

Scalar = Float[Array, ""]


class AbstractBijector(eqx.Module):
    """Abstract base for a scalar bijector ``forward: R -> physical``.

    Subclasses implement :meth:`forward`, :meth:`inverse`, and
    :meth:`forward_log_det_jacobian`. The class is a plain ``eqx.Module`` so
    that an instance is a PyTree; downstream code treats it as a single leaf
    via ``is_leaf=lambda x: isinstance(x, AbstractBijector)``.

    Notes
    -----
    Methods are defined per-element on scalars; apply :func:`jax.vmap` for
    arrays. The change-of-variables identity each subclass must satisfy is

    .. math::

        \\text{forward\\_log\\_det\\_jacobian}(u)
            = \\log\\left| \\frac{d\\,\\text{forward}(u)}{du} \\right| .
    """

    def forward(self, u: Scalar) -> Scalar:  # pragma: no cover - abstract
        """Map unconstrained ``u`` to constrained physical value."""
        raise NotImplementedError

    def inverse(self, x: Scalar) -> Scalar:  # pragma: no cover - abstract
        """Map constrained physical ``x`` back to unconstrained ``u``."""
        raise NotImplementedError

    def forward_log_det_jacobian(self, u: Scalar) -> Scalar:  # pragma: no cover
        """Analytic ``log|d forward / du|`` evaluated at ``u``."""
        raise NotImplementedError


class Identity(AbstractBijector):
    r"""Identity map :math:`x = u`.

    The Jacobian is unity, so :math:`\log|d x / d u| = 0`.
    """

    def forward(self, u: Scalar) -> Scalar:
        return u

    def inverse(self, x: Scalar) -> Scalar:
        return x

    def forward_log_det_jacobian(self, u: Scalar) -> Scalar:
        return jnp.zeros_like(u)


class Exp(AbstractBijector):
    r"""Exponential map :math:`x = e^{u}` for strictly-positive parameters.

    With :math:`x = e^{u}` we have :math:`dx/du = e^{u} = x`, hence

    .. math::

        \log\left| \frac{dx}{du} \right| = u .

    Suitable for ``mass > 0``, ``r_h > 0``.
    """

    def forward(self, u: Scalar) -> Scalar:
        return jnp.exp(u)

    def inverse(self, x: Scalar) -> Scalar:
        return jnp.log(x)

    def forward_log_det_jacobian(self, u: Scalar) -> Scalar:
        return u


class Softplus(AbstractBijector):
    r"""Softplus map :math:`x = \mathrm{softplus}(u) = \log(1 + e^{u})`.

    A gentler positivity transform than :class:`Exp` (linear for large ``u``).
    Its derivative is the logistic sigmoid,

    .. math::

        \frac{dx}{du} = \sigma(u) = \frac{1}{1 + e^{-u}},
        \qquad
        \log\left| \frac{dx}{du} \right| = \log \sigma(u),

    computed stably with :func:`jax.nn.log_sigmoid`. The inverse is
    :math:`u = \log(e^{x} - 1)`, evaluated via :func:`jnp.expm1` for accuracy
    near ``x = 0``.
    """

    def forward(self, u: Scalar) -> Scalar:
        return jax.nn.softplus(u)

    def inverse(self, x: Scalar) -> Scalar:
        return jnp.log(jnp.expm1(x))

    def forward_log_det_jacobian(self, u: Scalar) -> Scalar:
        return jax.nn.log_sigmoid(u)


class Sigmoid(AbstractBijector):
    r"""Affine-scaled sigmoid onto the open interval :math:`(lo, hi)`.

    .. math::

        x = lo + (hi - lo)\, \sigma(u),
        \qquad \sigma(u) = \frac{1}{1 + e^{-u}} .

    The derivative factorises through :math:`\sigma'(u) = \sigma(u)\,\sigma(-u)`,
    giving the stable analytic log-Jacobian

    .. math::

        \log\left| \frac{dx}{du} \right|
            = \log(hi - lo) + \log \sigma(u) + \log \sigma(-u),

    where each :math:`\log \sigma(\cdot)` is evaluated with
    :func:`jax.nn.log_sigmoid`. The inverse uses the numerically stable logit
    :func:`jax.scipy.special.logit` on the rescaled value
    :math:`(x - lo) / (hi - lo)`.

    Parameters
    ----------
    lo, hi : float
        Lower/upper bounds of the constrained interval, stored as static
        (non-traced) floats. Requires ``hi > lo``.

    Notes
    -----
    Suitable for bounded parameters such as a virial ratio ``0 < Q < 1``.
    """

    lo: float = eqx.field(static=True)
    hi: float = eqx.field(static=True)

    def forward(self, u: Scalar) -> Scalar:
        return self.lo + (self.hi - self.lo) * jax.scipy.special.expit(u)

    def inverse(self, x: Scalar) -> Scalar:
        return jax.scipy.special.logit((x - self.lo) / (self.hi - self.lo))

    def forward_log_det_jacobian(self, u: Scalar) -> Scalar:
        width = jnp.log(jnp.asarray(self.hi - self.lo))
        return width + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u)
