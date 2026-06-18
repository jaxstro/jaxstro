# src/jaxstro/params/parameterization.py
"""
Free/fixed parameter marking and a PyTree <-> flat-vector bridge.

This module provides :class:`Parameterization`, a small, pure, static helper
that marks a subset of an Equinox model's array leaves as *free* (to be
optimized or sampled) while holding the rest *fixed*, and that maps between the
model (a PyTree) and a flat 1-D vector of just the free parameters.

This is the inverse of the usual problem in differentiable astrophysics: an
optimizer (optax) or sampler (numpyro/blackjax) wants a flat vector ``v`` in
:math:`\\mathbb{R}^n`, while the physics code wants a structured Equinox module.
:class:`Parameterization` is the static, differentiable adapter between the two.

Design notes
------------
- **Static / pure.** The only field is a static boolean PyTree (``free_spec``)
  marking free array leaves. No closures, arrays, or other dynamic state are
  stored on the module, so it is hashable and JIT-friendly as a static argument.
- **JAX-native.** Built entirely on :mod:`jax`, :mod:`equinox`, and
  :func:`jax.flatten_util.ravel_pytree`. No ``numpy``/``scipy``.
- **Differentiable.** :meth:`from_vector` is differentiable in ``vec``; the
  round-trip ``from_vector(model, to_vector(model))`` reconstructs ``model``
  exactly on its array leaves (identity), and gradients flow through the flat
  vector into the reconstructed model (see the module tests).

This Task-1 core supports the *identity* parameterization only (the flat vector
lives in the same space as the free leaves). Unconstrained-space transforms
(bijectors) are layered on in a later task.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree


class Parameterization(eqx.Module):
    r"""Mark free/fixed model leaves and bridge PyTree <-> flat vector.

    A ``Parameterization`` records which array leaves of an Equinox model are
    *free* via a static boolean PyTree (``free_spec``). It then maps between the
    model and a flat 1-D vector containing exactly the free leaves, in PyTree
    flattening order.

    Parameters
    ----------
    free_spec : PyTree of bool
        A boolean PyTree, aligned with the model's *array* leaves, that is
        ``True`` at every free leaf and ``False`` at every fixed leaf.
        Non-array (static) leaves are excluded (``None``). This is a static
        field: it carries no traced data and makes the module hashable.

    Notes
    -----
    The flat vector produced by :meth:`to_vector` has length equal to the total
    number of scalar entries across the free leaves (e.g. a free leaf of shape
    ``(2,)`` contributes 2 entries). The ordering follows
    :func:`jax.flatten_util.ravel_pytree`, i.e. JAX's PyTree leaf order.

    Guarantees
    ----------
    Round-trip identity
        ``from_vector(model, to_vector(model))`` equals ``model`` on every
        array leaf (free leaves are reconstructed from the vector; fixed leaves
        are copied through unchanged).
    Fixed-leaf preservation
        Fixed leaves are never read from or written by the flat vector; they are
        carried through verbatim from the original ``model``.
    Differentiability
        :meth:`from_vector` is differentiable with respect to ``vec`` and is
        compatible with :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

    See Also
    --------
    from_where : Construct from a ``where`` selector over model leaves.
    from_filter : Construct from an explicit boolean spec.
    """

    free_spec: PyTree = eqx.field(static=True)

    @classmethod
    def from_filter(
        cls, model: PyTree, free_spec: PyTree
    ) -> "Parameterization":
        """Construct from an explicit boolean filter spec.

        Parameters
        ----------
        model : PyTree
            The Equinox model whose leaves ``free_spec`` is aligned with. Only
            used for documentation/symmetry with :meth:`from_where`; the spec is
            stored as-is.
        free_spec : PyTree of bool
            Boolean PyTree marking free (``True``) vs fixed (``False``) array
            leaves of ``model``.

        Returns
        -------
        Parameterization
            A parameterization carrying ``free_spec``.
        """
        del model  # stored spec is authoritative; kept for API symmetry
        return cls(free_spec=free_spec)

    @classmethod
    def from_where(
        cls,
        model: PyTree,
        where: Callable[[PyTree], object],
    ) -> "Parameterization":
        """Construct from a ``where`` selector over the model's leaves.

        Builds an all-``False`` boolean template over the model's array leaves,
        sets the leaves selected by ``where`` to ``True``, and delegates to
        :meth:`from_filter`.

        Parameters
        ----------
        model : PyTree
            The Equinox model to parameterize.
        where : callable
            A selector taking ``model`` and returning the free leaf or a tuple
            of free leaves, e.g. ``lambda m: (m.a, m.b)``. Returning an empty
            tuple ``()`` marks *no* leaves free, in which case
            :meth:`to_vector` returns a vector of shape ``(0,)``.

        Returns
        -------
        Parameterization
            A parameterization with the selected leaves marked free.

        Notes
        -----
        The empty-selection case (``where=lambda m: ()``) is handled
        explicitly: :func:`equinox.tree_at` is not invoked (it requires a
        non-empty replacement), so the all-``False`` template is used directly.
        """
        template = jax.tree_util.tree_map(
            lambda _: False, eqx.filter(model, eqx.is_array)
        )
        selected = where(model)
        # Normalize to a tuple of selected leaves so we can count them.
        if not isinstance(selected, tuple):
            selected = (selected,)

        if len(selected) == 0:
            # No free leaves: the all-False template is already correct.
            free_spec = template
        else:
            free_spec = eqx.tree_at(
                where, template, replace=tuple(True for _ in selected)
            )
        return cls.from_filter(model, free_spec)

    def _partition(self, model: PyTree) -> tuple[PyTree, PyTree]:
        """Split ``model`` into (free, fixed) subtrees per ``free_spec``.

        Parameters
        ----------
        model : PyTree
            The Equinox model to partition.

        Returns
        -------
        free : PyTree
            Subtree with free leaves present and all others replaced by
            ``None``.
        fixed : PyTree
            Subtree with fixed (and static) leaves present and free leaves
            replaced by ``None``.
        """
        return eqx.partition(model, self.free_spec)

    def to_vector(self, model: PyTree) -> Float[Array, " n"]:
        """Flatten the free leaves of ``model`` into a 1-D vector.

        Parameters
        ----------
        model : PyTree
            The Equinox model to read free leaves from.

        Returns
        -------
        jax.Array, shape ``(n,)``
            The concatenation of all free leaves in PyTree order, where ``n`` is
            the total number of scalar entries across free leaves. If no leaves
            are free, the result has shape ``(0,)``.
        """
        free = eqx.filter(model, self.free_spec)
        return ravel_pytree(free)[0]

    def from_vector(
        self, model: PyTree, vec: Float[Array, " n"]
    ) -> PyTree:
        """Reconstruct a model from ``vec``, keeping fixed leaves from ``model``.

        The free leaves are unflattened from ``vec`` (using the structure of
        ``model``'s free partition), and recombined with ``model``'s fixed and
        static leaves.

        Parameters
        ----------
        model : PyTree
            The reference model supplying both the free-leaf structure and the
            fixed/static leaves to carry through.
        vec : jax.Array, shape ``(n,)``
            Flat vector of free-parameter values, in the same ordering produced
            by :meth:`to_vector`.

        Returns
        -------
        PyTree
            A new model identical to ``model`` except with free leaves replaced
            by the values unflattened from ``vec``.

        Notes
        -----
        This method is differentiable in ``vec``: ``jax.grad`` of a scalar loss
        composed with :meth:`from_vector` flows cleanly back to the flat vector.
        """
        free, fixed = self._partition(model)
        _, unravel = ravel_pytree(free)
        return eqx.combine(unravel(vec), fixed)
