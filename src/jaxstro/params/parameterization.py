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

Unconstrained-space transforms
------------------------------
Each free leaf may carry a :class:`~jaxstro.params.transforms.AbstractBijector`
that maps it to/from an *unconstrained* real space. The flat vector then lives in
:math:`\\mathbb{R}^n` even when the physical parameters are bounded (e.g.
``r_h > 0`` via :class:`~jaxstro.params.transforms.Exp`, or ``0 < Q < 1`` via
:class:`~jaxstro.params.transforms.Sigmoid`). With the default
(``transforms=None``) every free leaf uses
:class:`~jaxstro.params.transforms.Identity`, recovering the identity
parameterization byte-for-byte.

The bijectors are stored **co-aligned with the free leaves** as a static
leaf-aligned PyTree (``transform_spec``), built by riding the *same*
:func:`equinox.tree_at` lowering used for ``free_spec``. The flat vector is
ordered by PyTree-leaf order, so each leaf's bijector travels with its leaf by
construction -- the ``where``-tuple order is irrelevant.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree

from .transforms import AbstractBijector, Identity


def _is_bijector(x: object) -> bool:
    """Leaf predicate: treat a whole bijector as a single PyTree leaf."""
    return isinstance(x, AbstractBijector)


class _Meta:
    """Opaque, hashable per-free-leaf ``(shape, bijector)`` carrier.

    Deliberately *not* a registered PyTree: JAX treats it as a single opaque
    leaf, so it survives ``tree_map``/``tree_leaves`` intact and stays static
    (no array leaves). Holds a free leaf's ``shape`` and its bijector.
    """

    __slots__ = ("shape", "bijector")

    def __init__(self, shape: tuple, bijector: AbstractBijector) -> None:
        self.shape = shape
        self.bijector = bijector


def _is_meta(x: object) -> bool:
    """Leaf predicate: treat a :class:`_Meta` as a single PyTree leaf."""
    return isinstance(x, _Meta)


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
    transform_spec: PyTree = eqx.field(static=True)
    free_meta: tuple = eqx.field(static=True)

    @classmethod
    def from_filter(
        cls,
        model: PyTree,
        free_spec: PyTree,
        transforms: Optional[Sequence[AbstractBijector]] = None,
    ) -> "Parameterization":
        """Construct from an explicit boolean filter spec.

        Parameters
        ----------
        model : PyTree
            The Equinox model whose leaves ``free_spec`` is aligned with. Used
            both to lower ``transforms`` to a leaf-aligned ``transform_spec``
            and for API symmetry with :meth:`from_where`.
        free_spec : PyTree of bool
            Boolean PyTree marking free (``True``) vs fixed (``False``) array
            leaves of ``model``.
        transforms : sequence of AbstractBijector, optional
            Bijectors aligned with the *free* leaves of ``free_spec`` in PyTree
            order (one per free leaf). ``None`` (default) uses
            :class:`~jaxstro.params.transforms.Identity` for every free leaf.

        Returns
        -------
        Parameterization
            A parameterization carrying ``free_spec`` and a leaf-aligned
            ``transform_spec``.
        """
        transform_spec = cls._build_transform_spec(model, free_spec, transforms)
        return cls(
            free_spec=free_spec,
            transform_spec=transform_spec,
            free_meta=cls._build_free_meta(model, free_spec, transform_spec),
        )

    @staticmethod
    def _build_free_meta(
        model: PyTree, free_spec: PyTree, transform_spec: PyTree
    ) -> tuple:
        """Per-free-leaf ``(shape, bijector)`` metadata in PyTree order.

        Built in a SINGLE structural pass: ``eqx.filter(model, free_spec)`` (the
        free physical leaves) is paired leaf-for-leaf with the same-filtered
        ``transform_spec`` (the free bijectors) under one
        :func:`jax.tree_util.tree_map`. Because both subtrees are filtered by the
        identical ``free_spec`` and walked together, the ``(shape, bijector)``
        pairs are guaranteed aligned -- no second, independent ``tree_leaves``
        enumeration that could desync. Stored statically (bijectors carry no
        array leaves) and consumed by :meth:`log_det_jacobian`.
        """
        free = eqx.filter(model, free_spec)
        free_transforms = eqx.filter(
            transform_spec, free_spec, is_leaf=_is_bijector
        )
        pairs = jax.tree_util.tree_map(
            lambda leaf, bij: _Meta(jnp.shape(leaf), bij),
            free,
            free_transforms,
            is_leaf=_is_bijector,
        )
        # Each _Meta(shape, bijector) is a single leaf (it has no array leaves),
        # so tree_leaves returns them in PyTree order -- the same order
        # to_vector's ravel uses (JAX's deterministic tree-flatten).
        return tuple(
            (m.shape, m.bijector)
            for m in jax.tree_util.tree_leaves(pairs, is_leaf=_is_meta)
        )

    @classmethod
    def from_where(
        cls,
        model: PyTree,
        where: Callable[[PyTree], object],
        transforms: Optional[Sequence[AbstractBijector]] = None,
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
        transforms : sequence of AbstractBijector, optional
            Bijectors aligned with the ``where`` *selection* order (one per
            selected leaf). They are lowered to a static leaf-aligned
            ``transform_spec`` by riding the *same* :func:`equinox.tree_at`
            lowering as ``free_spec``, so each bijector travels with its leaf
            irrespective of the ``where``-tuple ordering. ``None`` (default)
            uses :class:`~jaxstro.params.transforms.Identity` for every free
            leaf.

        Returns
        -------
        Parameterization
            A parameterization with the selected leaves marked free and a
            leaf-aligned ``transform_spec``.

        Notes
        -----
        The empty-selection case (``where=lambda m: ()``) is handled
        explicitly: :func:`equinox.tree_at` is not invoked (it requires a
        non-empty replacement), so the all-``False`` template is used directly.
        """
        bool_template = jax.tree_util.tree_map(
            lambda _: False, eqx.filter(model, eqx.is_array)
        )
        selected_raw = where(model)
        # ``where`` may return a single leaf or a tuple of leaves. Track which,
        # so the ``replace`` arity passed to ``eqx.tree_at`` matches: a single
        # leaf takes a scalar replacement, a tuple takes a tuple. (Passing a
        # 1-tuple for a single-leaf ``where`` would write the tuple *into* the
        # leaf, corrupting the spec.)
        if isinstance(selected_raw, tuple):
            is_single = False
            selected = selected_raw
        else:
            is_single = True
            selected = (selected_raw,)

        if len(selected) == 0:
            # No free leaves: the all-False template is already correct.
            free_spec = bool_template
        else:
            free_spec = eqx.tree_at(
                where,
                bool_template,
                replace=True if is_single else tuple(True for _ in selected),
            )

        if transforms is None:
            return cls.from_filter(model, free_spec, transforms=None)

        if len(transforms) != len(selected):
            raise ValueError(
                "transforms must align 1:1 with the where selection: got "
                f"{len(transforms)} transforms for {len(selected)} selected "
                "leaves."
            )
        # Lower transforms onto an Identity template by riding the SAME tree_at
        # lowering as free_spec; tuple order is discarded exactly as for it.
        identity_template = jax.tree_util.tree_map(
            lambda _: Identity(), eqx.filter(model, eqx.is_array)
        )
        transform_spec = eqx.tree_at(
            where,
            identity_template,
            replace=transforms[0] if is_single else tuple(transforms),
            is_leaf=_is_bijector,
        )
        return cls(
            free_spec=free_spec,
            transform_spec=transform_spec,
            free_meta=cls._build_free_meta(model, free_spec, transform_spec),
        )

    @staticmethod
    def _build_transform_spec(
        model: PyTree,
        free_spec: PyTree,
        transforms: Optional[Sequence[AbstractBijector]],
    ) -> PyTree:
        """Build a leaf-aligned ``transform_spec`` (all-``Identity`` default).

        For the explicit-spec path (:meth:`from_filter`), ``transforms`` are
        aligned with the free leaves of ``free_spec`` in PyTree order.
        """
        identity_template = jax.tree_util.tree_map(
            lambda _: Identity(), eqx.filter(model, eqx.is_array)
        )
        if transforms is None:
            return identity_template
        transforms = list(transforms)
        # Length precheck: count free (True) leaves and require a 1:1 match so
        # too-FEW transforms raise a clean ValueError (matching the too-MANY
        # case below) instead of a PEP 479 RuntimeError from the next() below.
        n_free = sum(
            bool(b) for b in jax.tree_util.tree_leaves(free_spec)
        )
        if len(transforms) != n_free:
            raise ValueError(
                "transforms must align 1:1 with the free leaves of free_spec "
                f"in PyTree order: got {len(transforms)} transforms for "
                f"{n_free} free leaves."
            )
        it = iter(transforms)
        # Walk array leaves in PyTree order; assign the next bijector at each
        # free (True) leaf, Identity at fixed (False) leaves.
        spec = jax.tree_util.tree_map(
            lambda is_free, ident: (next(it) if is_free else ident),
            free_spec,
            identity_template,
            is_leaf=_is_bijector,
        )
        leftover = list(it)
        if leftover:
            raise ValueError(
                "transforms has more entries than free leaves in free_spec."
            )
        return spec

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

    def _free_transforms(self) -> PyTree:
        """Bijector tree aligned with the *free* partition structure.

        Returns the leaf-aligned ``transform_spec`` filtered to free leaves
        (bijectors at free leaves, ``None`` elsewhere), so it shares the exact
        PyTree structure of ``eqx.filter(model, free_spec)`` and can be paired
        with it under :func:`jax.tree_util.tree_map`.
        """
        return eqx.filter(
            self.transform_spec, self.free_spec, is_leaf=_is_bijector
        )

    def to_vector(self, model: PyTree) -> Float[Array, " n"]:
        r"""Flatten the free leaves of ``model`` into an unconstrained vector.

        Each free leaf is mapped to unconstrained :math:`\mathbb{R}` by its
        bijector's :meth:`~jaxstro.params.transforms.AbstractBijector.inverse`
        (the identity under the default all-``Identity`` spec), then all free
        leaves are raveled in PyTree order.

        Parameters
        ----------
        model : PyTree
            The Equinox model to read free leaves from.

        Returns
        -------
        jax.Array, shape ``(n,)``
            The concatenation of all (inverse-transformed) free leaves in
            PyTree order, where ``n`` is the total number of scalar entries
            across free leaves. If no leaves are free, the result has shape
            ``(0,)``.
        """
        free = eqx.filter(model, self.free_spec)
        unconstrained = jax.tree_util.tree_map(
            lambda leaf, bij: bij.inverse(leaf),
            free,
            self._free_transforms(),
            is_leaf=_is_bijector,
        )
        return ravel_pytree(unconstrained)[0]

    def from_vector(
        self, model: PyTree, vec: Float[Array, " n"]
    ) -> PyTree:
        r"""Reconstruct a model from an unconstrained ``vec``.

        The vector is unflattened into the free-partition structure, each free
        leaf is mapped back to physical space by its bijector's
        :meth:`~jaxstro.params.transforms.AbstractBijector.forward` (identity
        under the default spec), and the result is recombined with ``model``'s
        fixed and static leaves.

        Parameters
        ----------
        model : PyTree
            The reference model supplying both the free-leaf structure and the
            fixed/static leaves to carry through.
        vec : jax.Array, shape ``(n,)``
            Flat vector of free-parameter values in *unconstrained* space, in
            the same ordering produced by :meth:`to_vector`.

        Returns
        -------
        PyTree
            A new model identical to ``model`` except with free leaves replaced
            by the (forward-transformed) values unflattened from ``vec``.

        Notes
        -----
        This method is differentiable in ``vec``: ``jax.grad`` of a scalar loss
        composed with :meth:`from_vector` flows cleanly back to the flat vector.
        """
        free, fixed = self._partition(model)
        _, unravel = ravel_pytree(free)
        unconstrained = unravel(vec)
        physical = jax.tree_util.tree_map(
            lambda leaf, bij: bij.forward(leaf),
            unconstrained,
            self._free_transforms(),
            is_leaf=_is_bijector,
        )
        return eqx.combine(physical, fixed)

    def log_det_jacobian(self, vec: Float[Array, " n"]) -> Float[Array, ""]:
        r"""Total log-abs-det Jacobian of the unconstrained -> physical map.

        For a change of variables from the unconstrained vector ``vec`` to the
        physical free leaves, the log absolute determinant of the (diagonal)
        Jacobian is the sum over free leaves of each bijector's
        :meth:`~jaxstro.params.transforms.AbstractBijector.forward_log_det_jacobian`.
        Use this as the change-of-variables term when sampling/optimizing in
        unconstrained space (e.g. a numpyro model on ``vec``).

        Parameters
        ----------
        vec : jax.Array, shape ``(n,)``
            Flat vector in *unconstrained* space, ordered as :meth:`to_vector`.

        Returns
        -------
        jax.Array, scalar
            :math:`\sum_i \log\left|\partial\,\mathrm{forward}_i / \partial u_i\right|`
            over all scalar entries of all free leaves. Zero (scalar) when no
            leaves are free or every bijector is :class:`Identity`.
        """
        # Consume the SINGLE pre-aligned ``(shape, bijector)`` structure built
        # in one structural pass at construction (``_build_free_meta``). No
        # runtime tree_leaves and no parallel-ordering assumption: ordering
        # matches to_vector's ravel by JAX's deterministic tree-flatten (the
        # same invariant to_vector/from_vector already rely on). Each bijector
        # acts element-wise, so the diagonal Jacobian's log-abs-det is the sum
        # over all scalar entries of every free leaf.
        if not self.free_meta:
            return jnp.zeros(())

        total = jnp.zeros(())
        offset = 0
        for shape, bij in self.free_meta:
            size = 1
            for d in shape:
                size *= d
            seg = jax.lax.dynamic_slice_in_dim(vec, offset, size)
            total = total + jnp.sum(bij.forward_log_det_jacobian(seg))
            offset += size
        return total
