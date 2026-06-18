#!/usr/bin/env python
# validation/validate_params.py
"""
Scientific validation of :mod:`jaxstro.params` on a real downstream model.

Recovers a known, injected TRUTH for a positive scalar physical parameter two
independent ways, both driven through the ``jaxstro.params`` PyTree<->vector
bridge with an ``Exp`` unconstrained transform (so descent/sampling happens in
unconstrained R while the physical parameter stays strictly positive):

  (a) **optax** gradient descent on the flat unconstrained vector, and
  (b) a tiny **numpyro** NUTS chain that samples the unconstrained vector under a
      simple Normal prior and adds ``parameterization.log_det_jacobian(vec)`` as
      the change-of-variables term, recovering the posterior mean.

It also reports a **finite-difference vs autodiff** gradient check on the loss
(max relative error), confirming the bridge is differentiable end-to-end.

The two backends are independent: the script runs whichever of optax / numpyro
is installed and only aborts if *both* are missing (the ``[ml]`` extra ships
both).

VALIDATION TARGET
-----------------
Primary: ``progenax.PlummerProfile`` -- an ``eqx.Module`` storing a scalar
half-mass radius ``r_h`` AND its derived scale radius ``a`` as *separate* array
leaves. We fit the **scale radius ``a > 0``** to a synthetic enclosed-mass
profile, because that is the leaf the observable ``enclosed_mass_fraction``
actually reads.

  ADOPTION CAVEAT (the reason we fit ``a``, not ``r_h``): ``PlummerProfile``
  caches ``a`` (computed from ``r_h`` in ``__init__``) as its own leaf. The
  ``params`` bridge ``from_vector`` *replaces leaves*; it does NOT re-run
  ``__init__``. So freeing only ``r_h`` would leave the cached ``a`` stale and
  the observable's gradient w.r.t. ``r_h`` is exactly zero. The general rule:
  free the leaf the observable depends on (or use a model that derives such
  quantities lazily from stored leaves). Here ``a`` is that leaf.

Fallback: if progenax (or a suitable model) cannot be imported, a self-contained
toy ``eqx.Module`` that computes its scale radius lazily from ``r_h`` is used
(so ``r_h`` itself is the directly-fit leaf), and the script prints which path
ran.

Run::

    env -u VIRTUAL_ENV uv run --no-sync --extra ml python validation/validate_params.py
"""

from __future__ import annotations

import sys
from typing import Callable

# float64 is essential for the grad-check and tight recovery tolerances.
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxstro.params import Parameterization  # noqa: E402
from jaxstro.params.transforms import Exp  # noqa: E402


# --------------------------------------------------------------------------- #
# Optional ML deps (validation-only, behind the [ml] extra).
# --------------------------------------------------------------------------- #
def _load_ml():
    """Load optax and numpyro *independently*; either may be absent.

    Returns ``(optax_or_None, numpyro_or_None)``. The two recovery paths are
    independent, so the script degrades gracefully: it runs whichever backend is
    present (e.g. optax-only in an env that lacks numpyro) and only aborts if
    *both* are missing.
    """
    try:
        import optax
    except ImportError:
        optax = None
    try:
        import numpyro
    except ImportError:
        numpyro = None

    if optax is None and numpyro is None:  # pragma: no cover - only without [ml]
        print(
            "ERROR: validation needs at least one of optax / numpyro "
            "(the [ml] extra provides both).\n"
            "  run with: env -u VIRTUAL_ENV uv run --no-sync --extra ml "
            "python validation/validate_params.py"
        )
        sys.exit(2)
    return optax, numpyro


# --------------------------------------------------------------------------- #
# Model selection: real progenax model, else a toy fallback.
#
# A "target" is a small record describing how to inject a truth, build the loss,
# and read back the fit parameter:
#   label       : human-readable description of which path ran
#   make_model  : value -> eqx.Module  (constructs the model from the free scalar)
#   predict     : (model, grid) -> observable vector to fit
#   where       : model -> tuple of free leaves (the params marking selector)
#   read_free   : model -> scalar value of the free parameter (for reporting)
#   truth_value : the injected true value of the free parameter
#   free_name   : column label
#   grid        : evaluation grid
# --------------------------------------------------------------------------- #
def _build_target():
    grid = jnp.linspace(0.2, 4.0, 48)

    try:
        # Best-effort: progenax is a *downstream* consumer of jaxstro (one-way
        # dependency arrow), so it is not a jaxstro dependency and won't be on
        # the path by default. Inject its source tree so the real model is used
        # whenever its own deps (diffrax, ...) are present in this env.
        import os

        _progenax_src = os.path.join(
            os.path.dirname(__file__), "..", "..", "progenax", "src"
        )
        if os.path.isdir(_progenax_src) and _progenax_src not in sys.path:
            sys.path.insert(0, _progenax_src)

        import progenax

        Profile = progenax.PlummerProfile

        # We fit the scale radius ``a`` -- the leaf ``enclosed_mass_fraction``
        # reads -- not ``r_h`` (whose effect is mediated by the cached ``a``
        # leaf that ``from_vector`` does not recompute). See module docstring.
        def make_model(a_value):
            # Construct from r_h=1 (giving some a), then overwrite the ``a`` leaf
            # with the requested value via eqx.tree_at, so ``a`` is set directly.
            base = Profile(r_h=1.0)
            return eqx.tree_at(lambda m: m.a, base, jnp.asarray(a_value))

        def predict(model, x):
            # Enclosed-mass fraction M(<r)/M is a smooth, monotone, well-scaled
            # observable in [0, 1] -- a clean target. It depends on ``a``.
            return model.enclosed_mass_fraction(x)

        # truth: the scale radius implied by a Plummer profile with r_h = 1.6.
        a_truth = float(Profile(r_h=1.6).a)
        label = "progenax.PlummerProfile (free: scale radius a > 0)"
        return (
            label,
            make_model,
            predict,
            (lambda m: (m.a,)),
            (lambda m: jnp.ravel(m.a)[0]),  # traceable scalar (vmap-safe)
            a_truth,
            "a",
            grid,
        )

    except Exception as exc:  # pragma: no cover - only without progenax
        reason = f"{type(exc).__name__}: {exc}"

        class _ToyPlummer(eqx.Module):
            # Stores only r_h; the scale radius is derived lazily inside the
            # observable, so r_h IS the leaf the observable depends on.
            r_h: jax.Array

            def enclosed(self, r):
                a = self.r_h * jnp.sqrt((1.0 - 0.5 ** (2 / 3)) / 0.5 ** (2 / 3))
                return r**3 / (r**2 + a**2) ** 1.5

        def make_model(r_h):
            return _ToyPlummer(r_h=jnp.asarray(r_h))

        def predict(model, x):
            return model.enclosed(x)

        label = f"toy fallback (progenax unavailable: {reason})"
        return (
            label,
            make_model,
            predict,
            (lambda m: (m.r_h,)),
            (lambda m: jnp.ravel(m.r_h)[0]),  # traceable scalar (vmap-safe)
            1.6,
            "r_h",
            grid,
        )


# --------------------------------------------------------------------------- #
# Recovery (a): optax gradient descent.
# --------------------------------------------------------------------------- #
def _recover_optax(optax, param, init_model, predict, grid, data, read_free):
    @jax.jit
    def loss(vec):
        model = param.from_vector(init_model, vec)
        return jnp.mean((predict(model, grid) - data) ** 2)

    optimizer = optax.adam(learning_rate=5e-2)
    vec = param.to_vector(init_model)
    opt_state = optimizer.init(vec)

    @jax.jit
    def step(vec, opt_state):
        grads = jax.grad(loss)(vec)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(vec, updates), opt_state

    for _ in range(3000):
        vec, opt_state = step(vec, opt_state)

    recovered = param.from_vector(init_model, vec)
    return float(read_free(recovered)), loss, vec


# --------------------------------------------------------------------------- #
# Recovery (b): tiny numpyro NUTS chain over the unconstrained vector.
# --------------------------------------------------------------------------- #
def _recover_numpyro(
    numpyro, param, init_model, predict, grid, data, noise_sigma, read_free
):
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model_fn():
        # Sample the unconstrained free vector with a broad standard-normal
        # prior on R, then add the change-of-variables Jacobian so the implied
        # prior on the *physical* parameter is consistent.
        vec = numpyro.sample("vec", dist.Normal(jnp.zeros(1), 5.0).to_event(1))
        numpyro.factor("ldj", param.log_det_jacobian(vec))
        model = param.from_vector(init_model, vec)
        mu = predict(model, grid)
        numpyro.sample("obs", dist.Normal(mu, noise_sigma).to_event(1), obs=data)

    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=400, num_samples=600, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()["vec"]  # (num_samples, 1), unconstrained
    # Map each unconstrained draw forward to the physical free param, mean it.
    draws = jax.vmap(lambda v: read_free(param.from_vector(init_model, v)))(samples)
    return float(jnp.mean(jnp.ravel(draws)))


# --------------------------------------------------------------------------- #
# FD-vs-AD gradient check on the loss.
# --------------------------------------------------------------------------- #
def _grad_check(loss: Callable, vec: jnp.ndarray) -> float:
    ad = jax.grad(loss)(vec)
    h = 1e-6
    fd = jnp.array(
        [
            (loss(vec.at[i].add(h)) - loss(vec.at[i].add(-h))) / (2.0 * h)
            for i in range(vec.size)
        ]
    )
    denom = jnp.maximum(jnp.abs(ad), 1e-12)
    return float(jnp.max(jnp.abs(ad - fd) / denom))


def main() -> int:
    optax, numpyro = _load_ml()

    label, make_model, predict, where, read_free, truth_value, free_name, grid = (
        _build_target()
    )
    print("=" * 70)
    print(f"VALIDATION TARGET: {label}")
    print("=" * 70)

    # Inject a known truth; generate (near-)noiseless synthetic data.
    truth_model = make_model(truth_value)
    noise_sigma = 1e-3
    data = predict(truth_model, grid)

    # Start away from the truth; Exp keeps the param > 0 in unconstrained descent.
    init_model = make_model(0.5 * truth_value)
    param = Parameterization.from_where(init_model, where=where, transforms=(Exp(),))

    rec_optax = loss = None
    if optax is not None:
        rec_optax, loss, _vec_opt = _recover_optax(
            optax, param, init_model, predict, grid, data, read_free
        )

    rec_numpyro = None
    if numpyro is not None:
        rec_numpyro = _recover_numpyro(
            numpyro, param, init_model, predict, grid, data, noise_sigma, read_free
        )

    # Grad-check at the (non-converged) START point, where the gradient is well
    # away from zero -- at the optimum AD and FD are both ~0 and a relative
    # error is meaningless (0/0). This validates the differentiable bridge.
    # Build a standalone loss if optax (and hence its loss handle) is absent.
    if loss is None:

        @jax.jit
        def loss(vec):  # noqa: F811 - validation-only standalone loss
            model = param.from_vector(init_model, vec)
            return jnp.mean((predict(model, grid) - data) ** 2)

    vec_check = param.to_vector(init_model)
    max_rel_grad_err = _grad_check(loss, vec_check)

    # ----- results table ------------------------------------------------- #
    o_str = f"{rec_optax:>12.6f}" if rec_optax is not None else f"{'n/a':>12}"
    n_str = f"{rec_numpyro:>12.6f}" if rec_numpyro is not None else f"{'n/a':>12}"
    rec_primary = rec_optax if rec_optax is not None else rec_numpyro
    abs_err = abs(rec_primary - truth_value)
    rel_err = abs_err / abs(truth_value)
    print()
    header = (
        f"{'param':<6} {'true':>10} {'optax':>12} {'numpyro':>12} "
        f"{'abs_err':>11} {'rel_err':>11}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{free_name:<6} {truth_value:>10.5f} {o_str} "
        f"{n_str} {abs_err:>11.2e} {rel_err:>11.2e}"
    )
    print()
    print(
        f"FD-vs-AD gradient check (loss): max rel error = {max_rel_grad_err:.3e}"
    )
    print()

    # ----- pass/fail (a skipped backend does not fail the run) ----------- #
    grad_ok = max_rel_grad_err < 1e-5
    checks = [("grad check", grad_ok, f"max rel = {max_rel_grad_err:.2e}, tol 1e-5")]

    if rec_optax is not None:
        optax_ok = abs(rec_optax - truth_value) < 1e-3
        checks.append(
            (
                "optax recovery",
                optax_ok,
                f"|abs err| = {abs(rec_optax - truth_value):.2e}, tol 1e-3",
            )
        )
    else:
        print("optax recovery   : SKIP (optax not installed)")

    if rec_numpyro is not None:
        numpyro_err = abs(rec_numpyro - truth_value) / abs(truth_value)
        numpyro_ok = numpyro_err < 5e-2  # MCMC posterior mean, looser tolerance
        checks.append(
            ("numpyro recovery", numpyro_ok, f"rel err = {numpyro_err:.2e}, tol 5e-2")
        )
    else:
        print("numpyro recovery : SKIP (numpyro not installed)")

    for name, passed, detail in checks:
        print(f"{name:<16} : {'PASS' if passed else 'FAIL'} ({detail})")

    ok = all(passed for _, passed, _ in checks)
    print()
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
