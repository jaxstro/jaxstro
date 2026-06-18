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

It also reports a **finite-difference vs autodiff** gradient check on the optax
loss (max relative error), confirming the bridge is differentiable end-to-end.

VALIDATION TARGET
-----------------
Primary: ``progenax.PlummerProfile`` -- an ``eqx.Module`` with a scalar
half-mass radius ``r_h > 0``. We fit ``r_h`` to a synthetic radial density
profile generated from a known truth. (The scale radius ``a`` is a deterministic
function of ``r_h``, so the loss rebuilds the profile from the free ``r_h``.)

Fallback: if progenax (or a suitable model) cannot be imported, a self-contained
toy ``eqx.Module`` is used instead, and the script prints which path ran.

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
def _require_ml():
    """Import optax + numpyro or print an actionable message and exit."""
    try:
        import numpyro  # noqa: F401
        import optax  # noqa: F401

        return optax, numpyro
    except ImportError as exc:  # pragma: no cover - exercised only without [ml]
        print(
            "ERROR: validation requires the [ml] extra (optax + numpyro).\n"
            f"  missing: {exc}\n"
            "  install/run with: env -u VIRTUAL_ENV uv run --no-sync --extra ml "
            "python validation/validate_params.py"
        )
        sys.exit(2)


# --------------------------------------------------------------------------- #
# Model selection: real progenax model, else a toy fallback.
# --------------------------------------------------------------------------- #
def _build_target():
    """Return ``(label, make_model, predict, truth_value, free_name, grid)``.

    ``make_model(value)`` constructs the model from the scalar free parameter.
    ``predict(model, grid)`` returns the observable vector to fit.
    """
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

        def make_model(r_h):
            # PlummerProfile.__init__ recomputes the scale radius ``a`` from
            # ``r_h``, so rebuilding from the free scalar keeps ``a`` consistent.
            return Profile(r_h=r_h)

        def predict(model, x):
            # Enclosed-mass fraction M(<r)/M is a smooth, monotone, well-scaled
            # observable in [0, 1] -- a clean target for fitting r_h.
            return model.enclosed_mass_fraction(x)

        label = "progenax.PlummerProfile (free: r_h > 0)"
        return label, make_model, predict, 1.6, "r_h", grid

    except Exception as exc:  # pragma: no cover - only without progenax
        reason = f"{type(exc).__name__}: {exc}"

        class _ToyPlummer(eqx.Module):
            r_h: jax.Array

            def enclosed(self, r):
                a = self.r_h * jnp.sqrt((1.0 - 0.5 ** (2 / 3)) / 0.5 ** (2 / 3))
                return r**3 / (r**2 + a**2) ** 1.5

        def make_model(r_h):
            return _ToyPlummer(r_h=jnp.asarray(r_h))

        def predict(model, x):
            return model.enclosed(x)

        label = f"toy fallback (progenax unavailable: {reason})"
        return label, make_model, predict, 1.6, "r_h", grid


# --------------------------------------------------------------------------- #
# Recovery (a): optax gradient descent.
# --------------------------------------------------------------------------- #
def _recover_optax(optax, param, init_model, make_model, predict, grid, data):
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
    return float(jnp.ravel(recovered.r_h)[0]), loss, vec


# --------------------------------------------------------------------------- #
# Recovery (b): tiny numpyro NUTS chain over the unconstrained vector.
# --------------------------------------------------------------------------- #
def _recover_numpyro(
    numpyro, param, init_model, make_model, predict, grid, data, noise_sigma
):
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model_fn():
        # Sample the unconstrained free vector with a broad standard-normal
        # prior on R, then add the change-of-variables Jacobian so the implied
        # prior on the *physical* parameter is consistent.
        vec = numpyro.sample(
            "vec", dist.Normal(jnp.zeros(1), 5.0).to_event(1)
        )
        numpyro.factor("ldj", param.log_det_jacobian(vec))
        model = param.from_vector(init_model, vec)
        mu = predict(model, grid)
        numpyro.sample("obs", dist.Normal(mu, noise_sigma).to_event(1), obs=data)

    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=400, num_samples=600, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()["vec"]  # (num_samples, 1), unconstrained
    # Map each unconstrained draw forward to physical r_h, take posterior mean.
    r_h_draws = jax.vmap(lambda v: param.from_vector(init_model, v).r_h)(samples)
    return float(jnp.mean(jnp.ravel(r_h_draws)))


# --------------------------------------------------------------------------- #
# FD-vs-AD gradient check on the optax loss.
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
    optax, numpyro = _require_ml()

    label, make_model, predict, truth_value, free_name, grid = _build_target()
    print("=" * 70)
    print(f"VALIDATION TARGET: {label}")
    print("=" * 70)

    # Inject a known truth; generate (near-)noiseless synthetic data.
    truth_model = make_model(truth_value)
    noise_sigma = 1e-3
    data = predict(truth_model, grid)

    # Start away from the truth; Exp keeps r_h > 0 in unconstrained descent.
    init_model = make_model(0.7)
    param = Parameterization.from_where(
        init_model, where=lambda m: (m.r_h,), transforms=(Exp(),)
    )

    rec_optax, loss, _vec_opt = _recover_optax(
        optax, param, init_model, make_model, predict, grid, data
    )
    rec_numpyro = _recover_numpyro(
        numpyro, param, init_model, make_model, predict, grid, data, noise_sigma
    )

    # Grad-check at the (non-converged) START point, where the gradient is well
    # away from zero -- at the optimum AD and FD are both ~0 and a relative
    # error is meaningless (0/0). This validates the differentiable bridge.
    vec_check = param.to_vector(init_model)
    max_rel_grad_err = _grad_check(loss, vec_check)

    # ----- results table ------------------------------------------------- #
    abs_err = abs(rec_optax - truth_value)
    rel_err = abs_err / abs(truth_value)
    print()
    header = (
        f"{'param':<6} {'true':>10} {'optax':>12} {'numpyro':>12} "
        f"{'abs_err':>11} {'rel_err':>11}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{free_name:<6} {truth_value:>10.5f} {rec_optax:>12.6f} "
        f"{rec_numpyro:>12.6f} {abs_err:>11.2e} {rel_err:>11.2e}"
    )
    print()
    print(
        f"FD-vs-AD gradient check (optax loss): max rel error = "
        f"{max_rel_grad_err:.3e}"
    )
    print()

    # ----- pass/fail ----------------------------------------------------- #
    optax_ok = abs_err < 1e-3
    numpyro_err = abs(rec_numpyro - truth_value) / abs(truth_value)
    numpyro_ok = numpyro_err < 5e-2  # MCMC posterior mean, looser tolerance
    grad_ok = max_rel_grad_err < 1e-5

    print(
        f"optax recovery   : {'PASS' if optax_ok else 'FAIL'} "
        f"(|abs err| = {abs_err:.2e}, tol 1e-3)"
    )
    print(
        f"numpyro recovery : {'PASS' if numpyro_ok else 'FAIL'} "
        f"(rel err = {numpyro_err:.2e}, tol 5e-2)"
    )
    print(
        f"grad check       : {'PASS' if grad_ok else 'FAIL'} "
        f"(max rel = {max_rel_grad_err:.2e}, tol 1e-5)"
    )

    ok = optax_ok and numpyro_ok and grad_ok
    print()
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
