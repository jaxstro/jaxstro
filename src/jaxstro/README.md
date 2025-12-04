# jaxstro

Core JAX utilities and shared infrastructure for a differentiable astrophysics ecosystem.

`jaxstro` is the small, opinionated base library that underpins a family of JAX-native astrophysics codes:

* **gravax** – N-body and collisional dynamics
* **startrax** – rapid single/binary stellar evolution fits (SSE/BSE)
* **stellax** – 1D stellar evolution (MESA-like, but JAX-native)
* **nebulax** – hydro/MHD and ISM physics
* **fluxax** – observables, rendering, and survey likelihoods
* **nucleax** – microphysics (EOS, nuclear networks, opacities)

The goal of `jaxstro` is to centralize the low-level numerical and physical infrastructure those codes share:

* shared physical constants, unit systems, and astrometric conversions
* consistent JAX configuration for 64-bit, high-precision work
* small, reusable numerical helpers (interpolation, rootfinding, integration, RNG)
* lightweight validation checks for catching numerical pathologies

Everything here is **JAX-first** and deliberately domain-agnostic.

---

## Features

### Shared configuration and types

* `jaxstro.jaxconfig.enable_high_precision()` – opt-in helper to configure JAX for:

  * `jax_enable_x64 = True` (float64 everywhere)
  * `jax_default_matmul_precision = "highest"`
* `jaxstro.types` – central type aliases used across the ecosystem:

  * `Array` – alias for `jax.numpy.ndarray`
  * `ScalarFn` – scalar → scalar function type, `Callable[[Array], Array]`
  * `PyTree` – generic PyTree label for JAX-compatible trees

### Physical constants and units

* `jaxstro.constants` – CGS and solar constants (G, Msun, Rsun, Lsun, pc, AU, etc.) with explicit provenance.

* `jaxstro.units` – unit systems for different astrophysical regimes:

  * **Stellar dynamics**: (Msun, pc, Myr), G in these code units
  * **Planetary/binary dynamics**: (Msun, AU, yr)

* Context-based unit management:

  ```python
  from jaxstro.units import use_unit_system, get_G

  with use_unit_system("stellar"):
      G = get_G()  # G in Msun–pc–Myr units
      # ... run cluster dynamics ...
  ```

* Helper conversions for velocities and astrometry (e.g., mas/yr ↔ km/s at a given distance).

### Numerical utilities (`jaxstro.numerics`)

The `numerics` subpackage collects small, JAX-native helpers that show up everywhere but don’t belong to any one domain.

#### Compensated summation

`jaxstro.numerics.compensated` implements Neumaier-style compensated summation:

* `neumaier_add(s, c, y)` – single compensated update step
* `compensated_sum(*terms)` – sum a small number of arrays with reduced cancellation error
* `compensated_vector_sum(vectors)` – sum an (N, D) stack of vectors along axis 0
* `compensated_dot(a, b)` – dot product with compensation

Useful for long integrations, energy accounting, and any place where naïve summation loses small corrections.

#### Stats helpers

`jaxstro.numerics.stats` provides low-level numerical pieces for log-likelihoods and losses:

* `safe_log(x, eps)` – `log` with floor to avoid `-inf`
* `logsumexp(x, axis, keepdims)` – stable log-sum-exp wrapper
* `gaussian_logpdf(x, mu, sigma)` – scalar normal log-PDF
* `gaussian_loglikelihood(data, mu, sigma, axis)` – independent normal log-likelihood
* `stable_log1p`, `stable_expm1` – thin wrappers for small-argument regimes

These are meant as building blocks; full probabilistic modeling still lives in dedicated inference libraries.

#### Interpolation

`jaxstro.numerics.interpolation` offers simple, differentiable 1D interpolation:

* `interp1d(x, y, x_new, axis=-1, left=None, right=None, extrapolate=False)` – linear interpolation on a monotone grid
* `TabulatedFunction1D(x, y)` – JAX-pytree-friendly wrapper for tabulated functions

Designed for stellar fits, opacities, cooling tables, photometric grids, etc., without pulling in a heavy interpolation stack.

#### Rootfinding

`jaxstro.numerics.rootfinding` provides scalar root solvers:

* `bisect(f, a, b, maxiter=50, tol=1e-8)` – bracketed bisection for monotone functions
* `newton_1d(f, df, x0, maxiter=30, tol=1e-10)` – simple 1D Newton iteration

These are intentionally small. For larger systems or more sophisticated solves, the ecosystem relies on libraries like `optimistix` and `lineax`.

#### Integration

`jaxstro.numerics.integration` collects simple 1D quadrature:

* `trapz(y, x=None, axis=-1)` – trapezoidal rule along a given axis
* `cumulative_trapz(y, x=None, axis=-1)` – cumulative integral (first value is zero)
* `simpson(y, x=None, axis=-1)` – Simpson’s rule (odd number of points)

Good for masses, luminosities, cooling integrals, and cumulative quantities over gridded data.

#### Linear algebra helpers

`jaxstro.numerics.linear_algebra` adds a few sharp tools on top of `jax.numpy.linalg`:

* `norm2(x, axis=None, keepdims=False)` – Euclidean norm
* `project_onto(a, b, axis=-1, eps=0.0)` – projection of `a` onto `b`
* `condition_number(A)` – 2-norm condition number via singular values

For serious linear solves, the ecosystem leans on `lineax` and JAX’s built-in linalg.

#### RNG helpers

`jaxstro.numerics.rng` standardizes common JAX PRNG patterns:

* `split_key(key, num)` – split a base key into `num` subkeys
* `split_tree(key, shape)` – reshape a flat split into an array of keys with the desired shape
* `fold_in_indices(key, indices)` – derive keys by folding indices into a base key

This lets different codes share consistent key-handling patterns without each re-inventing them.

### Numerical checks

`jaxstro.checks` offers small validation helpers for catching numerical pathologies early:

* `is_finite(x)` / `all_finite(x)` – NaN/inf checks
* `is_monotonic(x, strict=True)` – monotone-increasing check for 1D grids
* `in_range(x, lo=None, hi=None, inclusive=True)` – value range masks

These can be used inside library code, tests, or debug builds to guard against silently-broken tables and states.

---

## Installation

Right now `jaxstro` is designed as a shared internal library for the jaxstro ecosystem.

Typical development install from source:

```bash
# Clone the repository
git clone https://github.com/your-org/jaxstro.git
cd jaxstro

# Create and activate an environment (example using uv or venv)
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

In the future, the package may be published on PyPI; when that happens, installation would be as simple as:

```bash
pip install jaxstro
```

---

## Quick start

### Configure JAX for high precision

In a top-level package (e.g., `gravax/__init__.py`):

```python
from jaxstro.jaxconfig import enable_high_precision as _enable_jax_hp

_enable_jax_hp()
del _enable_jax_hp  # avoid leaking into the public API

# Now it is safe to import the rest of your JAX-heavy modules
from . import dynamics  # noqa: E402
```

### Use a unit system

```python
from jaxstro.units import use_unit_system, get_G

with use_unit_system("stellar"):
    G = get_G()  # Msun–pc–Myr units
    # ... run a cluster integration with G in code units ...
```

### Compensated sum in a dynamics code

```python
from jaxstro.numerics.compensated import compensated_sum

# forces_from_near, forces_from_far, external_forces all have shape (N, 3)
forces_total = compensated_sum(forces_from_near, forces_from_far, external_forces)
```

### Interpolation from a tabulated fit

```python
from jaxstro.numerics.interpolation import TabulatedFunction1D

logm = jnp.linspace(-1.0, 2.0, 64)       # log10(M / Msun)
logL = some_fit(logm)                    # tabulated log10(L / Lsun)
L_table = TabulatedFunction1D(logm, logL)

logL_eval = L_table(jnp.array([0.0, 1.0]))
```

---

## Project philosophy

`jaxstro` is deliberately small and conservative:

* **JAX-native first** – everything is compatible with `jit`, `vmap`, and JAX pytrees.
* **Domain-agnostic** – no references to “stars”, “clusters”, or “hydro” in this package.
* **Sharp but minimal tools** – when a function starts to look like a full solver, it belongs in domain codes or external libraries (e.g., diffrax, optimistix, lineax, numpyro).
* **Shared infrastructure, not a framework** – `jaxstro` is there to reduce duplication and make the rest of the ecosystem cleaner, not to dictate how every code must be structured.

If you find yourself writing the same low-level numerical helper in multiple jaxstro ecosystem packages, it probably belongs here instead.

---