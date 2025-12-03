# jaxstro

Core building blocks for the **jaxstro** differentiable astrophysics stack.

`jaxstro` is a small, shared library that sits at the bottom of the ecosystem and provides:

- Physical **constants** and **units** compatible with JAX.
- Lightweight **numerical helpers** that work cleanly with `jax.jit`, `vmap`, and `grad`.
- Common utilities and types that higher-level codes can import without creating dependency knots.

The intent is for this package to stay **narrow and stable**: no heavy physics models, no simulation logic—just the infrastructure that everything else can safely depend on.

---

## Project status

> **Early development / pre-release**

The layout and API are still being shaped. During this phase you should expect:

- Modules and function names to be renamed or moved.
- Constants and unit definitions to be reorganized.
- New helpers to appear as other jaxstro packages mature.

Once the interface settles, the project will move to semantic versioning (e.g. `0.1.0`, `0.2.0`, …) with documented changelogs.

---

## Installation

`jaxstro` targets **Python 3.10+** and recent versions of **JAX**.

### From source (recommended for now)

```bash
git clone https://github.com/jaxstro/jaxstro.git
cd jaxstro
pip install -e ".[dev]"
```

This installs:

- The `jaxstro` package itself.
- Development tools like `pytest`, `ruff`, and `mypy` via the `dev` extra.

Once the package is published on PyPI, a standard install will look like:

```bash
pip install jaxstro
```

---

## Basic usage

### Constants and units

The top-level namespace exposes physical constants and (eventually) a clear unit system:

```python
import jax.numpy as jnp
from jax import jit

from jaxstro import constants as C
from jaxstro import units as U

# Example: escape speed from a 1 M_sun, 1 R_sun star
M = 1.0 * C.M_sun
R = 1.0 * C.R_sun

v_esc = jnp.sqrt(2.0 * C.G * M / R)
```

Design goals for constants/units:

- Descriptive names (`M_sun`, `R_sun`, `L_sun`, `sigma_SB`, `k_B`, …).
- JAX-friendly dtypes so everything works under `jit` and `vmap`.
- A well-documented base unit convention, referenced from the `units` module.

### JAX-aware helpers

Over time `jaxstro` will also host small, reusable helpers, for example:

- Safer variants of common operations (e.g. `safe_log`, `safe_exp`).
- Shape- and dtype-aware math utilities.
- PRNG helpers that standardize key management across the ecosystem.

Example sketch (API subject to change):

```python
from jaxstro import math as jm

x = jnp.array([1e-30, 1.0, 10.0])
y = jm.safe_log(x)  # finite and well-behaved near zero
```

---

## Design principles

`jaxstro` is guided by a few simple rules:

1. **Infrastructure only**  
   This package should not contain domain-specific physics models (no stellar evolution, no N-body, no hydro). It focuses on shared infrastructure: constants, units, small utilities, and possibly base types.

2. **JAX-first**  
   Everything should be compatible with the standard JAX transforms:
   - `jit`
   - `vmap`
   - `grad` / `jvp` / `vjp`

3. **Minimal runtime dependencies**  
   The core library should only depend on JAX and the Python standard library so that higher-level codes don’t inherit unnecessary baggage.

4. **One-way dependency arrows**  
   Higher-level packages (e.g. `gravax`, `stellax`, `startrax`, `nucleax`, `nebulax`, `radax`, `fluxax`, `progenax`) depend on `jaxstro`, not the other way around. This keeps the ecosystem modular and easier to evolve.

---

## Place in the ecosystem

`jaxstro` is intended to serve as the common foundation for a broader differentiable astrophysics suite, including (planned):

- **`gravax`** – N-body dynamics and star cluster evolution in JAX.
- **`startrax`** – Rapid single and binary stellar evolution (Hurley-style fits in JAX).
- **`stellax`** – Full 1D stellar evolution, inference-ready.
- **`nucleax`** – Microphysics: nuclear reaction networks and equations of state.
- **`nebulax`** – ISM / gas dynamics.
- **`radax`** – Radiative transfer and ray tracing.
- **`fluxax`** – Synthetic observables and survey rendering.
- **`progenax`** – Initial conditions and population synthesis.

As those projects become public, this README will link to their docs, examples, and cross-package tutorials.

---

## Contributing

Contributions and feedback are very welcome, especially around:

- The design and coverage of constants and units.
- Useful JAX-friendly numerical helpers that would benefit multiple codes.
- Improvements to typing, tests, and tooling that make the core more robust.

Because the API is still in motion, please open an issue to discuss larger changes before investing in a big pull request. The goal is to keep the core clean and coherent so the rest of the ecosystem has a stable base.

---

## License

`jaxstro` is distributed under the **BSD 3-Clause** license. See `LICENSE` for details.

