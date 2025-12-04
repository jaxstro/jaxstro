# jaxstro

Core utilities for the **jaxstro** differentiable astrophysics ecosystem.

`jaxstro` is a small, shared library that provides:

- Physical **constants** and **unit systems** compatible with JAX
- Lightweight **numerical helpers** that work cleanly with `jax.jit`, `vmap`, and `grad`
- Common utilities and types for higher-level astrophysics codes

---

## Overview

`jaxstro` is the foundation layer for a family of JAX-native astrophysics packages:

| Package | Description |
|---------|-------------|
| **gravax** | N-body dynamics and star cluster evolution |
| **startrax** | Rapid single/binary stellar evolution (SSE/BSE fits) |
| **stellax** | 1D stellar evolution (MESA-like, inference-ready) |
| **nucleax** | Microphysics: EOS, nuclear networks, opacities |
| **nebulax** | Hydro/MHD and ISM physics |
| **radax** | Radiative transfer and ray tracing |
| **fluxax** | Synthetic observables and survey rendering |
| **progenax** | Initial conditions and population synthesis |

### Design Principles

1. **Infrastructure only** - No domain-specific models; just shared building blocks
2. **JAX-first** - Everything works with `jit`, `vmap`, and `grad`
3. **Minimal dependencies** - Only JAX and Python standard library
4. **One-way arrows** - Higher-level packages depend on jaxstro, not the reverse

---

## Installation

**Python 3.10+** and **JAX ≥ 0.4.28** required.

### From source (current)

```bash
git clone https://github.com/jaxstro/jaxstro.git
cd jaxstro
pip install -e ".[dev]"
```

### Future PyPI

```bash
pip install jaxstro
```

---

## Quick Start

### Enable float64 precision

Call this before importing other JAX modules:

```python
from jaxstro.jaxconfig import enable_high_precision
enable_high_precision()  # Sets jax_enable_x64=True
```

### Use constants and units

```python
import jax.numpy as jnp
from jaxstro import constants as C, units as U

# Solar mass and radius in CGS
M = 1.0 * C.MSUN_G     # 1.9884e33 g
R = 1.0 * C.RSUN_CM    # 6.957e10 cm

# Escape velocity
v_esc = jnp.sqrt(2.0 * C.G_CGS * M / R)

# Use a unit system
stellar = U.ASTRO_STELLAR  # (Msun, Rsun, Myr)
m_cgs, r_cgs, t_cgs = stellar.to_cgs(1.0, 1.0, 1.0)
```

### Basic numerics

```python
from jaxstro.numerics import stats, rootfinding

# Safe log (no -inf)
x = jnp.array([1e-30, 1.0, 10.0])
y = stats.safe_log(x)

# Root finding (works with jit, vmap, grad)
root = rootfinding.bisect(lambda x: x**2 - 2.0, 1.0, 2.0)  # √2
```

---

## API Reference

### `jaxstro.constants`

Physical constants in CGS units.

| Constant | Value | Description |
|----------|-------|-------------|
| `G_CGS` | 6.67430e-8 | Gravitational constant [cm³ g⁻¹ s⁻²] |
| `C_CGS` | 2.99792458e10 | Speed of light [cm/s] |
| `K_B` | 1.380649e-16 | Boltzmann constant [erg/K] |
| `SIGMA_SB` | 5.670374e-5 | Stefan-Boltzmann [erg cm⁻² s⁻¹ K⁻⁴] |
| `MSUN_G` | 1.9884e33 | Solar mass [g] |
| `RSUN_CM` | 6.957e10 | Solar radius [cm] |
| `LSUN_ERG_S` | 3.828e33 | Solar luminosity [erg/s] |
| `TEFF_SUN` | 5772.0 | Solar effective temperature [K] |
| `X_SUN`, `Y_SUN`, `Z_SUN` | 0.738, 0.249, 0.013 | Solar composition |
| `PC_CM` | 3.086e18 | Parsec [cm] |
| `AU_CM` | 1.496e13 | Astronomical unit [cm] |
| `MYR_S`, `YR_S` | 3.156e13, 3.156e7 | Time conversions [s] |

### `jaxstro.units`

Unit systems for different astrophysical regimes.

```python
from jaxstro import units as U

# Available systems
U.CGS              # (g, cm, s)
U.ASTRO_STELLAR    # (Msun, Rsun, Myr)
U.ASTRO_DYNAMICAL  # (Msun, pc, Myr)
U.ASTRO_PLANETARY  # (Msun, AU, yr)

# UnitSystem methods
us = U.ASTRO_STELLAR
m_cgs, r_cgs, t_cgs = us.to_cgs(mass, length, time)
m, r, t = us.from_cgs(m_cgs, r_cgs, t_cgs)
v_kms = us.velocity_scale_km_s
```

### `jaxstro.astrometry`

Astrometric constants and angular conversions.

```python
from jaxstro.astrometry import K_PROPER_MOTION, MAS_PER_RAD

# Convert proper motion to velocity
# mu [mas/yr] at distance d [kpc] -> v [km/s]
v_kms = mu_mas_yr * K_PROPER_MOTION * d_kpc  # K = 4.74047
```

### `jaxstro.jaxconfig`

JAX configuration helpers.

```python
from jaxstro.jaxconfig import enable_high_precision
enable_high_precision()  # jax_enable_x64=True, matmul_precision="highest"
```

### `jaxstro.numerics`

Numerical utilities organized by submodule.

#### `numerics.stats`

```python
from jaxstro.numerics import stats

stats.safe_log(x, eps=1e-30)           # log with floor
stats.safe_exp(x, max_exp=100.0)       # exp with ceiling
stats.safe_div(a, b, epsilon=1e-100)   # division with epsilon
stats.logsumexp(x, axis=None)          # stable log-sum-exp
stats.gaussian_logpdf(x, mu, sigma)    # normal log-PDF
stats.relative_error(x_new, x_old)     # |x_new - x_old| / |x_old|
stats.check_convergence(x_new, x_old, tol=1e-6)  # bool
```

#### `numerics.interpolation`

```python
from jaxstro.numerics import interpolation

# Linear interpolation
y_new = interpolation.interp1d(x, y, x_new)

# Pytree-compatible tabulated function
table = interpolation.TabulatedFunction1D(x, y)
y_eval = table(x_query)
```

#### `numerics.rootfinding`

All solvers use `lax.scan` with fixed iteration count for full `jit`/`vmap`/`grad` compatibility.

```python
from jaxstro.numerics import rootfinding

# Bisection (bracketed)
root = rootfinding.bisect(f, a, b, max_steps=50)

# Newton with automatic derivative
root = rootfinding.newton(f, x0, max_steps=30)

# Newton with user-provided derivative
root = rootfinding.newton_with_grad(f, df, x0, max_steps=30)
```

#### `numerics.integration`

```python
from jaxstro.numerics import integration

integration.trapz(y, x=None, axis=-1)           # trapezoidal rule
integration.cumulative_trapz(y, x=None)         # cumulative integral
integration.simpson(y, x=None, axis=-1)         # Simpson's rule
```

#### `numerics.checks`

Numerical validation helpers. Pure predicates (`is_*`, `all_*`) are JIT-compatible; `assert_*` functions are for eager code and tests.

```python
from jaxstro.numerics import checks

# JIT-compatible predicates
checks.all_finite(x)                    # no NaN/inf
checks.is_monotonic_increasing(x)       # strictly increasing
checks.is_monotonic_decreasing(x)       # strictly decreasing
checks.in_range(x, lo=0.0, hi=1.0)      # elementwise range check
checks.all_positive(x)                  # all > 0
checks.all_non_negative(x)              # all >= 0

# Eager assertions (for tests/debug)
checks.assert_all_finite(x, name="density")
checks.assert_monotonic(x, strict=True, decreasing=False)
checks.assert_in_range(x, lo=0.0, hi=1.0)
```

#### `numerics.compensated`

Compensated summation for reduced floating-point error.

```python
from jaxstro.numerics import compensated

compensated.neumaier_add(s, c, y)       # single update step
compensated.compensated_sum(*arrays)    # sum with compensation
compensated.compensated_dot(a, b)       # dot product
```

#### `numerics.linear_algebra`

```python
from jaxstro.numerics import linear_algebra as la

la.norm2(x, axis=None)                  # Euclidean norm
la.project_onto(a, b, axis=-1)          # vector projection
la.condition_number(A)                  # 2-norm condition number
```

#### `numerics.rng`

PRNG key management helpers.

```python
from jaxstro.numerics import rng

keys = rng.split_key(key, num=4)                 # split into N keys
keys = rng.split_tree(key, shape=(3, 4))         # reshape split
keys = rng.fold_in_indices(key, indices)         # fold indices
```

---

## Project Status

**Version 0.1.0** - First development release. The core API is stabilizing but may still evolve.

---

## Contributing

Contributions welcome, especially around:

- Constants and units coverage
- Useful JAX-friendly numerical helpers
- Tests, typing, and documentation

Please open an issue to discuss larger changes before submitting a pull request.

---

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
