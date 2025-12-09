# jaxstro

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.28+-green.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-lightgrey.svg)](LICENSE)

**Core utilities for differentiable astrophysics in JAX.**

> 🔬 Compute gradients through your simulations for Bayesian inference,
> parameter optimization, and machine learning applications.

---

## ✨ Features

- 🌟 **Physical constants** — CODATA 2018 values in CGS units
- 📐 **Unit systems** — Seamless conversion between stellar, dynamical, and planetary scales
- 🌍 **Coordinate transforms** — Sky projections, galactic/equatorial, parallax, proper motions
- 🔢 **Numerical helpers** — Root-finding, interpolation, compensated summation
- 📦 **Spatial algorithms** — Morton encoding, grid binning, neighbor queries

Everything works with `jax.jit`, `jax.vmap`, and `jax.grad`.

---

## 🏗️ Ecosystem

`jaxstro` is the foundation layer for a family of JAX-native astrophysics packages:

| Package | Description | Status |
|---------|-------------|--------|
| [**gravax**](https://github.com/jaxstro/gravax) | $N$-body dynamics and star cluster evolution | 🚧 Active |
| **progenax** | Initial conditions and population synthesis | 🚧 Active |
| **fluxax** | Synthetic observables and survey rendering | 🚧 Active |
| **stellax** | 1D stellar structure (MESA-like) | 📋 Planned |
| **startrax** | Rapid stellar evolution (SSE/BSE fits) | 📋 Planned |

### Design Principles

1. **Infrastructure only** — No domain-specific physics; just shared building blocks
2. **JAX-first** — Full compatibility with `jit`, `vmap`, and `grad`
3. **Minimal dependencies** — Only JAX and `jaxtyping`
4. **One-way arrows** — Higher-level packages depend on jaxstro, not the reverse

---

## 📦 Installation

**Requirements:** Python 3.10+ and JAX $\geq$ 0.4.28

### From source

```bash
git clone https://github.com/jaxstro/jaxstro.git
cd jaxstro

# With uv (recommended, 10-100× faster)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### PyPI (coming soon)

```bash
pip install jaxstro
```

---

## 🚀 Quick Start

### Enable float64 precision

Call this **before** importing other JAX modules. This is the standard approach across the jaxstro ecosystem - high precision is configured before any JAX arrays are created:

```python
from jaxstro.jaxconfig import enable_high_precision
enable_high_precision()  # Sets jax_enable_x64=True, matmul_precision="highest"
```

**Note:** Higher-level packages (gravax, progenax, etc.) call this automatically at import time, so you typically don't need to call it yourself.

### Constants and units

```python
import jax.numpy as jnp
from jaxstro import constants as C, units as U

# Solar mass and radius in CGS
M = 1.0 * C.MSUN_G     # 1.9884×10³³ g
R = 1.0 * C.RSUN_CM    # 6.957×10¹⁰ cm

# Escape velocity: v_esc = √(2GM/R)
v_esc = jnp.sqrt(2.0 * C.G_CGS * M / R)

# Get G in any unit system
G = U.ASTRO_DYNAMICAL.G  # ≈ 0.00450 pc³ M⊙⁻¹ Myr⁻²
```

### Coordinate transforms

```python
from jaxstro.coords import sky_tangent, galactic_to_equatorial, compute_parallax

# Project 3D positions to (RA, Dec)
positions_pc = jnp.array([[1.0, 0.5, -0.2], [0.0, 1.0, 0.3]])
ra_dec = sky_tangent(positions_pc, distance_pc=1000.0, ra_center_deg=180.0)

# Galactic → Equatorial
l, b = 45.0, 30.0  # degrees
ra, dec = galactic_to_equatorial(l, b)

# Distance → Parallax
parallax_mas = compute_parallax(distance_pc=100.0)  # → 10 mas
```

---

## 📐 Unit Systems

Different astrophysical regimes have natural scales. Choose the right unit system to keep $G$ and other quantities $\mathcal{O}(1)$:

| System | Mass | Length | Time | $G$ | Best for |
|--------|------|--------|------|-----|----------|
| `CGS` | g | cm | s | $6.67 \times 10^{-8}$ | Microphysics, EOS |
| `ASTRO_STELLAR` | $M_\odot$ | $R_\odot$ | Myr | 2942.2 | Stellar interiors, binary evolution |
| `ASTRO_DYNAMICAL` | $M_\odot$ | pc | Myr | 0.00450 | Star clusters, galaxies, $N$-body |
| `ASTRO_PLANETARY` | $M_\odot$ | AU | yr | $39.48 \approx 4\pi^2$ | Planetary systems, Kepler's laws |

### Usage

```python
from jaxstro import units as U

# Pick a unit system
us = U.ASTRO_DYNAMICAL  # (M⊙, pc, Myr)

# Convert to/from CGS
m_cgs, r_cgs, t_cgs = us.to_cgs(mass=1.0, length=1.0, time=1.0)
m, r, t = us.from_cgs(m_cgs, r_cgs, t_cgs)

# Access G in this system
G = us.G  # 0.00449... pc³ M⊙⁻¹ Myr⁻²

# Velocity scale
v_kms = us.velocity_scale_km_s  # ~0.978 km/s per (pc/Myr)
```

**Why this matters:** In `ASTRO_PLANETARY` units, Kepler's third law simplifies to $P^2 = a^3$ (for $M = 1\,M_\odot$) because $G \approx 4\pi^2$.

---

## 🔢 Numerical Utilities

### Safe math (no NaN/inf surprises)

```python
from jaxstro.numerics import stats

stats.safe_log(x, eps=1e-30)      # log with floor
stats.safe_exp(x, max_exp=100.0)  # exp with ceiling
stats.safe_div(a, b)              # division with ε
```

### Root-finding (fully differentiable)

All solvers use `lax.scan` with fixed iterations — compatible with `jit`, `vmap`, `grad`:

```python
from jaxstro.numerics import rootfinding

# Find √2 via bisection
root = rootfinding.bisect(lambda x: x**2 - 2.0, a=1.0, b=2.0)

# Newton's method (auto-differentiated)
root = rootfinding.newton(lambda x: x**2 - 2.0, x0=1.5)
```

### Compensated summation

Reduce floating-point error when summing many terms:

```python
from jaxstro.numerics.compensated import compensated_sum_array

# Standard sum loses precision
terms = jnp.array([1e16, 1.0, -1e16, 1.0])
jnp.sum(terms)  # → 0.0 (wrong!)

# Compensated sum preserves it
compensated_sum_array(terms)  # → 2.0 (correct)
```

---

## 📦 Spatial Algorithms

Efficient spatial data structures for particle simulations:

```python
import jax
import jax.numpy as jnp
from jaxstro.spatial import assign_particles_to_bins, fill_bins, approx_knn_candidates

# Random particle positions in [-2, 2]³
key = jax.random.PRNGKey(42)
pos = jax.random.uniform(key, (1000, 3)) * 4.0 - 2.0

# Assign to Morton-ordered spatial bins
bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=16)

# Fill bin arrays (with overflow handling)
particle_ids = jnp.arange(1000, dtype=jnp.int32)
bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=16**3, Bcap=32)

# Get approximate neighbor candidates
pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)
cand_idx, cand_mask = approx_knn_candidates(
    pos_sentinel, bin_members, bin_mask, bin_of,
    Nbins_per_dim=16, K_target=32
)
```

---

## 📚 API Reference

<details>
<summary><b>jaxstro.constants</b> — Physical constants (CGS)</summary>

| Constant | Value | Description |
|----------|-------|-------------|
| `G_CGS` | $6.674 \times 10^{-8}$ | Gravitational constant [cm³ g⁻¹ s⁻²] |
| `C_CGS` | $2.998 \times 10^{10}$ | Speed of light [cm/s] |
| `K_B` | $1.381 \times 10^{-16}$ | Boltzmann constant [erg/K] |
| `SIGMA_SB` | $5.670 \times 10^{-5}$ | Stefan-Boltzmann [erg cm⁻² s⁻¹ K⁻⁴] |
| `MSUN_G` | $1.988 \times 10^{33}$ | Solar mass [g] |
| `RSUN_CM` | $6.957 \times 10^{10}$ | Solar radius [cm] |
| `LSUN_ERG_S` | $3.828 \times 10^{33}$ | Solar luminosity [erg/s] |
| `PC_CM` | $3.086 \times 10^{18}$ | Parsec [cm] |
| `AU_CM` | $1.496 \times 10^{13}$ | Astronomical unit [cm] |

</details>

<details>
<summary><b>jaxstro.coords</b> — Coordinate transforms</summary>

```python
from jaxstro.coords import (
    sky_tangent,             # 3D positions → (RA, Dec)
    galactic_to_equatorial,  # (l, b) → (RA, Dec)
    equatorial_to_galactic,  # (RA, Dec) → (l, b)
    cartesian_to_spherical,  # (x, y, z) → (r, θ, φ)
    spherical_to_cartesian,  # (r, θ, φ) → (x, y, z)
    compute_parallax,        # distance [pc] → parallax [mas]
    compute_proper_motions,  # 3D velocity → (μ_α*, μ_δ) [mas/yr]
)
```

</details>

<details>
<summary><b>jaxstro.spatial</b> — Spatial algorithms</summary>

```python
from jaxstro.spatial import (
    # Morton (Z-order) encoding
    morton_encode_3d,       # 3D coords → 1D Morton code
    morton_decode_3d,       # Morton code → (x, y, z)

    # Grid binning
    assign_particles_to_bins,  # positions → bin IDs
    fill_bins,                 # bin arrays with overflow handling

    # Neighbor queries
    approx_knn_candidates,     # high-level API
)
```

</details>

<details>
<summary><b>jaxstro.numerics</b> — Numerical utilities</summary>

- `stats` — `safe_log`, `safe_exp`, `safe_div`, `logsumexp`
- `rootfinding` — `bisect`, `newton`, `newton_with_grad`
- `interpolation` — `interp1d`, `TabulatedFunction1D`
- `integration` — `trapz`, `cumulative_trapz`, `simpson`
- `checks` — `all_finite`, `is_monotonic`, `in_range`
- `compensated` — Neumaier summation for reduced FP error
- `linear_algebra` — `norm2`, `project_onto`, `condition_number`

</details>

---

## 👩‍🔬 Author

**Anna Rosen** — Lead developer

- 📧 [alrosen@sdsu.edu](mailto:alrosen@sdsu.edu)
- 🏛️ San Diego State University

---

## 📊 Project Status

**Version 0.1.0** — Development release. Core API is stabilizing but may evolve.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Constants and units coverage
- Coordinate transform utilities
- Spatial algorithm optimizations
- Tests and documentation

Please open an issue to discuss larger changes before submitting a PR.

---

## 📄 License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
