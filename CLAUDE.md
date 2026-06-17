# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Overview

jaxstro is the shared foundation library for a differentiable astrophysics ecosystem built on JAX. Provides physical constants, unit systems, coordinate transforms, spatial algorithms, and numerical utilities for higher-level packages (gravax, startrax, stellax, nebulax, etc.).

**Design principles:**

- Infrastructure only (no domain-specific physics simulations)
- JAX-first (compatible with `jit`, `vmap`, `grad`)
- Minimal dependencies (JAX + stdlib only)

## Commands

```bash
conda activate astro
pip install -e ".[dev]"
pytest                    # Run tests
ruff check src/ && ruff format src/  # Lint/format
mypy                      # Type check
```

## Architecture

```text
src/jaxstro/
├── __init__.py          # Exports: constants, units, astrometry, numerics, coords
├── jaxconfig.py         # enable_high_precision() for float64
├── constants.py         # CGS physical constants (CODATA 2018, IAU 2015)
├── units.py             # UnitSystem dataclass + predefined systems + G property
├── astrometry.py        # Astrometric constants (K_PROPER_MOTION, etc.)
├── coords.py            # Coordinate transforms (sky_tangent, galactic, spherical)
├── numerics/
│   ├── types.py         # Type aliases (Array, ScalarFn)
│   ├── stats.py         # safe_log/exp/div, logsumexp, convergence
│   ├── interpolation.py # interp1d, TabulatedFunction1D (pytree)
│   ├── rootfinding.py   # bisect, newton, newton_with_grad (lax.scan)
│   ├── integration.py   # trapz, cumulative_trapz, simpson
│   ├── checks.py        # Validation: all_finite, is_monotonic, in_range
│   ├── compensated.py   # Neumaier compensated summation
│   ├── linear_algebra.py# norm2, project_onto, condition_number
│   └── rng.py           # PRNG key helpers
└── spatial/
    ├── morton.py        # Morton (Z-order) encoding/decoding, wyhash32
    ├── grid.py          # assign_particles_to_bins, fill_bins (reservoir)
    └── neighbor.py      # approx_knn_candidates, stencil-based gathering
```

## Units Convention

**Always use CGS** (cm, g, s, erg) as base. Available systems:

- `CGS` - base (g, cm, s)
- `ASTRO_STELLAR` / `solar` - stellar evolution (Msun, Rsun, Myr)
- `ASTRO_DYNAMICAL` / `stellar` - star clusters (Msun, pc, Myr)
- `ASTRO_PLANETARY` / `binary` - solar system (Msun, AU, yr)

Each UnitSystem has a `.G` property for the gravitational constant in that system.

## Ecosystem Units Policy (Defaults)

Downstream packages must define a package-level `DEFAULT_UNITS` constant.
Core APIs should require explicit units or explicit `G`, or accept objects that
carry units. Convenience wrappers may accept `units=None` and resolve to
`DEFAULT_UNITS`. Do not use global context managers in core code.

## Key Patterns

```python
# Enable float64 (call before any JAX arrays)
# This is the standard approach across the jaxstro ecosystem
from jaxstro.jaxconfig import enable_high_precision
enable_high_precision()  # Sets jax_enable_x64=True, matmul_precision="highest"

# Use constants and units
from jaxstro import constants as C, units as U
us = U.ASTRO_DYNAMICAL
m, r, t = us.from_cgs(mass_g, length_cm, time_s)
G = us.G  # Gravitational constant in this unit system

# Coordinate transforms
from jaxstro.coords import sky_tangent, galactic_to_equatorial
ra_dec = sky_tangent(positions_pc, distance_pc=1000.0)

# Spatial binning (for neighbor lists, density estimates)
from jaxstro.spatial import assign_particles_to_bins, fill_bins, approx_knn_candidates
bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=16)
```

## Module Summary

| Module | Purpose |
|--------|---------|
| `constants` | Physical constants in CGS (G, c, k_B, Msun, Rsun, etc.) |
| `units` | UnitSystem dataclass with conversions and `.G` property |
| `astrometry` | Astrometric constants (K_PROPER_MOTION, MAS_PER_RAD) |
| `coords` | Coordinate transforms (sky-tangent, galactic, spherical, parallax) |
| `numerics` | Numerical utilities (stats, interpolation, rootfinding, etc.) |
| `spatial` | Morton encoding, grid binning, neighbor candidate gathering |

## Adding New Code

- Keep functions small and domain-agnostic
- Ensure `jit`, `vmap`, `grad` compatibility
- Full solvers belong in diffrax/optimistix/lineax
- Spatial algorithms go in `spatial/`
- Coordinate transforms go in `coords.py`


## Brain hub — this repo is a spoke of ~/brain (read-only from here)

- **Never edit `~/brain` from this session** — not hat homes, ADRs, configs, knowledge, or `_generated/`.
- **One write path home — the inbox, via capture** (works from any directory):
  `brain "what happened — short, factual"`
- **Cross-cutting insight** (something here also relevant to another project/paper)?
  `brain "xref: <insight> — touches <other project / paper>"` → becomes a brain concept that resurfaces here via `/brain-pack` (ADR-0019).
- **Full protocol + conventions:** read `~/brain/AGENTS.md` and `~/brain/guide/` before cross-session work
  (pull-only hub; spec → session → log handoffs, ADR-0018; modern mystmd if this is a MyST site).
- **Starting focused work here?** Pull a context pack from the hub: `/brain-pack jaxstro`.

<!-- brain-handshake: keep in sync with ~/brain/guide/how-to/set-up-a-project.md#spoke-stanza -->

<!-- brain-status-convention -->
## Brain status updates
When you make notable progress, hit a blocker, or set the next action, update this repo's `STATUS.md` (`next:` / `blocker:` / `due:` lines) — the brain pulls it into the portfolio dashboard + standup via `federate.py` (see `~/brain/work/meta/status-convention.md`). Brain stays pull-only: never hand-edit `~/brain`; capture events with `brain "…"`.
