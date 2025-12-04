# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Overview

jaxstro is the shared foundation library for a differentiable astrophysics ecosystem built on JAX. Provides physical constants, unit systems, and numerical utilities for higher-level packages (gravax, startrax, stellax, nebulax, etc.).

**Design principles:**
- Infrastructure only (no domain-specific physics)
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

```
src/jaxstro/
├── __init__.py          # Exports: constants, units, astrometry, numerics
├── jaxconfig.py         # enable_high_precision() for float64
├── constants.py         # CGS physical constants (CODATA 2018, IAU 2015)
├── units.py             # UnitSystem dataclass + predefined systems
├── astrometry.py        # Astrometric constants (K_PROPER_MOTION, etc.)
└── numerics/
    ├── types.py         # Type aliases (Array, ScalarFn)
    ├── stats.py         # safe_log/exp/div, logsumexp, convergence
    ├── interpolation.py # interp1d, TabulatedFunction1D (pytree)
    ├── rootfinding.py   # bisect, newton, newton_with_grad (lax.scan)
    ├── integration.py   # trapz, cumulative_trapz, simpson
    ├── checks.py        # Validation: all_finite, is_monotonic, in_range
    ├── compensated.py   # Neumaier compensated summation
    ├── linear_algebra.py# norm2, project_onto, condition_number
    └── rng.py           # PRNG key helpers
```

## Units Convention

**Always use CGS** (cm, g, s, erg) as base. Available systems:
- `CGS` - base (g, cm, s)
- `ASTRO_STELLAR` - stellar evolution (Msun, Rsun, Myr)
- `ASTRO_DYNAMICAL` - star clusters (Msun, pc, Myr)
- `ASTRO_PLANETARY` - solar system (Msun, AU, yr)

## Key Patterns

```python
# Enable float64 (call before any JAX arrays)
from jaxstro.jaxconfig import enable_high_precision
enable_high_precision()

# Use constants and units
from jaxstro import constants as C, units as U
us = U.ASTRO_DYNAMICAL
m, r, t = us.from_cgs(mass_g, length_cm, time_s)
```

## Adding to numerics/

- Keep functions small and domain-agnostic
- Ensure `jit`, `vmap`, `grad` compatibility
- Full solvers belong in diffrax/optimistix/lineax
