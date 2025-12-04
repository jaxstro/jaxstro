# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

jaxstro is the shared foundation library for a differentiable astrophysics ecosystem built on JAX. It provides physical constants, unit systems, and numerical utilities that higher-level packages (gravax, startrax, stellax, nebulax, nucleax, fluxax, radax, progenax) depend on.

**Design principles:**
- Infrastructure only (no domain-specific physics models)
- JAX-first (compatible with `jit`, `vmap`, `grad`)
- Minimal runtime dependencies (only JAX + stdlib)
- One-way dependency arrows (other packages depend on jaxstro, never the reverse)

## Development Commands

```bash
# Environment setup
conda activate astro

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test
pytest tests/test_module.py::test_function

# Linting and formatting
ruff check src/
ruff format src/

# Type checking
mypy
```

## Architecture

```
src/jaxstro/
├── __init__.py          # Top-level exports: constants, units, astrometry
├── jaxconfig.py         # enable_high_precision() for float64/matmul settings
├── constants.py         # CGS physical constants (G, Msun, Rsun, Lsun, pc, AU, etc.)
├── units.py             # UnitSystem dataclass + predefined systems (CGS, ASTRO_STELLAR, etc.)
├── astrometry.py        # Astrometric coordinate/velocity conversions
└── numerics/            # JAX-native numerical utilities
    ├── types.py         # Type aliases (Array, ScalarFn)
    ├── compensated.py   # Neumaier compensated summation
    ├── stats.py         # safe_log, logsumexp, gaussian_logpdf
    ├── interpolation.py # interp1d, TabulatedFunction1D
    ├── rootfinding.py   # bisect, newton_1d
    ├── integration.py   # trapz, cumulative_trapz, simpson
    ├── linear_algebra.py # norm2, project_onto, condition_number
    └── rng.py           # split_key, split_tree, fold_in_indices
```

## Units Convention

**Always use CGS units** (cm, g, s, erg) as the base system. Physical constants in `constants.py` are in CGS.

Available unit systems in `units.py`:
- `CGS` - base (g, cm, s)
- `ASTRO_STELLAR` - stellar evolution (Msun, Rsun, Myr)
- `ASTRO_DYNAMICAL` - star clusters (Msun, pc, Myr)
- `ASTRO_PLANETARY` - solar system/binaries (Msun, AU, yr)

## Key Patterns

**Enabling float64 precision** (call early, before JAX arrays are created):
```python
from jaxstro.jaxconfig import enable_high_precision
enable_high_precision()
```

**Using unit systems:**
```python
from jaxstro import constants as C, units as U

# Get scales for a unit system
us = U.ASTRO_DYNAMICAL
m_cgs = us.mass_scale_cgs  # 1 Msun in g

# Convert to/from CGS
m, r, t = us.from_cgs(mass_g, length_cm, time_s)
```

## Adding New Functionality

When adding helpers to `numerics/`:
- Keep functions small and domain-agnostic
- Ensure compatibility with `jit`, `vmap`, `grad`
- If it looks like a full solver, it belongs in domain packages or external libraries (diffrax, optimistix, lineax)