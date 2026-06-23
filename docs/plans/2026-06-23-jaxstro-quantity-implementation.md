# jaxstro Quantity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `jaxstro.quantity` as a JAX-aware physical quantity system with concrete units, exact dimensions, conversions, parser/serialization, astrophysical bases, constants, equivalencies, docs, and backwards-compatible coexistence with `jaxstro.units`.

**Architecture:** Add a layered `src/jaxstro/quantity/` package with immutable `Unit` and PyTree `Quantity` primitives, fixed-vector exact dimensions, layered registries, strict parser/canonical formatter, role-aware bases, versioned constants, explicit equivalencies, and dimension-aware math wrappers. Keep public APIs ergonomic through `jaxstro.quantity` reexports and keep existing `jaxstro.units` unchanged.

**Tech Stack:** Python dataclasses, `fractions.Fraction`, JAX/JAX NumPy, PyTree registration, no SciPy/SymPy runtime dependency, pytest, ruff, mypy, MyST docs.

**Execution Mode:** Implement this plan directly with `superpowers:executing-plans`.
Do not use `superpowers:subagent-driven-development` for this pass.

**Pre-flight Checks:**
- Confirm the working tree is clean or only contains intentional Quantity work:
  `git status --short --branch`.
- Read `CLAUDE.md`, `README.md`, `pyproject.toml`,
  `docs/20-architecture/quantity-system.md`, and
  `docs/plans/2026-06-23-jaxstro-quantity-design.md`.
- Keep `jaxstro.units` backwards compatible. Any compatibility break is a plan
  violation unless Anna explicitly approves it.
- Use TDD: write the targeted failing test, verify it fails for the expected
  reason, implement the smallest passing slice, run the task checks, then
  commit.

---

### Task 1: Dimension Core

**Files:**
- Create: `src/jaxstro/quantity/__init__.py`
- Create: `src/jaxstro/quantity/dimensions.py`
- Create: `src/jaxstro/quantity/errors.py`
- Test: `tests/unit/test_quantity_dimensions.py`

**Step 1: Write failing tests for exact dimensions**

Cover:
- base dimensions: mass, length, time, temperature, current, amount, luminosity;
- dimensionless singleton;
- multiplication/division/powers;
- rational powers with `Fraction(1, 2)`;
- readable derived dimensions: velocity, acceleration, energy, luminosity-like rate if included;
- hash/equality stability;
- invalid float exponent rejection at the dimension layer unless explicitly rationalized before entry.

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_quantity_dimensions.py -q`

Expected: fails because `jaxstro.quantity` does not exist.

**Step 2: Implement minimal dimension objects**

Use an immutable dataclass with a fixed tuple of `Fraction` exponents. Provide
named constants and arithmetic methods.

**Step 3: Run and commit**

Run:

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_quantity_dimensions.py -q
env -u VIRTUAL_ENV uv run --no-sync ruff check src tests
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests
```

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_dimensions.py
git commit -m "Add exact quantity dimensions"
```

### Task 2: Unit Algebra and Core Units

**Files:**
- Create: `src/jaxstro/quantity/unit.py`
- Create: `src/jaxstro/quantity/units.py`
- Create: `src/jaxstro/quantity/astro.py`
- Modify: `src/jaxstro/quantity/__init__.py`
- Test: `tests/unit/test_quantity_units.py`

**Step 1: Write failing tests for units**

Cover:
- concrete objects: `cm`, `m`, `km`, `g`, `kg`, `s`, `day`, `yr`, `erg`, `Hz`,
  `K`, `rad`, `deg`, `nm`, `micron`, `um`, `AU`, `pc`, `Msun`, `Rsun`, `Lsun`;
- scale-to-CGS values for representative conversions;
- unit multiplication/division/powers;
- exact rational powers;
- strict symbols and stable repr/string;
- `3 * q.cm` creates a quantity after Task 3 or initially raises pending behavior.

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_quantity_units.py -q`

**Step 2: Implement unit dataclass and documented units**

Keep unit objects immutable. Make `Unit.__mul__`, `__truediv__`, and `__pow__`
return composite units. Defer quantity construction to Task 3 if necessary.

**Step 3: Run and commit**

Run targeted pytest, ruff check, ruff format check.

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_units.py
git commit -m "Add quantity unit algebra"
```

### Task 3: Quantity Arithmetic and JAX PyTree Support

**Files:**
- Create: `src/jaxstro/quantity/quantity.py`
- Modify: `src/jaxstro/quantity/unit.py`
- Modify: `src/jaxstro/quantity/__init__.py`
- Modify: `src/jaxstro/__init__.py`
- Test: `tests/unit/test_quantity_arithmetic.py`
- Test: `tests/unit/test_quantity_import_surface.py`
- Test: `tests/validation/test_quantity_jax_transforms.py`

**Step 1: Write failing tests for arithmetic**

Cover:
- `q.Quantity(value, unit)`;
- `value * q.cm`;
- addition/subtraction with compatible units preserving left unit;
- incompatible addition raises `DimensionError` with structured fields;
- multiplication/division combine units;
- powers;
- raw scalar rules;
- `.to(unit)`, `.to_value(unit)`, `.to_cgs()`, `.to_cgs_value()`;
- one unit per array.

**Step 2: Write failing JAX transform tests**

Cover:
- `jax.jit(lambda x: (x * q.cm).to_value(q.m))(value)`;
- `jax.vmap` over quantity values;
- `jax.grad` through arithmetic and conversion scale factors;
- static unit metadata does not become a traced array.

**Step 3: Write failing import-surface tests**

Cover:
- `import jaxstro.quantity as q`;
- `from jaxstro import quantity`;
- `"quantity" in jaxstro.__all__`;
- ergonomic reexports such as `q.Quantity`, `q.Unit`, `q.cm`, `q.Msun`.

**Step 4: Implement PyTree Quantity**

Register `Quantity` as a PyTree with value as the child and unit as auxiliary
data. Unit metadata must be immutable and hashable so JAX can safely treat it as
static auxiliary data. Implement arithmetic conservatively and keep unit metadata
static.

**Step 5: Run and commit**

Run targeted tests plus `env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro`
if typing changed substantially.

Commit:

```bash
git add src/jaxstro/__init__.py src/jaxstro/quantity tests/unit/test_quantity_arithmetic.py tests/unit/test_quantity_import_surface.py tests/validation/test_quantity_jax_transforms.py
git commit -m "Add quantity arithmetic"
```

### Task 4: Layered Registries

**Files:**
- Create: `src/jaxstro/quantity/registry.py`
- Modify: `src/jaxstro/quantity/units.py`
- Modify: `src/jaxstro/quantity/astro.py`
- Modify: `src/jaxstro/quantity/__init__.py`
- Test: `tests/unit/test_quantity_registry.py`

**Step 1: Write failing tests for registries**

Cover:
- core and astro registry lookup;
- parent registry lookup;
- scoped extension registry;
- strict aliases;
- friendly close-match suggestions;
- global registration API marked/documented for interactive use.

**Step 2: Implement registries**

Avoid hidden global mutation in library paths. Make default registries immutable
or mutation-controlled after construction.

**Step 3: Run and commit**

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_registry.py
git commit -m "Add layered quantity registries"
```

### Task 5: Parser and Canonical Formatter

**Files:**
- Create: `src/jaxstro/quantity/parser.py`
- Modify: `src/jaxstro/quantity/__init__.py`
- Test: `tests/unit/test_quantity_parser.py`

**Step 1: Write failing parser tests**

Cover accepted strings:

```text
cm
km/s
Msun/yr
erg / s / cm^2 / Hz
g cm^2 s^-2
(km/s)^2
cm^(1/2)
cm^0.5
sqrt(cm)
(erg/cm^3)^0.5
```

Cover rejection:
- unknown symbol with suggestion;
- arbitrary function names;
- decimal exponent that does not rationalize within the documented tolerance;
- malformed parentheses.

Cover canonical formatting:
- `cm^0.5` formats as `cm^(1/2)`;
- composite expressions are deterministic.

**Step 2: Implement recursive-descent or token-based parser**

Do not use Python `eval`. Prefer a small explicit grammar. Use `Fraction` for
all powers. If decimal rationalization is implemented, document and test the
maximum denominator and tolerance.

**Step 3: Run and commit**

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_parser.py
git commit -m "Add quantity unit parser"
```

### Task 6: Serialization

**Files:**
- Create: `src/jaxstro/quantity/serialization.py`
- Modify: `src/jaxstro/quantity/quantity.py`
- Modify: `src/jaxstro/quantity/unit.py`
- Test: `tests/unit/test_quantity_serialization.py`

**Step 1: Write failing serialization tests**

Cover:
- compact known-unit quantity dicts;
- structured custom unit fallback;
- deterministic round-trip;
- array values represented in a JSON-safe way or rejected with a clear error if
  explicit array serialization helpers are deferred.

**Step 2: Implement serialization helpers**

Provide public helpers such as `to_dict`, `from_dict`, `unit_to_dict`,
`unit_from_dict`, and/or methods if that matches local style.

**Step 3: Run and commit**

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_serialization.py
git commit -m "Add quantity serialization"
```

### Task 7: Bases

**Files:**
- Create: `src/jaxstro/quantity/bases.py`
- Modify: `src/jaxstro/quantity/quantity.py`
- Modify: `src/jaxstro/quantity/__init__.py`
- Test: `tests/unit/test_quantity_bases.py`

**Step 1: Write failing basis tests**

Cover:
- `CGS`, `SI`, `STELLAR`, `PLANETARY`, `STAR_CLUSTER`, `CLOSE_BINARY`,
  `WIDE_BINARY`, `COMPACT_BINARY`;
- role-aware choices for stellar mass, planet mass, stellar radius, orbital
  separation, velocity, luminosity, and time;
- aliases such as `DYNAMICAL` and `EXOPLANET`;
- unknown role errors with suggestions.

**Step 2: Implement UnitPreference and UnitBasis**

Use immutable dataclasses. Add `Quantity.to_basis(basis, role=...)`.

**Step 3: Run and commit**

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_bases.py
git commit -m "Add astrophysical quantity bases"
```

### Task 8: Constants and Provenance

**Files:**
- Create: `src/jaxstro/quantity/constants.py`
- Test: `tests/unit/test_quantity_constants.py`
- Modify docs if source values are pinned: `docs/99-bibliography/references.bib`

**Step 1: Verify source values**

Use primary, current CODATA/NIST and IAU sources for the selected constant set.
Record exact source names, versions, URLs or citations, and retrieval/access
date in code metadata and docs. If current values differ from existing
`jaxstro.constants` values, keep `jaxstro.constants` stable and document the
intentional Quantity value set rather than silently changing legacy constants.

**Step 2: Write failing tests**

Cover:
- constants are quantities;
- source/version metadata is inspectable;
- raw-value helpers return expected CGS values;
- constants match existing `jaxstro.constants` where applicable or document any
  intentional version update.

**Step 3: Implement constants registry**

Avoid changing existing `jaxstro.constants` behavior in this task unless a
separate compatibility decision is made.

**Step 4: Run and commit**

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_constants.py docs/99-bibliography/references.bib
git commit -m "Add versioned quantity constants"
```

### Task 9: Equivalencies

**Files:**
- Create: `src/jaxstro/quantity/equivalencies.py`
- Modify: `src/jaxstro/quantity/quantity.py`
- Test: `tests/unit/test_quantity_equivalencies.py`

**Step 1: Write failing tests**

Cover:
- exact-dimensional conversion rejects wavelength to frequency without
  equivalency;
- `spectral()` converts wavelength, frequency, and photon energy both ways;
- `temperature_energy()` uses `k_B`;
- `mass_energy()` uses `c^2`;
- impossible equivalencies raise structured errors.

**Step 2: Implement explicit equivalency objects**

Keep equivalencies opt-in through `.to(..., equivalencies=...)`.

**Step 3: Run and commit**

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_equivalencies.py
git commit -m "Add explicit quantity equivalencies"
```

### Task 10: Dimension-Aware Math Wrappers

**Files:**
- Create: `src/jaxstro/quantity/math.py`
- Modify: `src/jaxstro/quantity/__init__.py`
- Test: `tests/unit/test_quantity_math.py`
- Test: `tests/validation/test_quantity_math_gradients.py`

**Step 1: Write failing tests**

Cover:
- `sqrt`, `square`;
- `log` and `exp` requiring dimensionless input;
- trig functions accepting tagged angles;
- `sum`, `mean`, and `where` preserving or checking units;
- gradient behavior through representative wrappers.

**Step 2: Implement wrappers**

Use JAX NumPy internally. Keep wrapper coverage deliberate rather than broad.

**Step 3: Run and commit**

Commit:

```bash
git add src/jaxstro/quantity tests/unit/test_quantity_math.py tests/validation/test_quantity_math_gradients.py
git commit -m "Add dimension-aware quantity math"
```

### Task 11: Compatibility Bridges

**Files:**
- Modify: `src/jaxstro/units.py`
- Create or modify: `tests/unit/test_quantity_units_compat.py`
- Modify docs: `docs/20-architecture/quantity-system.md`

**Step 1: Write failing compatibility tests**

Cover:
- existing `jaxstro.units` imports still work;
- optional bridge helpers from `UnitSystem` to representative quantity units or
  conversion scales;
- no deprecation warnings emitted in v1 unless explicitly approved.

**Step 2: Implement additive bridges only**

Do not move `UnitSystem`. Do not break existing aliases.

**Step 3: Run and commit**

Commit:

```bash
git add src/jaxstro/units.py tests/unit/test_quantity_units_compat.py docs/20-architecture/quantity-system.md
git commit -m "Bridge quantity and legacy unit systems"
```

### Task 12: Documentation

**Files:**
- Create: `docs/10-theory/quantities.md`
- Modify: `docs/10-theory/index.md`
- Modify: `docs/20-architecture/quantity-system.md`
- Modify: `docs/40-api/index.md`
- Modify: `docs/50-howto/index.md`
- Create: `docs/50-howto/quantity-migration.md`
- Modify: `docs/60-validation/index.md`
- Modify: `docs/myst.yml`
- Modify: `STATUS.md`

**Step 1: Write docs**

Cover:
- conceptual guide;
- parser grammar and canonical serialization;
- bases and roles;
- constants/provenance;
- equivalencies;
- migration from `jaxstro.units`;
- downstream boundary pattern;
- explicit roadmap exclusions.

**Step 2: Build docs**

Run:

```bash
cd docs && myst build
```

Expected: build succeeds without new warnings.

**Step 3: Run full gate**

Run:

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q
env -u VIRTUAL_ENV uv run --no-sync ruff check src tests
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
cd docs && myst build
```

**Step 4: Commit**

```bash
git add docs STATUS.md
git commit -m "Document quantity system"
```

### Task 13: Final Integration Gate

**Files:**
- Modify only if gate failures reveal necessary fixes.

**Step 1: Run full verification**

Run:

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q
env -u VIRTUAL_ENV uv run --no-sync ruff check src tests
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
cd docs && myst build
```

**Step 2: Fix only gate failures**

Keep fixes minimal and scoped.

**Step 3: Commit final polish**

If changes were needed:

```bash
git add <changed-files>
git commit -m "Polish quantity integration"
```

**Step 4: Report**

Summarize implemented behavior, docs, tests, and migration posture. Do not claim
downstream packages were migrated unless that work actually happened.
