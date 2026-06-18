# jaxstro.params Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan
> task-by-task (fresh subagent per task + code-review subagent between tasks). Apply research-workflow
> gates: gradient-validation (every public path FD-vs-AD grad-checked), numerical-method-validation
> (bijector log-Jacobians), verification-gate. Design: `docs/plans/2026-06-17-jaxstro-params-design.md`
> (ADR-0009). Test against **progenax**, not gravax.

**Goal:** Add `jaxstro.params` — an Equinox-only utility for gradient-based inference over a *subset* of an
Equinox model's parameters (free/fixed marking + PyTree↔flat-vector bridge + transform registry).

**Architecture:** `Parameterization` (an `eqx.Module` holding a static boolean filter-spec + a static
transform spec) lowers a typed `where`-callable to a bool mask, partitions the model via `eqx.partition`,
ravels the free subtree via `jax.flatten_util.ravel_pytree`, and reconstructs with `eqx.combine`. Bijectors
move bounded params to/from unconstrained ℝ with analytic log-Jacobians. No new core dependency; optax/
numpyro only behind the `[ml]` extra and only in the validation script.

**Tech Stack:** JAX (`jax.numpy`, `jax.flatten_util.ravel_pytree`, `jit`/`grad`/`vmap`), equinox
(`partition`/`combine`/`filter`/`tree_at`/`field(static=True)`), jaxtyping. Validation: optax + numpyro
(`[ml]` extra). float64 via `enable_high_precision()` in test conftest.

**Branch:** `feature/jaxstro-params` (already created off main; main has Phase B merged).

**Working dir:** `/Users/anna/projects/jaxstro-dev/jaxstro`. Sibling for validation: `../progenax`.

---

## Conventions (every task)

- **Verify:** `env -u VIRTUAL_ENV uv run --no-sync pytest <paths> -q`. Full baseline = **390 passed** (must
  not regress).
- **TDD:** failing test → run red → minimal impl → run green → commit. **Never weaken a test/tolerance.**
- **Grad-check** = finite-difference vs `jax.grad`/`jacrev` (prefer `jacrev`); under x64 (conftest enables it).
- **JAX-native:** `jax.numpy`, equinox; jit/grad/vmap-safe; no Python loops in hot paths; no `while_loop`/
  `argmax`/in-place on differentiated paths. numpy nowhere in core params.
- Stage files EXPLICITLY (never `git add -A`). Trailer ends with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do NOT push/merge.
- Size guidelines: ~500 LOC/file, ~100 LOC/function; split by responsibility.

---

## Task 0 — `[ml]` optional extra (validation deps)

**Files:** Modify `pyproject.toml`.

**Steps:**
1. Add an optional-dependencies group: `[project.optional-dependencies] ml = ["optax>=0.2.0", "numpyro>=0.15.0"]`
   (verify current latest acceptable lower bounds against the existing `uv.lock`/PyPI; keep core
   `dependencies` unchanged — jax + jaxtyping + equinox only).
2. `env -u VIRTUAL_ENV uv lock` then `env -u VIRTUAL_ENV uv sync --extra ml` to confirm it resolves.
3. **Verify:** `env -u VIRTUAL_ENV uv run --no-sync python -c "import optax, numpyro; print('ml ok')"`.
4. **Commit** (`pyproject.toml uv.lock`): `build: add [ml] optional extra (optax, numpyro) for params validation`.

**Gate:** verification-gate (paste resolve + import).

---

## Task 1 — `Parameterization` core (marking + PyTree↔vector bridge, NO transforms yet)

**Files:**
- Create: `src/jaxstro/params/__init__.py`, `src/jaxstro/params/parameterization.py`
- Test: `tests/unit/test_params_parameterization.py`
- Modify: `src/jaxstro/__init__.py` (export `params` subpackage, like `numerics`/`testing`)

**Step 1 — failing tests** (write all, run red):
```python
import jax, jax.numpy as jnp, equinox as eqx
from jaxstro.params import Parameterization

class Model(eqx.Module):
    a: jax.Array
    b: jax.Array
    name: str = eqx.field(static=True, default="m")

def _m(): return Model(a=jnp.array([1.0, 2.0]), b=jnp.array(3.0))

def test_roundtrip_identity():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.array_equal(m2.a, m.a) and jnp.array_equal(m2.b, m.b)

def test_fixed_leaf_preserved():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a,))   # only a is free
    m2 = p.from_vector(m, p.to_vector(m) + 10.0)
    assert jnp.array_equal(m2.b, m.b)                            # b untouched
    assert jnp.allclose(m2.a, m.a + 10.0)

def test_from_where_equals_from_filter():
    m = _m()
    pw = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    spec = eqx.tree_at(lambda x: (x.a, x.b), jax.tree_util.tree_map(lambda _: False, eqx.filter(m, eqx.is_array)),
                       replace=(True, True))
    pf = Parameterization.from_filter(m, spec)
    assert jnp.array_equal(pw.to_vector(m), pf.to_vector(m))

def test_vector_length_is_free_param_count():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    assert p.to_vector(m).shape == (3,)                         # 2 (a) + 1 (b)

def test_empty_free_set():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: ())
    assert p.to_vector(m).shape == (0,)
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.array_equal(m2.a, m.a) and jnp.array_equal(m2.b, m.b)

def test_grad_through_from_vector():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    v0 = p.to_vector(m)
    def loss(v): mm = p.from_vector(m, v); return jnp.sum(mm.a**2) + mm.b**2
    g = jax.grad(loss)(v0)
    h = 1e-6
    fd = jnp.array([(loss(v0.at[i].add(h)) - loss(v0.at[i].add(-h)))/(2*h) for i in range(v0.size)])
    assert jnp.allclose(g, fd, rtol=1e-5, atol=1e-6)

def test_jit_and_vmap():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    f = jax.jit(lambda v: p.from_vector(m, v).b)
    assert jnp.allclose(f(p.to_vector(m)), m.b)
    batch = jnp.stack([p.to_vector(m), p.to_vector(m)+1.0])
    out = jax.vmap(lambda v: p.from_vector(m, v).b)(batch)
    assert out.shape == (2,)
```
Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_params_parameterization.py -q` → RED.

**Step 2 — implement** `parameterization.py`:
- `Parameterization(eqx.Module)` with `free_spec: PyTree = eqx.field(static=True)` (a bool PyTree over the
  array leaves; non-array/static leaves are `None`/excluded).
- `from_filter(cls, model, free_spec)`: store `free_spec`.
- `from_where(cls, model, where)`: build an all-False template over array leaves
  (`jax.tree_util.tree_map(lambda _: False, eqx.filter(model, eqx.is_array))`), set the `where`-selected
  leaves True via `eqx.tree_at(where, template, replace=tuple(True for _ in selected))`, call `from_filter`.
- `_partition(model)`: `eqx.partition(model, self.free_spec)` → `(free, fixed)`.
- `to_vector(model)`: `ravel_pytree(eqx.filter(model, self.free_spec))[0]` (free leaves only; filter returns
  the free subtree with `None` elsewhere — ravel ignores `None`).
- `from_vector(model, vec)`: get `(free, fixed) = _partition(model)`; `_, unravel = ravel_pytree(free)`;
  `eqx.combine(unravel(vec), fixed)`.
- Keep it pure/static; no closures stored on the module.
- Full NumPy-style docstrings + jaxtyping; document round-trip + differentiability guarantees.
- `__init__.py` exports `Parameterization`.
- `src/jaxstro/__init__.py`: add `params` to the import line + `__all__`.

Run green. **Commit** (`src/jaxstro/params/__init__.py src/jaxstro/params/parameterization.py
src/jaxstro/__init__.py tests/unit/test_params_parameterization.py`):
`feat(params): Parameterization — free/fixed marking + PyTree<->vector bridge`.

**Gate:** gradient-validation, verification-gate.

---

## Task 2 — transform registry (bijectors)

**Files:**
- Create: `src/jaxstro/params/transforms.py`
- Test: `tests/unit/test_params_transforms.py`
- Modify: `src/jaxstro/params/__init__.py` (export bijectors)

**Step 1 — failing tests** (numpyro/TFP convention: `forward: ℝ→physical`):
```python
import jax, jax.numpy as jnp
from jaxstro.params.transforms import Identity, Exp, Softplus, Sigmoid

import pytest
@pytest.mark.parametrize("bij,x", [(Identity(), 0.7), (Exp(), 2.3), (Softplus(), 1.5), (Sigmoid(0.0, 1.0), 0.3)])
def test_roundtrip(bij, x):
    x = jnp.asarray(x)
    assert jnp.allclose(bij.forward(bij.inverse(x)), x, rtol=1e-10)

@pytest.mark.parametrize("bij,u", [(Exp(), 0.4), (Softplus(), -1.2), (Sigmoid(2.0, 5.0), 0.8), (Identity(), 1.1)])
def test_log_det_matches_autodiff(bij, u):
    u = jnp.asarray(u)
    ad = jnp.log(jnp.abs(jax.grad(lambda z: bij.forward(z))(u)))
    assert jnp.allclose(bij.forward_log_det_jacobian(u), ad, rtol=1e-6, atol=1e-8)

def test_sigmoid_bounds():
    s = Sigmoid(2.0, 5.0)
    us = jnp.linspace(-20, 20, 50)
    ys = jax.vmap(s.forward)(us)
    assert jnp.all((ys > 2.0) & (ys < 5.0))

def test_exp_positive():
    assert jnp.all(jax.vmap(Exp().forward)(jnp.linspace(-30, 30, 50)) > 0.0)
```
Run RED.

**Step 2 — implement** `transforms.py`: each bijector an `eqx.Module` with `forward`, `inverse`,
`forward_log_det_jacobian`. Use `jax.nn.softplus`, `jax.nn.log_sigmoid` for stability. `Sigmoid(lo, hi)`
stores `lo`,`hi` as static floats. Analytic log-dets per the design table. Docstrings cite the
change-of-variables formula.

Run green. **Commit** (`src/jaxstro/params/transforms.py src/jaxstro/params/__init__.py
tests/unit/test_params_transforms.py`): `feat(params): bijector registry (Exp/Softplus/Sigmoid) + analytic log-Jacobians`.

**Gate:** numerical-method-validation, gradient-validation.

---

## Task 3 — integrate transforms into `Parameterization`

**Files:**
- Modify: `src/jaxstro/params/parameterization.py` (add `transforms` static field + `log_det_jacobian`)
- Test: `tests/unit/test_params_transformed.py`

**Step 1 — failing tests:**
```python
import jax, jax.numpy as jnp, equinox as eqx
from jaxstro.params import Parameterization
from jaxstro.params.transforms import Exp, Sigmoid

class M(eqx.Module):
    r_h: jax.Array
    Q: jax.Array
def _m(): return M(r_h=jnp.array(1.3), Q=jnp.array(0.4))

def test_transformed_roundtrip_and_bounds():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0)))
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.allclose(m2.r_h, m.r_h) and jnp.allclose(m2.Q, m.Q)
    # vector lives in unconstrained R; perturb and confirm bounds hold after forward
    m3 = p.from_vector(m, p.to_vector(m) + jnp.array([5.0, -8.0]))
    assert m3.r_h > 0.0 and 0.0 < m3.Q < 1.0

def test_log_det_jacobian_sums_per_leaf():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0)))
    v = p.to_vector(m)
    expected = Exp().forward_log_det_jacobian(v[0]) + Sigmoid(0.0,1.0).forward_log_det_jacobian(v[1])
    assert jnp.allclose(p.log_det_jacobian(v), expected, rtol=1e-8)

def test_end_to_end_grad_through_transformed():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0)))
    v0 = p.to_vector(m)
    def loss(v): mm = p.from_vector(m, v); return mm.r_h**2 + mm.Q**2
    g = jax.grad(loss)(v0); assert jnp.all(jnp.isfinite(g))
```
Run RED.

**Step 2 — implement:** add `transforms: tuple = eqx.field(static=True, default=())` aligned with the
`where` tuple (default all-`Identity` when `None`); `to_vector` applies each bijector's `inverse` to the
corresponding free leaf before raveling; `from_vector` applies `forward` after unravel; `log_det_jacobian`
sums per-leaf `forward_log_det_jacobian` over the raveled vector slices. Map vector slices↔leaves via the
free partition's tree structure (static). Document the unconstrained-space contract.

Run green + the Task-1 tests still pass (untransformed default). **Commit**: `feat(params): unconstrained-space transforms in Parameterization (to/from + log_det_jacobian)`.

**Gate:** gradient-validation, numerical-method-validation.

---

## Task 4 — integration test + validation script + docs

**Files:**
- Test: `tests/integration/test_params_optax.py`
- Create: `validation/validate_params.py`, `.claude-work/JAXSTRO_PARAMS_COMPLETE.md`
- Modify: `README.md` (usage section)

**Steps:**
1. **Integration test** (`tests/integration/test_params_optax.py`): a real optax loop on a toy
   `eqx.Module` recovers a known scalar from noiseless data (assert recovered ≈ true to ~1e-3). Skip
   cleanly if optax absent (`pytest.importorskip("optax")`).
2. **Validation script** `validation/validate_params.py` (CLI): try to import a real **progenax** Equinox
   model (an IMF/profile model — discover one that's importable; e.g. a Plummer profile or IMF object with a
   scalar free param); mark 1–2 free params; inject a known truth; recover via (a) an optax descent and (b)
   a tiny numpyro chain (using `log_det_jacobian` for change-of-variables). Print a table: param · true ·
   recovered · abs/rel error · FD-vs-AD grad error. **Clean fallback** to a toy `eqx.Module` if progenax
   isn't importable (print which path ran). `pytest.importorskip`-style guards for optax/numpyro.
3. Run it: `env -u VIRTUAL_ENV uv run --no-sync --extra ml python validation/validate_params.py` — confirm
   recovery within tolerance + grad-check pass; paste the table into the completion doc.
4. **Sibling smoke-test:** `cd ../progenax && env -u VIRTUAL_ENV uv run --no-sync python -c "import progenax"`
   and same for fluxax/gravax — confirm the new jaxstro.params doesn't break sibling imports.
5. **README** usage section + **completion doc** `.claude-work/JAXSTRO_PARAMS_COMPLETE.md` (API, the
   Zodiax-vs-ours rationale, per-package adoption guide, test + validation results).
6. **Full suite** green: `env -u VIRTUAL_ENV uv run --no-sync pytest -q` (≥ 390 + new) + `ruff check src/` +
   `mypy src/jaxstro`.
7. **Commit**: `test(params): optax integration + progenax validation + docs`.

**Gate:** verification-gate (paste validation table + green suite + sibling smoke-test).

---

## Definition of complete (this plan)
1. 3-tier tests green (unit round-trip/edge/grad/jit/vmap + transforms + integration); FD-vs-AD grad-checks pass.
2. Validation recovers a known value on a progenax model (or documented toy fallback) via optax + numpyro;
   quantitative recovered-vs-true + grad error printed.
3. No new **core** dependency (optax/numpyro only under `[ml]`); siblings still import.
4. Completion doc + README usage + ADR-0009 reflected; STATUS + brain updated.
