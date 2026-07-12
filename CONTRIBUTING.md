# Contributing to jaxstro

Thank you for helping improve jaxstro. Contributions should preserve its role as
a lightweight, domain-agnostic JAX foundation: explicit physical conventions,
fixed-shape transformable kernels, and evidence-backed numerical claims.

## Development setup

Use Python 3.11 or newer and install the locked development environment:

```bash
uv sync --locked --extra dev
```

Run focused tests while developing. Before opening a pull request, run the local
release mirror:

```bash
bash scripts/check.sh
```

Documentation-only changes must also pass the rendered-site gate directly:

```bash
bash scripts/check_docs.sh
```

## Scientific changes

- Add a failing regression or invariant test before changing behavior.
- Cite primary sources with an exact table, equation, section, or archive locator.
- State units, frames, coordinate conventions, and differentiability boundaries.
- Keep host-side selection and I/O outside JIT-compiled array kernels.
- Do not weaken tolerances merely to make a failing numerical test pass.

Use `jax.numpy`, `jax.jit`, `jax.vmap`, and `jax.lax` primitives for transformable
code. Expected scientific gaps should use the package's structured status types
rather than hidden extrapolation or silent fallbacks.

## Documentation changes

Examples advertised as executable must run against public imports. Use native
MyST cross-references and verify rendered HTML for link, ID, and accessibility
claims. Figures need deterministic inputs and descriptive alternative text.

## Pull requests

Keep each pull request coherent and explain:

1. the behavior or claim that changed;
2. the evidence and tests supporting it;
3. any intentionally unsupported boundary; and
4. the exact verification commands run.

Do not include local datasets, credentials, generated build trees, or machine-
specific paths. By contributing, you agree that your work is released under the
repository's Apache-2.0 license.
