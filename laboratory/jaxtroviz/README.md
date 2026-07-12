# JaxtroViz

JaxtroViz is jaxstro's repository-local, evidence-backed figure laboratory. Its
design system is ported from Startrax's StarViz: declarative panel, encoding,
figure, and export specifications; a shared seaborn visual identity; modular
builders; one registry; and a thin CLI.

JaxtroViz is not part of the `jaxstro` wheel. Builders may consume public
jaxstro APIs, but documentation figures must not create a runtime dependency in
the opposite direction.

## Layout

- `specs.py`: `PanelSpec`, `EncodingSpec`, `ExportSpec`, and registered
  `FigureSpec` contracts.
- `style.py`: shared palette, axis polish, and PDF/PNG/WebP export.
- `architecture.py`, `spatial.py`, and numerical-method modules: one module per
  figure family.
- `registry.py`: the only place figures are registered.
- `cli.py`: list, render, and committed-WebP freshness checks.
- `design/`: human-readable story, evidence, and acceptance specifications.
- `plots/`: ignored paper/raster masters; website WebP copies live beside their
  MyST pages and are committed.

## Commands

```bash
uv run --extra viz python -m laboratory.jaxtroviz --list
uv run --extra viz python -m laboratory.jaxtroviz --only jaxstro-foundation
uv run --extra viz python -m laboratory.jaxtroviz --only spatial-neighbor-contracts
uv run --extra viz python -m laboratory.jaxtroviz --only bspline-local-support
uv run --extra viz python -m laboratory.jaxtroviz --only interpolation-shape-contracts
uv run --extra viz python -m laboratory.jaxtroviz --only regular-grid-contracts
uv run --extra viz python -m laboratory.jaxtroviz --only linear-algebra-contracts
uv run --extra viz python -m laboratory.jaxtroviz --only spectra-runtime-boundary
uv run --extra viz python -m laboratory.jaxtroviz --check
```

Every builder uses a fixed configuration and seed recorded in the registry.
Figure labels that claim numerical membership or status must be derived from the
public API result, not manually typed to resemble an expected result.
