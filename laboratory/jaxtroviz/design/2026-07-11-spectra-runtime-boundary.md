# Spectra runtime-boundary figure

## Learner question

Which atmosphere operations touch local files, and which object may safely enter
`jit`, `vmap`, and `grad`?

## Evidence source

The three-stage workflow follows the public ownership split among
`AtmosphereLibrary`, exact-product adapters, `PreparedRectilinearStencil`, and
downstream packages. The evidence strip calls public stencil evaluation on the
same in-memory fixture as the executable page and evaluates one public JAX
gradient.

## Visual encoding

- Gold is host-side catalog, artifact, and cell-selection work.
- Teal is the prepared array-only JAX object and its interpolation contract.
- Purple is downstream observable construction outside jaxstro.
- Directional arrows show which representation crosses each boundary.
- The neutral evidence strip reports measured flux, status codes, and derivative.

The alt text states the three stages without requiring color perception.

## Does not prove

This figure does not report installed atmosphere families, artifact counts,
scientific agreement between model libraries, complete runtime support, or a
derivative through file selection. It illustrates ownership and verifies one
portable interpolation fixture.

## Acceptance

- The midpoint flux is `[2.5, 3.5, 4.5]`.
- In-grid and outside-convex-hull status codes are `0` and `4`.
- Mapping the numeric spectrum payload over two points produces shape `(2, 3)`.
- The local first-flux derivative with respect to temperature is `0.002`.
- Registry, deterministic-render, page, and rendered-DOM checks pass.
