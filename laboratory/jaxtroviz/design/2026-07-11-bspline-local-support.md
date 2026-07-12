# B-spline local-support figure

## Learner question

What do “local support” and “partition of unity” look like for the cubic basis
used by jaxstro?

## Evidence source

The builder calls the public `open_uniform_knots` and `bspline_basis` APIs for
six cubic basis functions on `[0, 1]`. It asserts the returned `(401, 6)` shape,
nonnegativity within floating tolerance, and a basis sum equal to one before
drawing either panel.

## Visual encoding

- Six distinct curves: the six returned cubic basis columns.
- Teal line: the sum of those columns at each displayed query coordinate.
- Dashed neutral line: the reference value one.
- Light fill: any visible separation between the measured sum and one.

The website alt text states the local-support and sum-to-one relationships
without requiring color perception. Panel titles name both concepts directly.

## Does not prove

This fixed open-uniform example does not establish adaptive-knot quality,
smoothing-model selection, extrapolation behavior, or performance. Unit and
gradient tests own the general API contracts; the figure visualizes one
executable configuration.

## Acceptance

- Registry and deterministic-render tests pass.
- The generated WebP is fresh.
- The documentation example independently verifies basis shape, partition of
  unity, wrapper parity, and finite interior derivatives.
- The rendered MyST DOM contains the figure identifier, alt text, and built
  asset.
