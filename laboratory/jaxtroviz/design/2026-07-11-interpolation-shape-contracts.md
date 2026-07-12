# Interpolation shape-contract figure

## Learner question

How can a smooth natural cubic spline violate the shape implied by monotone
samples, and what does the PCHIP limiter preserve instead?

## Evidence source

The builder evaluates the public `natural_cubic_spline_coeffs`,
`eval_cubic_spline`, and `monotone_cubic_interp` APIs for the same five monotone
samples. It asserts that the natural spline minimum is below `-0.1`, while the
PCHIP result remains within `[0, 1]` and has no negative successive increment on
the displayed 801-point query grid.

## Visual encoding

- Purple curve: natural cubic interpolation.
- Teal curve: PCHIP interpolation.
- Dark points: the shared monotone samples.
- Coral fill: regions where the natural curve or its successive increments are
  negative.
- Dashed neutral line: the zero reference.

The website alt text names the natural-spline undershoot and nonnegative PCHIP
increments without depending on color.

## Does not prove

This fixture does not establish universal superiority of one interpolant,
population-wide error, or a physical extrapolation model. It demonstrates the
distinct smoothness and shape contracts for one executable monotone table.

## Acceptance

- Registry and deterministic-render tests pass.
- The generated WebP is fresh.
- The documentation example independently checks natural undershoot, PCHIP
  bounds and monotonicity, Hermite equivalence, and wrapper parity.
- The rendered MyST DOM contains the figure identifier, alt text, and built
  asset.
