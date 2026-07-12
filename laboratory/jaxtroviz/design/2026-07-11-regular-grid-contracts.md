# Regular-grid interpolation contract figure

## Learner question

How does one interior query combine the corners of a rectangular grid cell, and
what changes when the query leaves the grid domain?

## Evidence source

The builder sends four one-hot corner tables through the public
`bilinear_interp` API at `(0.3, 0.65)`. The returned values are therefore the
measured corner weights. It separately evaluates the same public API along a
fixed-`y` scan using `boundary="clamp"` and `boundary="fill"`.

## Visual encoding

- Line width from the query to each corner encodes its measured bilinear weight;
  every numeric label is computed from the public API result.
- The star marks the interior query and dark points mark grid corners.
- The right panel uses a solid teal clamp result and dashed purple fill result.
- Pale outer bands mark coordinates outside the grid domain.

The website alt text names the four-corner weighting and clamp/fill comparison
without depending on color.

## Does not prove

The figure does not establish interpolation error for nonlinear fields,
high-dimensional scaling, scattered-data behavior, or scientific validity of
either boundary policy. It visualizes the exact contract for one affine unit-cell
fixture.

## Acceptance

- Public-API weights sum to one and equal the analytic bilinear weights.
- Fill returns the selected sentinel outside the domain; clamp remains finite.
- Registry and deterministic-render tests pass and the committed WebP is fresh.
- The rendered MyST DOM contains the figure identifier, alt text, and built
  asset.
