# Linear-algebra contract figure

## Learner question

How can one declared observation weight change a least-squares fit, and what does
diagonal jitter actually do to an indefinite matrix?

## Evidence source

The left panel calls the public `weighted_lstsq` API twice for the same four
observations: once with implicit unit weights and once with the final outlier's
weight set to zero. The right panel calls `positive_definite_jitter` on a fixed
diagonal matrix and plots eigenvalues of the public returned matrix.

## Visual encoding

- Dark circles are unit-weight observations; the purple cross is the declared
  zero-weight outlier.
- Purple dashed and teal solid lines are the measured unweighted and weighted
  coefficient vectors.
- Paired bars show eigenvalues before and after the selected diagonal shift.
- The dashed zero line is the positive-definite boundary.

The website alt text names both comparisons without depending on color.

## Does not prove

This fixture does not establish robust-regression quality, prescribe zero weights,
compare solver performance, or claim the geometric jitter search is a nearest
positive-definite projection. It displays two explicit API contracts on fixed
small examples.

## Acceptance

- Public weighted coefficients equal `(1, 2)` and the unweighted coefficients
  equal `(-1.6, 5.9)`.
- The selected jitter is `0.1`; the smallest eigenvalue changes from negative to
  positive and `success` is true.
- Registry and deterministic-render tests pass and the committed WebP is fresh.
- The rendered MyST DOM contains the figure identifier, alt text, and built asset.
