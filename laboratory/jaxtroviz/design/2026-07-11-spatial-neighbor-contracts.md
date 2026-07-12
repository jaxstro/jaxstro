# Spatial neighbor contracts figure

## Learner question

Why is a grid candidate pool not already an exact neighbor list?

## Evidence source

The builder runs one fixed eight-particle cloud through the public
`assign_particles_to_bins`, `fill_bins_exact`,
`gather_candidates_from_bins`, and `gather_pairs_within_radius` APIs. Point
labels include their returned Morton bin IDs. Candidate and neighbor edges are
drawn from returned index masks.

For focal particle 0, the configured grid returns candidates 1, 2, 3, 4, and 5.
The exact `0 < r <= 0.5` filter retains only 1 and 2. The builder asserts neither
bin nor pair overflow through the displayed `did_overflow = False` result.

## Visual encoding

- Orange edges and points: grid candidates, including false positives.
- Teal circle, edges, and points: exact cutoff and accepted neighbors.
- Slate star: focal particle.
- Gray points: outside the candidate pool or rejected by the exact filter.

The website alt text describes the two-panel comparison without depending on
color. Point IDs and panel titles carry the same distinction textually.

## Does not prove

This single cloud does not establish population-wide approximate-kNN recall,
optimal capacity, or performance. Those require dedicated validation across
point distributions. It demonstrates the API distinction and exact result for
one executable configuration.

## Acceptance

- Registry and deterministic-render tests pass.
- The generated WebP is fresh.
- The spatial documentation test independently checks the exact cutoff,
  coincident-particle, and overflow contract.
- The rendered MyST DOM contains the figure identifier, alt text, and built
  asset.
