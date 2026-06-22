---
title: Grids and sampling utilities
description: >-
  Log grids, geometric bin helpers, conservative rebinning, and stratified
  uniform sampling with explicit differentiability boundaries.
---

Grid utilities look mundane, but they decide whether downstream calculations
preserve mass, probability, luminosity, or whatever quantity a bin total
represents. jaxstro keeps these helpers small and explicit: construction helpers
make common grids, conservative rebinning preserves bin totals over overlap, and
sampling helpers expose deterministic shapes.

## Log grids and bin centers

`log_grid(start, stop, num, base=10)` returns logarithmically spaced samples
between positive endpoints. `geometric_bin_edges(start, stop, n_bins)` is the same
idea for bin edges. `bin_centers(edges)` returns arithmetic centers, while
`geometric_bin_centers(edges)` returns $\sqrt{x_i x_{i+1}}$ for positive edges.

These are construction utilities. Gradients through grid construction are rarely
the scientific quantity of interest; most differentiable calculations should
treat the grid as fixed data and differentiate through values evaluated on that
grid.

## Conservative rebinning

`conservative_rebin(old_edges, values, new_edges)` interprets `values` as
integrated totals in the old bins. It distributes each old total uniformly across
its old bin and computes overlap with every new bin:

```{math}
v'_j = \sum_i v_i\,
\frac{\max(0, \min(e'_{j+1}, e_{i+1}) - \max(e'_j, e_i))}
{e_{i+1} - e_i}.
```

The sum of the new values equals the old total over the overlapping domain. New
bins outside the old domain receive zero contribution. This is the right helper
for count-like or integrated quantities. It is not a flux-density interpolator;
if the input represents density samples rather than bin totals, integrate or
resample with a density-aware method first.

The operation is linear in `values`, so gradients through bin totals are simple
and validated. Gradients with respect to bin edges are piecewise-defined and
change at overlap boundaries; treat edges as fixed preprocessing.

## Stratified uniforms

`stratified_uniform(key, n, minval=0, maxval=1)` draws one uniform sample from each
of `n` equal-width strata and returns them in stratum order. The shape is fixed by
the static `n`, making it `jit` friendly.

This is a variance-reduction primitive, not a distribution sampler by itself. To
sample from a tabulated density, compose stratified uniforms with
`inverse_cdf_draw(...)` via `vmap`. Keep the distinction clear:

- `stratified_uniform` chooses deterministic-coverage uniforms.
- `inverse_cdf_draw` maps uniforms through a tabulated inverse CDF.

## Deferred quasi-random sequences

Sobol and Halton sequences are useful, but a trustworthy implementation needs
direction numbers, scrambling policy, dimensional limits, and validation against
known reference sequences. Those are deferred rather than slipped in as a fragile
dependency-free toy.
