---
title: Science patterns enabled by Jaxstro
description: Research questions routed to generic primitives, transform contracts, evidence, and ownership.
---

# Science patterns enabled by Jaxstro

Modules remain the durable technical units of this site. This page takes the
other route: start with a research question, then find its equation, primitive,
transform boundary, evidence, and owner. Use every pattern with
[](../00-start-here/first-research-calculation.md). Signatures live in
[](../40-api/index.md), and claims resolve to [](../60-validation/index.md).

## Locate an event or equilibrium

**Question.** Where does a signed physical residual cross zero?

**Equation.** $G(x;\theta)=0$.

**Primitive.** [](../20-methods/change-constraints-evolution/rootfinding.md) uses
`safeguarded_bracketed_root` for an auditable value and trace.

**Transform boundary.** Scalar `jit`; value/shape `vmap`; physical-cost `lax.map`.

**Failure evidence.** Missing brackets, nonfinite evaluations, exhaustion, and exact roots have distinct statuses.

**Ownership.** Jaxstro owns scalar numerical evidence; the downstream project owns the physical residual and accepted state.

## Differentiate a certified equilibrium

**Question.** How does a unique smooth equilibrium move with a model parameter?

**Equation.** $dx^\star/d\theta=-(\partial_\theta G)/(\partial_x G)$.

**Primitive.** [](../20-methods/change-constraints-evolution/rootfinding.md)
provides `implicit_bracketed_root` separately from the value solver.

**Transform boundary.** The implicit derivative requires uniqueness and smoothness assertions plus convergence, finiteness, residual, width, and slope gates.

**Failure evidence.** Rejection returns NaN for the derivative-facing value and attempted gradient while retaining primal diagnostics.

**Ownership.** Jaxstro checks local numerical evidence; the scientist owns the global branch claim.

## Accumulate a conserved or integrated quantity

**Question.** What total follows from a sampled density, rate, or source term?

**Equation.** $Q=\int_a^b q(x)\,dx$.

**Primitive.**
[](../20-methods/approximation-integration/cumulative-trapz.md),
[](../20-methods/approximation-integration/quadrature.md), and
[](../20-methods/discrete-space/meshes.md) cover sampled, fixed-node, and
conservative accumulation.

**Transform boundary.** Fixed shapes compose with JAX; node choice and discretization error remain distinct from pathwise derivatives.

**Failure evidence.** Exactness, convergence, conservation, and FD comparisons support different claims.

**Ownership.** Jaxstro owns accumulation rules; applications own the integrand and resolution study.

## Interpolate a tabulated physical model

**Question.** How can an expensive table be evaluated between samples without inventing unsupported behavior?

**Equation.** $y(x)$ is reconstructed locally from values, slopes, or grid stencils.

**Primitive.** [](../20-methods/approximation-integration/interpolation.md),
[](../20-methods/approximation-integration/regular-grid.md), and
[](../20-methods/approximation-integration/bsplines.md).

**Transform boundary.** Prepared fixed-rank evaluation is JAX-native; topology and policy selection stay discrete.

**Failure evidence.** Boundary policy, monotonicity, support, affine recovery, and holdouts are separate checks.

**Ownership.** Jaxstro owns interpolation mechanics; source adapters own coordinate semantics and policy evidence.

## Cross a limiting distribution parameter smoothly

**Question.** Can a normalized distribution remain differentiable where its closed form has a removable singularity?

**Equation.** A finite power law approaches a logarithmic form as $\alpha\rightarrow-1$.

**Primitive.** [](../20-methods/probability-sampling/distributions.md) uses
smooth kernels in `powerlaw_logpdf`, `powerlaw_cdf`, and `powerlaw_ppf`.

**Transform boundary.** Interior parameter derivatives are smooth through the limit; support boundaries remain explicit.

**Failure evidence.** Normalization, round trips, analytic limits, and central FD versus AD are checked.

**Ownership.** Jaxstro owns finite kernels, not population semantics or probabilistic programming.

## Transform coordinates, units, and spectra

**Question.** How can an observable change representation without changing physical meaning?

**Equation.** Rotations, dimensional conversions, and density Jacobians preserve named invariants.

**Primitive.** [](./quantities.md), [](./geometry.md), and [](../20-architecture/spectra-data-architecture.md).

**Transform boundary.** Smooth transforms support AD away from singularities; parsing and data preparation stay host-side.

**Failure evidence.** Round trips, conventions, dimensional errors, singular contracts, and product identities are tested.

**Ownership.** Jaxstro owns representation mechanics; domain packages own observational interpretation.

## Find local spatial interactions

**Question.** Which objects can interact within a cutoff, and was allocation sufficient?

**Equation.** Candidate gathering precedes $\lVert x_i-x_j\rVert\le r$.

**Primitive.** [](../20-methods/discrete-space/spatial.md) separates
conservative recall from exact fixed-radius pairs.

**Transform boundary.** Topology is discrete; fixed-shape pair payloads feed JAX kernels.

**Failure evidence.** Recall, symmetry, cutoff, capacity, and overflow are distinct contracts.

**Ownership.** Jaxstro owns indexing and filtering; consumers own interaction physics and overflow policy.

## Connect model PyTrees to inference parameters

**Question.** How can a structured model expose selected leaves to inference?

**Equation.** Bijections map unconstrained vectors to constrained parameters with log-Jacobian terms.

**Primitive.** The parameter bridge in [](../20-architecture/science-general-vision.md) connects PyTrees, vectors, and transforms.

**Transform boundary.** Selected inexact leaves transform; reconstruction and cached-derived values require explicit ownership.

**Failure evidence.** Round trips, transform identities, log determinants, and recovery tests bound the claim.

**Ownership.** Jaxstro owns parameter mechanics, not an optimizer, sampler, likelihood, or model.

## Preserve provenance from artifact to claim

**Question.** Can a result be traced to source data, method, environment, and validation?

**Equation.** Provenance is a graph of artifacts, transformations, assertions, and evidence.

**Primitive.** [](../20-architecture/provenance.md) covers runtime manifests and provenance cards.

**Transform boundary.** Hashing and registry resolution are host-side; identifiers travel beside JAX computations.

**Failure evidence.** Missing artifacts, stale generated pages, unresolved tests, and changed hashes fail explicit gates.

**Ownership.** Jaxstro owns reusable records and routing; scientists own source adequacy and the scientific claim.
