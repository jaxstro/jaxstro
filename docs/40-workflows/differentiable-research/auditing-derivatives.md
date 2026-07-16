# Auditing derivatives

Use this page when an automatic derivative must be checked independently before
it supports a numerical or scientific claim.

## Name the claim first

An audit begins by naming the derivative: input, output, direction, units,
operating point, precision, and smooth domain. A derivative with respect to a
normalized parameter is not interchangeable with one with respect to its
dimensionful physical value. Likewise, a derivative of a finite solver trace is
not automatically an implicit solution sensitivity.

## Analytic and directional checks

Prefer an analytic derivative, invariant, or limiting case when one is
available. For a multivariate map, test a directional finite difference rather
than materializing a full Jacobian. The centered estimator is

```{math}
:label: eq-workflow-directional-fd

D_h f(x;v)
=
\frac{f(x+h v)-f(x-h v)}{2h}
=
J_f(x)v+\mathcal{O}(h^2).
```

Compare $D_hf$ to a JVP and compare VJP/JVP contractions through
$w^{\mathsf T}(Jv)=(w^{\mathsf T}J)v$. These comparisons exercise different AD
modes and catch errors in argument selection or PyTree flattening.

## Step-size studies

One value of $h$ is not evidence of convergence. Evaluate a geometric sequence
of step sizes. At large $h$, truncation error dominates; at small $h$,
subtractive cancellation and roundoff dominate. A credible audit shows an
intermediate region where error decreases near the expected order before the
roundoff floor appears. Scale $h$ relative to the magnitude and units of each
input direction.

Precision is part of the evidence. Float32 may hide the convergence window for
a sensitive map. Enable and record float64 when the contract requires it, but
do not infer that higher precision repairs a nonsmooth or ill-conditioned
problem.

## Nondifferentiable points and branches

Do not center a finite difference across a knot, clamp boundary, sign decision,
integer index change, or other nonsmooth point and then describe the result as a
local derivative. Audit one-sided behavior when scientifically relevant and
state that no unique derivative exists at the boundary. Test invalid domains
explicitly; a finite fallback tangent can conceal a failed primal assumption.

## Concrete audit procedure

1. Freeze the primal fixture and record units, dtype, and expected smooth
   domain.
2. Derive the analytic result or independent limiting identity.
3. Choose normalized directions that exercise every input group.
4. Compare forward and reverse AD contractions.
5. Run a logarithmic step-size study for central finite differences.
6. Repeat near, but not across, relevant branch and singular boundaries.
7. Record errors as named metrics with thresholds justified by the study.
8. Link the executable validation target and state the limitation.

## Auditing an adaptive integral

An adaptive integral needs two finite-difference questions because its public
program contains both continuous arithmetic and discrete decisions.

1. Compare replay AD with the analytic derivative of the exact integral.
2. Freeze the center run's accepted regions or level and finite-difference that
   fixed formula.
3. Rerun the full adaptive solve at both finite-difference samples.
4. Record the accepted regions or levels for every comparison.

The frozen-formula comparison tests the custom derivative directly. The
adaptive-rerun comparison is diagnostic: a change in partition or level can
create a difference without proving that replay is wrong. Repeat the audit over
at least three tolerances and two nonbinding capacities.

For a parameterized integral, record four distinct values:

```{math}
:label: eq-workflow-quad-audit

\frac{\mathrm{d}I}{\mathrm{d}\theta}\bigg|_{\mathrm{analytic}},
\qquad
\frac{\mathrm{d}Q}{\mathrm{d}\theta}\bigg|_{\mathrm{replay\ AD}},
\qquad
D_h Q_{\mathrm{frozen}},
\qquad
D_h Q_{\mathrm{adaptive\ rerun}}.
```

Moving-bound audits must use the signed interval map and include reversed and
coincident bounds. Invalid and nonfinite primal statuses have undefined
derivatives; audit their statuses and fail-closed values without inventing a
tangent layout.

When quantities are involved, differentiate selected raw numerical values and
record the parameter, integral, and derivative units. Repeat one fixture in two
compatible unit representations and convert the resulting physical
derivatives to one common unit.

See [](../../20-methods/approximation-integration/differentiating-an-integral.md)
for the derivation and
[](../../60-validation/numerical/quadrature-replay-derivatives.md) for the
generated five-method evidence.

## Evidence and claim boundaries

AD-versus-FD agreement supports the local derivative of the implemented map at
the tested fixtures. It does not establish global smoothness, branch uniqueness,
model correctness, parameter identifiability, or adequate conditioning.
Self-consistency between two AD modes is weaker than agreement with an
independent method because both modes transform the same implementation.

## Connected ideas

Review [](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md),
[](../../20-methods/change-constraints-evolution/autodiff.md),
[](./branches-limits-implicit-sensitivities.md), and the executable
[](../investigations/powerlaw-removable-limit.md) study.
