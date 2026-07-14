# Branches, limits, and implicit sensitivities

Use this page when a derivative crosses piecewise logic, approaches a removable
limit, or is intended to represent the sensitivity of an implicitly defined
solution.

## Piecewise programs

For a piecewise map

```{math}
:label: eq-workflow-piecewise-map

f(x)=
\begin{cases}
f_-(x), & x<0,\\
f_+(x), & x\ge 0,
\end{cases}
```

JAX differentiates the branch executed at the traced value. It does not
differentiate the discrete selection itself. If the branch values or slopes do
not join consistently, no single derivative claim is available at the
boundary. Selection by cell index, clipping, and safeguarded solver decisions
have the same basic concern.

## Removable limits

An exact equality branch can return the correct limiting value while exposing
the wrong parameter derivative. Prefer a shared smooth kernel derived from a
series or stable elementary function. For example,
$\operatorname{expm1}(u)/u$ approaches one smoothly as $u\rightarrow0$; a
separate `u == 0` constant branch can erase the derivative information needed
by a parameter sensitivity.

The audit requires values and derivatives from both sides, a series-derived
coefficient, and a finite-difference step-size study. The finite power-law
[](../investigations/powerlaw-removable-limit.md) investigation demonstrates
that distinction.

## Implicit function theorem

If $F(x,\theta)=0$ defines a locally unique smooth branch and
$\partial_xF\ne0$, the implicit function theorem gives

```{math}
:label: eq-workflow-implicit-sensitivity

\frac{dx^\star}{d\theta}
=
-\frac{\partial_\theta F(x^\star,\theta)}
       {\partial_x F(x^\star,\theta)}.
```

This formula assumes more than numerical convergence. The branch must be
locally unique, $F$ must be differentiable on it, the computed root must satisfy
residual and bracket-width gates, all values must be finite, and the denominator
must be sufficiently far from zero. Small $|\partial_xF|$ signals poor
conditioning and amplifies both model and numerical errors.

## Custom and implicit derivative rules

A custom JVP or VJP changes the derivative contract while leaving the primal
program visible. It must be treated as a scientific method with explicit
assumptions and independent validation, not as a way to force gradients through
unsupported control flow. An implicit rule may deliberately ignore solver
branch history, but only because it claims the mathematical solution derivative
behind separate gates.

## Concrete audit procedure

1. Map every piecewise, clipping, indexing, and stopping boundary.
2. Identify whether the target is an executed-map, limiting, or implicit
   derivative.
3. Derive the analytic or series result and its assumptions.
4. Check primal convergence, residual, width, finiteness, and conditioning.
5. Compare the custom rule to independent central finite differences of the
   converged mathematical branch.
6. Test rejection cases: multiple roots, nonsmooth residuals, nonconvergence,
   invalid brackets, and near-zero slopes.
7. Make sensitivity claims fail closed: an unsupported sensitivity must not
   look like an accepted finite result.

## Where the claim stops

A fail-closed certificate supports a local sensitivity for the declared branch
and fixture. It does not establish global uniqueness, physical adequacy, or
identifiability. A value-first solver can remain scientifically useful while
making no derivative claim.

## Connected ideas

See [](../../20-methods/change-constraints-evolution/rootfinding.md),
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md),
[](./auditing-derivatives.md), and
[](../investigations/root-values-and-sensitivities.md).
