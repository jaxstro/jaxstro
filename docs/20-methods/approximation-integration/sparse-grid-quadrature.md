---
title: Sparse-grid quadrature
description: >-
  Smolyak constructions, hierarchical surpluses, dimension-adaptive
  refinement, exact node reuse, and honest stopping evidence.
---

## The question this method answers

How can we integrate a smooth function of several variables without paying the
full cost of a high-order tensor product? Sparse grids replace one large
Cartesian rule by a structured sum of smaller tensor increments. They are most
useful when the integrand has mixed smoothness or when only a few coordinate
directions need high resolution.

:::{tip}
Use `Smolyak` when a fixed, reproducible level or a known anisotropy is part of
the computation plan. Use `AdaptiveSmolyak` when the important directions are
not known in advance and the frontier surplus is acceptable stopping evidence.
:::

## Why tensor products become expensive

Suppose a one-dimensional rule uses \(n\) nodes. Its \(d\)-dimensional tensor
product uses

```{math}
:label: eq-sparse-tensor-cost
N_{\mathrm{tensor}}=n^d
```

nodes. This is appropriate when every direction needs the same resolution, but
it becomes wasteful when most high-order interactions contribute little.
Sparse grids organize the calculation by *incremental resolution* instead.

:::{important}
Sparse grids reduce a particular dimensional scaling problem; they do not make
high-dimensional integration universally easy. Discontinuities, narrow
features, and strong interactions can still require many indices or a
different method family.
:::

## Build a hierarchy in one dimension

Let \(Q_{\ell}\) be the nested Clenshaw-Curtis rule at level \(\ell\). Jaxstro
uses the midpoint as the base rule,

```{math}
:label: eq-sparse-level-one
Q_1[f]=f\left(\frac{1}{2}\right),
```

and uses \(2^{\ell-1}+1\) Clenshaw-Curtis nodes for \(\ell\geq 2\). Define the
hierarchical difference

```{math}
:label: eq-sparse-hierarchical-difference
\Delta_{\ell}=Q_{\ell}-Q_{\ell-1},
\qquad Q_0=0.
```

The telescoping identity

```{math}
:label: eq-sparse-telescoping
Q_L=\sum_{\ell=1}^{L}\Delta_{\ell}
```

shows why the differences are useful: each \(\Delta_{\ell}\) asks what changed
when one more resolution level was introduced.

## Lift the hierarchy to several dimensions

For a multi-index
\(\boldsymbol{\ell}=(\ell_1,\ldots,\ell_d)\), define

```{math}
:label: eq-sparse-tensor-increment
\Delta_{\boldsymbol{\ell}}
=
\Delta_{\ell_1}\otimes\cdots\otimes\Delta_{\ell_d}.
```

A downward-closed index set \(\mathcal{I}\) produces the Smolyak formula

```{math}
:label: eq-sparse-smolyak
A_{\mathcal{I}}[f]
=
\sum_{\boldsymbol{\ell}\in\mathcal{I}}
\Delta_{\boldsymbol{\ell}}[f].
```

Downward closure means that whenever \(\boldsymbol{\ell}\in\mathcal{I}\) and
\(\ell_j>1\), the immediate predecessor
\(\boldsymbol{\ell}-\boldsymbol{e}_j\) also belongs to \(\mathcal{I}\). The
formula never claims a high-resolution interaction without including the
lower-resolution information on which it depends.

### Fixed isotropic and anisotropic sets

`Smolyak(level=L)` uses the isotropic set

```{math}
:label: eq-sparse-isotropic-set
\sum_{j=1}^{d}(\ell_j-1)\leq L-1.
```

Static positive anisotropy weights \(w_j\) change the budget to

```{math}
:label: eq-sparse-anisotropic-set
\sum_{j=1}^{d}w_j(\ell_j-1)\leq L-1.
```

A larger \(w_j\) makes refinement along axis \(j\) more expensive. These
weights are algorithm configuration, not differentiable scientific
parameters.

## Let the frontier choose the next direction

`AdaptiveSmolyak` begins with a downward-closed set and considers admissible
forward neighbors. A candidate is admissible only after every valid immediate
backward neighbor has been accepted. Its selection profit is

```{math}
:label: eq-sparse-profit
P_{\boldsymbol{\ell}}
=
\frac{
\left\|\Delta_{\boldsymbol{\ell}}[f]\right\|
}{
\max\left(1,\Delta N_{\boldsymbol{\ell}}\right)
},
```

where \(\Delta N_{\boldsymbol{\ell}}\) is the number of genuinely new nodes.
The largest profit is selected, with lexicographic tie breaking for
determinism.

The reported stopping evidence is the active-frontier surplus sum,

```{math}
:label: eq-sparse-frontier-error
E_{\mathrm{frontier}}
=
\sum_{\boldsymbol{\ell}\in\mathcal{F}}
\left\|\Delta_{\boldsymbol{\ell}}[f]\right\|.
```

:::{warning}
`ErrorKind.SPARSE_GRID_SURPLUS` is convergence evidence for the executed
hierarchy. It is not a universal upper bound on
\(\left|I[f]-A_{\mathcal{I}}[f]\right|\). A localized feature that has not been
sampled can make the frontier look smaller than the true error.
:::

## Reuse nested nodes exactly

Clenshaw-Curtis nodes can recur at several levels and in several tensor
increments. Jaxstro identifies each one-dimensional node by its reduced
integer dyadic-angle identity before a floating coordinate is constructed. A
multidimensional node identity is the tuple of those axis identities.

This gives three auditable properties:

1. repeated mathematical nodes are evaluated once;
2. work counts report unique physical evaluations; and
3. node reuse does not depend on approximate floating-point equality.

## Use the public API

```python
import jax.numpy as jnp

from jaxstro import quad

domain = quad.Hyperrectangle(jnp.zeros(4), jnp.ones(4))

fixed = quad.integrate(
    lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
    domain,
    method=quad.Smolyak(level=5),
    epsabs=1.0e-8,
    epsrel=1.0e-8,
    max_evaluations=4096,
    max_indices=1024,
    max_frontier=1024,
    max_nodes=4096,
    gradient="stop",
)

adaptive = quad.integrate(
    lambda x: jnp.exp(-8.0 * x[:, 0]),
    domain,
    method=quad.AdaptiveSmolyak(initial_level=1),
    epsabs=1.0e-8,
    epsrel=1.0e-8,
    max_evaluations=512,
    max_indices=8,
    max_frontier=33,
    max_nodes=512,
    gradient="stop",
)
```

`max_indices`, `max_frontier`, and `max_nodes` are distinct static capacities.
`max_evaluations` is the logical integrand-evaluation budget. For adaptive
integration, the declared frontier capacity must satisfy

```{math}
:label: eq-sparse-frontier-capacity
N_{\mathrm{frontier,max}}
\geq
1+dN_{\mathrm{indices,max}}.
```

## Astrophysical research patterns

Sparse grids are a strong candidate when a deterministic integral has a small
to moderate number of continuous nuisance dimensions:

- integrating a smooth selection function over distance, extinction, and
  population parameters;
- marginalizing a deterministic likelihood approximation over a few
  calibration parameters;
- integrating a smooth phase-space observable when only one or two coordinates
  require fine resolution; and
- constructing reproducible reference integrals for a lower-fidelity emulator
  or randomized estimator.

For example, an extinction-sensitive selection function may vary sharply with
one dust coordinate but slowly with several calibration coordinates.
Dimension-adaptive profit can discover that imbalance instead of refining
every tensor direction equally.

:::{note}
The method is domain agnostic. An astrophysical interpretation, physical
units, and an acceptance threshold remain the caller's responsibility.
Quantity-mode multidimensional certification is deferred to Phase B4.
:::

## What has been validated

The B2 gate checks analytic product, exponential, rotated-quadratic, localized
Gaussian, and strongly anisotropic integrands. Fixed-grid truth checks cover
dimensions \(2\), \(4\), \(8\), and \(16\) where declared by each case.
Adaptive dimension-\(16\) evidence is deliberately anisotropic; it is not a
claim that every smooth sixteen-dimensional integral is cheap.

The current methods support eager execution, `jax.jit`, and `jax.vmap` in
`gradient="stop"` mode for real, array, and complex payloads. Replay
derivatives, quantity certification, backend-wide performance claims, and
cross-method memory optimization remain Phase B4 work.

:::{warning}
The level-\(5\), dimension-\(16\) fixed construction is excluded from the B2
certification control because its dense per-index increment matrix creates a
multi-gigabyte memory path. The certified fixed dimension-\(16\) control uses
level \(4\) and verifies improvement over level \(3\). B4 owns a more compact
representation and the final memory envelope.
:::

## Audit a sparse-grid result

Record at least:

- the method declaration and anisotropy, if any;
- all four capacities and the active JAX dtype;
- the returned status, unique evaluation count, and accepted refinement count;
- the frontier-surplus norm and requested tolerance;
- a level or capacity refinement check against independent truth when
  possible; and
- whether the scientific integrand is plausibly smooth in mixed coordinate
  directions.

The implementation and validation contracts are linked from the
[quadrature API page](../../50-api/approximation-integration/quad.md).
