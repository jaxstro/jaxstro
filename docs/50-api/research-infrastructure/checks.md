---
title: Numerical checks
---

# Numerical checks

## Owner import path

`jaxstro.numerics.checks`

## Purpose

Finiteness, monotonicity, range, sign, and concrete-validation helpers.

## Public records and callables

`try_concrete_bool`, `is_finite`, `all_finite`, `assert_all_finite`,
`is_monotonic`, `is_monotonic_increasing`, `is_monotonic_decreasing`,
`assert_monotonic`, `in_range`, `all_in_range`, `all_positive`,
`all_non_negative`, `assert_in_range`, `assert_positive`, and
`assert_non_negative`.

## Shape and dtype expectations

Predicates accept scalar or array values and reduce according to each helper.
Monotonic checks operate along the documented one-dimensional sequence axis.

## JAX transforms and AD classification

Predicates compose with JIT as values. Assertion helpers perform eager concrete
validation where possible and make no AD claim.

## Failure behavior

Assertion helpers raise on concrete contract violations. Traced values that
cannot be converted to a host boolean are not falsely certified.

## Contract and evidence links

See [](../../60-validation/validation.md) and [](./testing.md).

## Canonical import example

```python
from jaxstro.numerics.checks import assert_all_finite
```
