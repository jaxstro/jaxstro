---
title: Qualified scientific core
description: The deliberately bounded, evidence-complete Jaxstro scientific profile.
---

# Qualified scientific core v1

This profile names the small public surface whose existing contracts have
value/failure boundaries, limitations, and independent validation evidence.

- [`jaxstro.units`](../50-api/physical-representations/units.md) is the static-module
  exception: its qualified contract is ownership/non-ownership, CGS dimensional
  policy, and [scale/conversion/default evidence](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/test_units.py).
  It makes no JAX-transform or numerical-failure claim.
- [`jaxstro.numerics.safeguarded_bracketed_root`](../50-api/change-constraints/rootfinding.md)
  has public analytical-root and typed-failure validation in
  [`tests/validation/test_bracketed_root_algorithms.py`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_bracketed_root_algorithms.py).
- [`jaxstro.numerics.implicit_bracketed_root`](../50-api/change-constraints/rootfinding.md)
  has derivative validation in
  [`tests/validation/test_implicit_root_gradients.py`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_implicit_root_gradients.py).
- [`jaxstro.numerics.universal_kepler_step`](../50-api/change-constraints/kepler.md)
  has fixed-route derivative validation in
  [`tests/validation/test_kepler_gradients.py`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_kepler_gradients.py).

This profile does not qualify every importable callable, a downstream physical
model, automatic differentiation across status or discrete-route changes, or
any GPU, TPU, macOS, Windows, or unlisted Python runtime.

`jaxstro.quad.fixed` and `jaxstro.quad.integrate` remain experimental and are not
promoted by this profile.
