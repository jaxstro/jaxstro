---
title: Migrate from UnitSystem to quantities
description: >-
  A boundary-first recipe for adopting jaxstro.quantity while preserving
  backwards-compatible jaxstro.units code.
---

Use `jaxstro.quantity` at public boundaries and keep raw arrays inside kernels.
Do not rewrite a working `UnitSystem` code path just to be modern; migrate where
dimension checks, config readability, or provenance matter.

## Keep existing UnitSystem code working

Legacy code can stay exactly as it is:

```python
from jaxstro import units as U

G = U.STELLAR.G
mass_cgs, radius_cgs, time_cgs = U.STELLAR.to_cgs(1.0, 2.0, 0.5)
```

The compatibility bridge exposes representative quantity units without changing
legacy behavior:

```python
mass_unit, length_unit, time_unit = U.STELLAR.quantity_units
scales = U.STELLAR.quantity_scales
```

No deprecation warnings are emitted.

## Accept quantities at the boundary

```python
import jaxstro.quantity as q

def escape_speed(radius, mass):
    r_cm = radius.to_value(q.cm)
    m_g = mass.to_value(q.g)
    return _escape_speed_cgs(r_cm, m_g) * (q.cm / q.s)
```

Callers can now pass natural units:

```python
escape_speed(1 * q.Rsun, 1 * q.Msun).to(q.km / q.s)
```

## Parse configs explicitly

For config files, parse unit strings once and serialize canonically:

```python
unit = q.parse_unit("Msun/yr")
payload = q.to_dict(q.Quantity(1.0, unit))
```

Unknown symbols fail closed with suggestions. Decimal powers serialize as exact
rational powers when they are accepted.

## Use equivalencies only when requested

Do not silently treat wavelength, frequency, and photon energy as the same
dimension:

```python
wavelength = 500 * q.nm
frequency = wavelength.to(q.Hz, equivalencies=q.equivalencies.spectral())
```

The same explicit rule applies to temperature-energy and mass-energy conversion.
