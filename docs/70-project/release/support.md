---
title: Qualified support
description: Release-qualified platform boundary for Jaxstro 0.1.0.
---

## Qualified support for 0.1.0

The release qualification target is CPython 3.13 on Ubuntu x86_64 CPU with
JAX_ENABLE_X64=1. The release mirror exercises this configuration.

Installation requires Python >=3.13. GPU, TPU, macOS, Windows, and Python
versions other than CPython 3.13 are not a qualified support claim. JAX may
make an installation run on another backend; that is not evidence that Jaxstro
numerical contracts have been qualified there.

`jaxstro.quad` is public but experimental. Its method pages define accepted
domains, statuses, replay boundary, and non-claims; it is not in the qualified
core.
