# src/jaxstro/jaxconfig.py

"""
Shared JAX configuration helpers for the jaxstro ecosystem.

This module does **not** change any JAX settings on import.

Instead, it provides a small helper that top-level packages
(e.g., gravax, startrax, stellax, nebulax) can call at the very
start of their import path to enable high-precision JAX behavior.

Why this exists
---------------
JAX defaults to 32-bit floats and may choose reduced matmul
precision for performance. For many astrophysical simulations
and stiff ODE problems, this is not sufficient:

    - Energy conservation often requires float64 (∆E/E ≲ 1e-6).
    - Implicit solvers and PN corrections rely on good conditioning.
    - Multidecadal or Gyr integrations amplify float32 errors.

We therefore provide a single, well-documented helper that:

    - Enables 64-bit floats globally (jax_enable_x64 = True).
    - Sets matmul precision to "highest".

Usage pattern
-------------
Each *application-level* package should call this once, as early
as possible, before importing modules that create JAX arrays or
define JIT-compiled functions.

Example: gravax
~~~~~~~~~~~~~~~
In ``gravax/__init__.py``::

    # Configure JAX precision for the entire process
    from jaxstro.jaxconfig import enable_high_precision as _enable_jax_hp

    _enable_jax_hp()
    del _enable_jax_hp  # avoid leaking into public API

    # Now it is safe to import the rest of gravax
    from . import units, dynamics, stellax  # noqa: E402

Example: startrax
~~~~~~~~~~~~~~~~~
In ``startrax/__init__.py``::

    from jaxstro.jaxconfig import enable_high_precision as _enable_jax_hp

    _enable_jax_hp()
    del _enable_jax_hp

    from . import sse, bse  # noqa: E402

Example: scripts and CLIs
~~~~~~~~~~~~~~~~~~~~~~~~~
For a standalone script that uses multiple jaxstro-based packages::

    from jaxstro.jaxconfig import enable_high_precision
    enable_high_precision()

    import gravax
    import startrax
    # ... rest of your script ...

Notes and caveats
-----------------
- JAX configuration flags are **global** within a Python process.
  This helper should be called before any other JAX usage.
- If you need to disable x64 for a particular environment (e.g.,
  training on certain accelerators), you can choose not to call
  this helper and instead rely on JAX's default configuration.
- Environment variables (such as JAX_ENABLE_X64) are an
  alternative for setting these flags *outside* Python. This
  helper is intended for controlled, library-driven configuration.
"""

from jax import config as jax_config


def enable_high_precision() -> None:
    """
    Configure JAX for 64-bit, high-precision numerical work.

    This function should be called **once**, as early as possible
    in your process, before any JAX arrays are created or any
    JIT-compiled functions are defined.

    Effects
    -------
    - Sets ``jax_enable_x64 = True``, enabling float64 throughout.
    - Sets ``jax_default_matmul_precision = "highest"``, requesting
      the highest available precision for matrix multiplications.

    Typical usage
    -------------
    In a top-level package's ``__init__.py``::

        from jaxstro.jaxconfig import enable_high_precision as _enable_jax_hp

        _enable_jax_hp()
        del _enable_jax_hp

        # now import the rest of your package
        from . import core  # noqa: E402

    In a standalone script::

        from jaxstro.jaxconfig import enable_high_precision
        enable_high_precision()

        import gravax, startrax
        # your code here

    Returns
    -------
    None
        The function is called for its side effects only.
    """
    jax_config.update("jax_enable_x64", True)
    jax_config.update("jax_default_matmul_precision", "highest")


__all__ = ["enable_high_precision"]

