# src/jaxstro/params/__init__.py
"""
Parameterization utilities: free/fixed marking and PyTree <-> vector bridges.

This subpackage adapts structured Equinox models to the flat-vector interface
expected by optimizers (optax) and samplers (numpyro/blackjax), while keeping
the mapping pure, static, and differentiable.
"""

from .parameterization import Parameterization

__all__ = ["Parameterization"]
