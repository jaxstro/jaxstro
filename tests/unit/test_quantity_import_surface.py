"""Tests for the quantity public import surface."""

import jaxstro
import jaxstro.quantity as q
from jaxstro import quantity


def test_quantity_module_imports():
    assert quantity is q
    assert "quantity" in jaxstro.__all__


def test_ergonomic_reexports():
    assert q.Quantity is not None
    assert q.Unit is not None
    assert q.cm.symbol == "cm"
    assert q.Msun.symbol == "Msun"
