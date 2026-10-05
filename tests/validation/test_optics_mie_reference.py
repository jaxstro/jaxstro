"""jaxstro.optics.mie against independent references (Anna, 2026-10-04).

- 50-digit evaluation of Bohren & Huffman eq. 4.53 with mpmath (Bessel functions, no
  recurrences): the reference for the agreed tolerance. Max relative error in Q_ext and
  Q_sca <= 1e-10; |dg| <= 1e-10 max(|g|, 1e-3). Measured 2026-10-04: 1.9e-13, 5.8e-15, 7.3e-13.
- miepython 3.3.0 over 5 m x 200 x in [1e-3, 1e4]: cross-check of the whole domain. Q_sca
  and g to the same bounds (measured 7.9e-12, 7.5e-12); miepython truncates its series at
  x + 4.05 x^{1/3} + 2 terms, so its Q_ext differs from ours by up to 7.8e-10 for weakly
  absorbing large spheres (m = 1.33 + 1e-8 i, x = 1552), where the 50-digit value sides with
  jaxstro; bound 1.5e-9.
Data: tests/validation/data/optics, built by scripts/build_mie_reference.py.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.optics.mie import mie_efficiencies, mie_term_counts

DATA = Path(__file__).resolve().parent / "data" / "optics"


def _run(m, x):
    nt, ns = mie_term_counts(jnp.asarray(m), jnp.asarray(x))
    return mie_efficiencies(jnp.asarray(m), jnp.asarray(x), n_terms=nt, n_start=ns)


def _g_error(g, g_ref):
    return np.abs(g - g_ref) / np.maximum(np.abs(g_ref), 1e-3)


def test_against_50_digit_reference():
    rows = json.loads((DATA / "mie_mpmath_50digits.json").read_text())["rows"]
    for n, k, x, q_ext, q_sca, g in rows:
        r = _run(complex(n, k), np.array([x]))
        assert abs(float(r.q_ext[0]) / q_ext - 1) <= 1e-10, (n, k, x)
        assert abs(float(r.q_sca[0]) / q_sca - 1) <= 1e-10, (n, k, x)
        assert float(_g_error(np.asarray(r.g), g)[0]) <= 1e-10, (n, k, x)


@pytest.mark.parametrize("x_range", [(0, 10), (10, 1e3), (1e3, 1e5)])
def test_against_miepython_sweep(x_range):
    d = np.load(DATA / "mie_miepython_sweep.npz")
    for m in np.unique(d["m"]):
        s = (d["m"] == m) & (d["x"] >= x_range[0]) & (d["x"] < x_range[1])
        r = _run(d["m"][s], d["x"][s])
        assert np.max(np.abs(np.asarray(r.q_sca) / d["q_sca"][s] - 1)) <= 1e-10
        assert np.max(_g_error(np.asarray(r.g), d["g"][s])) <= 1e-10
        assert np.max(np.abs(np.asarray(r.q_ext) / d["q_ext"][s] - 1)) <= 1.5e-9
