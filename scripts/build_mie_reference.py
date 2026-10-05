"""Build the Mie reference data for tests/validation/test_optics_mie_reference.py.

Run in an environment with miepython and mpmath (not jaxstro dependencies), e.g.

    uv venv /tmp/mievenv && VIRTUAL_ENV=/tmp/mievenv uv pip install miepython mpmath numpy
    /tmp/mievenv/bin/python scripts/build_mie_reference.py

Writes tests/validation/data/optics/:

- mie_mpmath_50digits.json: Q_ext, Q_sca, g at 50 digits for a sample of (m, x), from
  Bohren & Huffman (1983) eq. 4.53 with Riccati-Bessel functions evaluated directly by mpmath
  (psi_n(z) = sqrt(pi z / 2) J_{n+1/2}(z), chi_n(x) = -sqrt(pi x / 2) Y_{n+1/2}(x)), no
  recurrences, summed to N = x + 4.05 x^{1/3} + 2 + 10 terms (adding 20 more changes Q_ext by
  < 1.4e-15). The independent reference for the 1e-10 tolerance (Anna, 2026-10-04).
- mie_mpmath_zeros.json (--zeros; needs mpmath only): the same 50-digit evaluation at
  zeros of psi_n(x) with n < x (x = pi, 2 pi, 10 pi; first zeros of psi_1 and psi_5),
  where building psi_n from downward ratios alone loses all precision (2026-10-05).
- mie_miepython_sweep.npz: miepython's efficiencies on 5 m x 200 log-spaced x in
  [1e-3, 1e4]; for |m| x < 0.1, where miepython switches to a small-sphere approximation,
  the series from miepython's own Mie coefficients. A cross-check over the whole domain;
  miepython stops at x + 4.05 x^{1/3} + 2 terms, so its Q_ext is truncated at ~1e-9 for
  weakly absorbing large spheres.

Convention: m = n + i k with k >= 0 here; miepython uses n - i k.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import mpmath as mp
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "validation" / "data" / "optics"
M_SET = [(1.33, 1e-8), (1.5, 0.01), (1.7, 0.3), (3.0, 4.0), (10.0, 10.0)]


def _psi(n, z):
    return mp.sqrt(mp.pi * z / 2) * mp.besselj(n + mp.mpf(1) / 2, z)


def _chi(n, x):
    return -mp.sqrt(mp.pi * x / 2) * mp.bessely(n + mp.mpf(1) / 2, x)


def mie_mpmath(m: complex, x: float, extra: int = 10) -> tuple[float, float, float]:
    mp.mp.dps = 50
    m, x = mp.mpc(m), mp.mpf(x)
    y = m * x
    n_terms = int(float(x) + 4.05 * float(x) ** (1 / 3) + 2) + extra
    ext = sca = asy = mp.mpf(0)
    a_prev = b_prev = None
    psx, psy, chx = [_psi(0, x)], [_psi(0, y)], [_chi(0, x)]
    for n in range(1, n_terms + 1):
        psx.append(_psi(n, x))
        psy.append(_psi(n, y))
        chx.append(_chi(n, x))
        xi, xi_prev = psx[n] - 1j * chx[n], psx[n - 1] - 1j * chx[n - 1]
        dpx = psx[n - 1] - n * psx[n] / x
        dpy = psy[n - 1] - n * psy[n] / y
        dxi = xi_prev - n * xi / x
        a = (m * psy[n] * dpx - psx[n] * dpy) / (m * psy[n] * dxi - xi * dpy)
        b = (psy[n] * dpx - m * psx[n] * dpy) / (psy[n] * dxi - m * xi * dpy)
        ext += (2 * n + 1) * mp.re(a + b)
        sca += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
        if a_prev is not None:
            asy += (n - 1) * (n + 1) / mp.mpf(n) * mp.re(a_prev * mp.conj(a) + b_prev * mp.conj(b))
        asy += (2 * n + 1) / mp.mpf(n * (n + 1)) * mp.re(a * mp.conj(b))
        a_prev, b_prev = a, b
    q_ext, q_sca = 2 / x**2 * ext, 2 / x**2 * sca
    return float(q_ext), float(q_sca), float(4 / (x**2 * q_sca) * asy)


def _miepython_series(m: complex, x: float) -> tuple[float, float, float]:
    a, b = miepython.coefficients(np.conj(m), x)
    n = np.arange(1, len(a) + 1)
    q_ext = 2 / x**2 * np.sum((2 * n + 1) * np.real(a + b))
    q_sca = 2 / x**2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2))
    asy = np.sum(n[:-1] * (n[:-1] + 2) / (n[:-1] + 1) * np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])))
    asy += np.sum((2 * n + 1) / (n * (n + 1)) * np.real(a * np.conj(b)))
    return float(q_ext), float(q_sca), float(4 / (x**2 * q_sca) * asy)


def zeros() -> None:
    mp.mp.dps = 50
    xs = [mp.pi, 2 * mp.pi, 10 * mp.pi,
          mp.findroot(lambda z: _psi(1, z), 4.49), mp.findroot(lambda z: _psi(5, z), 9.36)]
    ms = [(1.666, 0.03114), (1.5, 0.01), (1.7, 0.3)]
    # Evaluated at the float x that jaxstro sees (within ~1e-16 of the zero).
    rows = [[mr, mi, float(x), *mie_mpmath(complex(mr, mi), float(x))] for mr, mi in ms for x in xs]
    (OUT / "mie_mpmath_zeros.json").write_text(json.dumps({
        "source": "scripts/build_mie_reference.py --zeros (Bohren & Huffman eq. 4.53, mpmath, 50 digits)",
        "mpmath": mp.__version__, "columns": ["n", "k", "x", "q_ext", "q_sca", "g"], "rows": rows,
    }, indent=1))
    print(f"wrote {OUT / 'mie_mpmath_zeros.json'}: {len(rows)} cases")


def main() -> None:
    import miepython

    OUT.mkdir(parents=True, exist_ok=True)
    sample_x = [1e-3, 0.1, 1.0, 10.0, 100.0, 188.96523396912076]
    cases = [(mr, mi, x) for mr, mi in M_SET for x in sample_x]
    cases += [(1.33, 1e-8, 1000.0), (1.5, 0.01, 1000.0), (1.33, 1e-8, 1552.225357427048)]
    rows = [[mr, mi, x, *mie_mpmath(complex(mr, mi), x)] for mr, mi, x in cases]
    (OUT / "mie_mpmath_50digits.json").write_text(json.dumps({
        "source": "scripts/build_mie_reference.py (Bohren & Huffman eq. 4.53, mpmath, 50 digits)",
        "mpmath": mp.__version__, "columns": ["n", "k", "x", "q_ext", "q_sca", "g"], "rows": rows,
    }, indent=1))
    ms = np.array([complex(mr, mi) for mr, mi in M_SET])
    m_grid, x_grid = (v.ravel() for v in np.meshgrid(ms, np.geomspace(1e-3, 1e4, 200), indexing="ij"))
    q_ext, q_sca, _, g = miepython.efficiencies_mx(np.conj(m_grid), x_grid)
    small = np.abs(m_grid) * x_grid < 0.1
    for i in np.flatnonzero(small):
        q_ext[i], q_sca[i], g[i] = _miepython_series(m_grid[i], x_grid[i])
    np.savez(OUT / "mie_miepython_sweep.npz", m=m_grid, x=x_grid, q_ext=q_ext, q_sca=q_sca, g=g,
             series_below_0p1=small, miepython=miepython.__version__)
    print(f"wrote {OUT}: {len(rows)} mpmath cases, {x_grid.size} miepython cases")


if __name__ == "__main__":
    zeros() if "--zeros" in sys.argv else main()
