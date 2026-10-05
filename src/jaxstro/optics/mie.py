"""Mie scattering by homogeneous spheres: Q_ext, Q_sca, Q_abs and g = <cos Theta>.

Bohren & Huffman (1983, BHMIE) with two changes for accuracy: the logarithmic derivative
D_n(m x) by downward recurrence from n_start = max(N_stop, |m x|) + 15 + 2 sqrt|m x| (the
extra margin converges weakly absorbing large spheres), psi_n(x) from the downward ratio
psi_n / psi_{n-1} (no cancellation at small x; as in Wiscombe 1980 and miepython), chi_n(x) by
upward recurrence (stable: chi grows), and

    a_n = [(D_n/m + n/x) psi_n - psi_{n-1}] / [(D_n/m + n/x) xi_n - xi_{n-1}]
    b_n = [(m D_n + n/x) psi_n - psi_{n-1}] / [(m D_n + n/x) xi_n - xi_{n-1}],  xi = psi - i chi,

    Q_ext = (2/x^2) sum (2n+1) Re(a_n + b_n),   Q_sca = (2/x^2) sum (2n+1)(|a_n|^2 + |b_n|^2),
    g Q_sca = (4/x^2) sum [n(n+2)/(n+1) Re(a_n a*_{n+1} + b_n b*_{n+1})
                           + (2n+1)/(n(n+1)) Re(a_n b*_n)].

Terms run to N_stop(x) = x + 4 x^{1/3} + 2 (Wiscombe 1980). Under jit the series has a static
length `n_terms` (>= N_stop of the largest x in the batch) and terms beyond each element's
N_stop are zero; past N_stop the recurrences are frozen, so no element overflows (the upward
psi, chi recurrence diverges for n >> x) and gradients stay finite. D_n is recurred from the
static `n_start` in two scans, keeping only n <= n_terms in memory: memory is n_terms per
element, time n_start per element. Use `mie_term_counts` on the host to size both.

Differentiable with respect to the refractive index and the size parameter (fixed-length
scans; the term mask is a step in x whose masked terms are below round-off).
Design and decisions: docs/plans/2026-10-04-mie-scattering-design.md.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex


class MieEfficiencies(NamedTuple):
    q_ext: Float[Array, "..."]
    q_sca: Float[Array, "..."]
    q_abs: Float[Array, "..."]
    g: Float[Array, "..."]  # asymmetry parameter <cos Theta>


def n_stop(x: float) -> int:
    """Wiscombe's (1980) number of terms for size parameter x."""
    return int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)


def mie_term_counts(m, x) -> tuple[int, int]:
    """Static (n_terms, n_start) for concrete arrays m, x: n_terms = N_stop(max x),
    n_start = max(n_terms, max |m x|) + 15 + 2 sqrt(max |m x|). Bohren & Huffman's
    max(N_stop, |m x|) + 15 leaves the downward recurrence of D_n unconverged for weakly
    absorbing large spheres (m = 1.33 + 1e-8 i: Q_ext off by 2.5e-5 at x = 100, 9.7e-4 at
    x = 955, 1.4e-4 at x = 1e4); the 2 sqrt|m x| margin makes it converge to round-off
    (measured 2026-10-04)."""
    x_max = float(jnp.max(jnp.asarray(x)))
    y_max = float(jnp.max(jnp.abs(jnp.asarray(m) * jnp.asarray(x))))
    n_terms = n_stop(x_max)
    return n_terms, max(n_terms, math.ceil(y_max)) + 15 + math.ceil(2.0 * math.sqrt(y_max))


def mie_efficiencies(
    m: Complex[Array, "..."], x: Float[Array, "..."], *, n_terms: int, n_start: int
) -> MieEfficiencies:
    """Efficiencies and asymmetry for refractive index m (n + i k, k >= 0) and size parameter
    x = 2 pi a / lambda, broadcast elementwise (module doc). `n_terms` and `n_start` are
    static; `mie_term_counts` gives the values for a batch."""
    if n_start < n_terms:
        raise ValueError("n_start must be >= n_terms")
    m, x = jnp.broadcast_arrays(jnp.asarray(m, dtype=jnp.complex128), jnp.asarray(x, dtype=jnp.float64))
    shape = x.shape
    m, x = m.reshape(-1), x.reshape(-1)
    y = m * x
    stop = jnp.floor(x + 4.0 * jnp.cbrt(x) + 2.0)

    # D_n(y) downward: carry from n_start to n_terms + 1, then keep D_n for n = n_terms..1.
    def down(d, n):
        return n / y - 1.0 / (d + n / y), None

    d_top, _ = jax.lax.scan(down, jnp.zeros_like(y), jnp.arange(n_start, n_terms, -1, dtype=jnp.float64))

    def down_keep(d, n):
        d_prev = n / y - 1.0 / (d + n / y)  # D_{n-1}
        return d_prev, d_prev

    # Step n gives D_{n-1}: d_rev[k] = D_{n_terms - 1 - k}, k = 0 .. n_terms - 1 (ends at D_0).
    _, d_rev = jax.lax.scan(down_keep, d_top, jnp.arange(n_terms, 0, -1, dtype=jnp.float64))
    d_n = jnp.concatenate([d_rev[::-1][1:], d_top[None, :]], axis=0)  # D_1 .. D_{n_terms}

    # psi_n(x) = R_n psi_{n-1}, with R_n = psi_n / psi_{n-1} by downward recurrence
    # R_n = 1 / ((2n + 1)/x - R_{n+1}) (stable), from the same start. The upward three-term
    # recurrence for psi cancels catastrophically for small x (relative error ~ 1/x^(2n)):
    # g at x = 1e-3 was off by 3.8e-4 (2026-10-04).
    def ratio(r, n):
        return 1.0 / ((2.0 * n + 1.0) / x - r), None

    r_top, _ = jax.lax.scan(ratio, jnp.zeros_like(x), jnp.arange(n_start, n_terms, -1, dtype=jnp.float64))

    def ratio_keep(r, n):
        r_n = 1.0 / ((2.0 * n + 1.0) / x - r)  # R_n from R_{n+1}
        return r_n, r_n

    _, r_rev = jax.lax.scan(ratio_keep, r_top, jnp.arange(n_terms, 0, -1, dtype=jnp.float64))
    r_n = r_rev[::-1]  # R_1 .. R_{n_terms}

    def up(carry, inputs):
        psi0, psi1, chi0, chi1, a_prev, b_prev, ext, sca, asy = carry
        n, d, r = inputs
        active = n <= stop
        psi = r * psi1
        chi = (2.0 * n - 1.0) / x * chi1 - chi0
        xi, xi1 = psi - 1j * chi, psi1 - 1j * chi1
        da, db = d / m + n / x, m * d + n / x
        a = jnp.where(active, (da * psi - psi1) / (da * xi - xi1), 0.0)
        b = jnp.where(active, (db * psi - psi1) / (db * xi - xi1), 0.0)
        ext = ext + (2.0 * n + 1.0) * jnp.real(a + b)
        sca = sca + (2.0 * n + 1.0) * (jnp.abs(a) ** 2 + jnp.abs(b) ** 2)
        asy = asy + jnp.where(
            n > 1.0, (n - 1.0) * (n + 1.0) / n * jnp.real(a_prev * jnp.conj(a) + b_prev * jnp.conj(b)), 0.0
        ) + (2.0 * n + 1.0) / (n * (n + 1.0)) * jnp.real(a * jnp.conj(b))
        # Freeze the recurrences past N_stop so the upward psi, chi stay finite.
        keep = lambda new, old: jnp.where(active, new, old)  # noqa: E731
        carry = (keep(psi1, psi0), keep(psi, psi1), keep(chi1, chi0), keep(chi, chi1), a, b, ext, sca, asy)
        return carry, None

    zeros = jnp.zeros_like(x)
    czeros = jnp.zeros_like(y)
    carry0 = (jnp.cos(x), jnp.sin(x), -jnp.sin(x), jnp.cos(x), czeros, czeros, zeros, zeros, zeros)
    (_, _, _, _, _, _, ext, sca, asy), _ = jax.lax.scan(
        up, carry0, (jnp.arange(1, n_terms + 1, dtype=jnp.float64), d_n, r_n)
    )
    q_ext = 2.0 / x**2 * ext
    q_sca = 2.0 / x**2 * sca
    g = 4.0 / (x**2 * q_sca) * asy
    out = MieEfficiencies(q_ext, q_sca, q_ext - q_sca, g)
    return MieEfficiencies(*(v.reshape(shape) for v in out))


__all__ = ["MieEfficiencies", "mie_efficiencies", "mie_term_counts", "n_stop"]
