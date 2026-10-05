"""Figure for the Mie validation: docs/60-validation/figures/optics-mie.png.

    .venv/bin/python scripts/plot_mie_validation.py

Top: Q_ext, Q_sca, g of jaxstro.optics.mie against size parameter x for five refractive
indices. Bottom: relative error against miepython 3.3.0 (lines) and against the 50-digit
mpmath evaluation (markers), with the 1e-10 tolerance (Anna, 2026-10-04). Reference data:
tests/validation/data/optics (scripts/build_mie_reference.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jaxstro.optics.mie import mie_efficiencies, mie_term_counts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "validation" / "data" / "optics"
OUT = ROOT / "docs" / "60-validation" / "figures" / "optics-mie.png"


def _run(m, x):
    nt, ns = mie_term_counts(jnp.asarray(m), jnp.asarray(x))
    return mie_efficiencies(jnp.asarray(m), jnp.asarray(x), n_terms=nt, n_start=ns)


def main() -> None:
    sweep = np.load(DATA / "mie_miepython_sweep.npz")
    rows = json.loads((DATA / "mie_mpmath_50digits.json").read_text())["rows"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    names = ("q_ext", "q_sca", "g")
    labels = ("Q_ext", "Q_sca", "g = <cos Theta>")
    for idx, m in enumerate(np.unique(sweep["m"])):
        s = sweep["m"] == m
        x = sweep["x"][s]
        out = {}
        for lo, hi in ((0, 10), (10, 1e3), (1e3, 1e5)):  # chunk by x: static series per chunk
            c = (x >= lo) & (x < hi)
            r = _run(np.full(c.sum(), m), x[c])
            for k in names:
                out.setdefault(k, []).append(np.asarray(getattr(r, k)))
        color = f"C{idx}"
        for j, k in enumerate(names):
            ours = np.concatenate(out[k])
            axes[0, j].semilogx(x, ours, color=color, lw=1, label=f"m = {m.real:g} + {m.imag:g} i")
            ref = sweep[k][s]
            err = np.abs(ours - ref) / (np.maximum(np.abs(ref), 1e-3) if k == "g" else np.abs(ref))
            axes[1, j].loglog(x, np.maximum(err, 1e-17), color=color, lw=0.8)
        for n, kk, xx, *vals in rows:
            if complex(n, kk) != m:
                continue
            r = _run(complex(n, kk), np.array([xx]))
            for j, (k, v) in enumerate(zip(names, vals)):
                ours = float(getattr(r, k)[0])
                err = abs(ours - v) / (max(abs(v), 1e-3) if k == "g" else abs(v))
                axes[1, j].loglog(xx, max(err, 1e-17), "o", color=color, ms=5, mec="k")
    for j, lab in enumerate(labels):
        axes[0, j].set_ylabel(lab)
        axes[1, j].axhline(1e-10, color="k", ls="--", lw=1, label="tolerance 1e-10")
        axes[1, j].set_ylabel(f"error in {lab}" + (" / max(|g|, 1e-3)" if j == 2 else " (relative)"))
        axes[1, j].set_xlabel("size parameter x = 2 pi a / lambda")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(["vs miepython 3.3.0 (lines)"], fontsize=8, loc="upper left")
    fig.suptitle("jaxstro.optics.mie against miepython (lines, 200 x per m) and a 50-digit mpmath evaluation "
                 "of Bohren & Huffman eq. 4.53 (markers)", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(OUT)


if __name__ == "__main__":
    main()
