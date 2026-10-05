"""Write tests/fixtures/pc00/ocp_reference.json: PC00 eq. (16) and its first two
Gamma derivatives from a 60-digit mpmath evaluation of the Table I "Ref. [6]" row.
Run: python scripts/gen_pc00_reference.py  (needs mpmath, the `reference` group)."""
import json
import pathlib

import mpmath as mp

mp.mp.dps = 60
A1, A2, B1, B2, B3, B4 = (mp.mpf(x) for x in ("-0.9070", "0.62954", "4.56e-3", "211.6", "-1.0e-4", "4.62e-3"))
A3 = -mp.sqrt(3) / 2 - A1 / mp.sqrt(A2)


def f(g):
    sg = mp.sqrt(g)
    return (A1 * (mp.sqrt(g * (A2 + g)) - A2 * mp.log(sg / mp.sqrt(A2) + mp.sqrt(1 + g / A2)))
            + 2 * A3 * (sg - mp.atan(sg)) + B1 * (g - B2 * mp.log(1 + g / B2))
            + B3 / 2 * mp.log(1 + g * g / B4))


grid = [mp.mpf(10) ** (mp.mpf(k) / 4) for k in range(-32, 10)] + [mp.mpf(175)]
rows = []
for g in grid:
    rows.append({"gamma": mp.nstr(g, 25), "f": mp.nstr(f(g), 25),
                 "f1": mp.nstr(mp.diff(f, g), 25), "f2": mp.nstr(mp.diff(f, g, 2), 25)})
# Debye-Huckel approach: c(gamma) = (f / f_DH - 1) / sqrt(gamma)
dh = []
for e in range(-16, -1):
    g = mp.mpf(10) ** e
    dh.append({"gamma": mp.nstr(g, 25), "f_over_dh_minus_1": mp.nstr(f(g) / (-g ** mp.mpf(1.5) / mp.sqrt(3)) - 1, 25)})
out = {"source": "PC00 eq. (16), Table I row Ref. [6]; mpmath 60 digits", "rows": rows, "dh": dh}
path = pathlib.Path(__file__).resolve().parents[1] / "tests/fixtures/pc00/ocp_reference.json"
path.write_text(json.dumps(out, indent=1) + "\n")
