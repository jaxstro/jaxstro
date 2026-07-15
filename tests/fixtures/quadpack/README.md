# Pinned Netlib QUADPACK Gauss-Kronrod sources

These six files were retrieved verbatim from
`https://www.netlib.org/quadpack/` on 2026-07-15. They are provenance fixtures,
not independent numerical oracles. The independent checks are analytic moment
identities in `tests/validation/test_quad_gk_tables.py`.

| File | SHA-256 |
| --- | --- |
| `dqk15.f` | `471664a145516508ce6147f1a425d7d3d7f3afe50fd9584ffff020749f534276` |
| `dqk21.f` | `4f89799878f42504549e952bf999d5e80e5253096a77dfe074f6353a92cceed7` |
| `dqk31.f` | `72017e6f65cd3ee9b49ed067962fd0254d077fa735821929aa46cdba30b75e9d` |
| `dqk41.f` | `3cc35dfb473cd7ccd7525307a698bfab99f4e3c509f541fa7afe706e80e92489` |
| `dqk51.f` | `a0ff6ad1b7cf30a4454494ed9fc1eddfefdb8fa12e8f0d8c173444616b13b55c` |
| `dqk61.f` | `39decfe58e8892e38e58827335306b0ebbf04d6fb4d701f68edbac8857ee0f72` |

Run `python scripts/build_quadpack_gk_fixture.py --check` to verify source
hashes and both generated artifacts.
