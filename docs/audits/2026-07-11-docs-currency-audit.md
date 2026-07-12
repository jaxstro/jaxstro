# Documentation currency audit — Slice C0

**Status:** inventory and verification complete; no pedagogical page has been
revised in this slice.

## Scope and method

This audit compares the public documentation with the installed `jaxstro` source
after the quantity, spatial-neighbor, Slice-A hardening, and Slice-B provenance
work. It covers `README.md`, the site landing page and navigation, and the 38
Markdown pages under Getting Started, Theory, Architecture, API, How-to, and
Validation.

The checks used four kinds of evidence:

1. every fenced Python block was classified and mapped to the public API it is
   meant to teach;
2. README numerical claims were reproduced in a clean Python subprocess with
   x64 enabled before JAX arrays were created;
3. documented import paths and call signatures were compared with the installed
   package;
4. the MyST HTML build, generated xref data, and rendered HTML were inspected
   separately. A successful parse is not treated as proof that a link or DOM
   contract is correct.

Atmospheres is explicitly an in-progress subsystem. Its honest capability and
backend boundaries are recorded here, but its unfinished Sonora/TLUSTY runtime
work does not block hardening the rest of jaxstro.

`jaxstro.units`, `UnitSystem`, `DEFAULT_UNITS`, and `PhotometricUnits` are the
current canonical ecosystem contracts. `jaxstro.quantity` is implemented, but
ecosystem adoption and any replacement cutover are explicitly deferred until a
separate cross-package program is approved.

The fenced-snippet line numbers and initial finding evidence below refer to the
C0 snapshot at commit `814229e`. Resolution notes are updated as approved page
slices land.

## Inventory

- **40 public-facing sources:** `README.md`, `docs/index.md`, and 38 scoped site
  pages.
- **43 fenced Python blocks** across 12 files.
- **15 standalone executable-intent blocks:** 14 map to runnable installed
  imports/calls; one currently fails because the documented call is wrong.
- **8 continuation blocks:** valid only with state established in an earlier
  block on the same page.
- **20 illustrative, local-data-dependent, or intentional pseudocode blocks:**
  these need an explicit label or complete values before they can be advertised
  as copy-paste examples.

### Fenced-snippet ledger

| Source and line | Class | Installed symbol / expected behavior | Audit result |
| --- | --- | --- | --- |
| `README.md:89` | Executable | `jaxstro.jaxconfig.enable_high_precision`; enable x64 before arrays | Covered by subprocess test |
| `README.md:98` | Executable | `constants`, `units`; solar escape speed and dynamical `G` | Covered; about 617.7 km/s and 0.00450 |
| `README.md:115` | Executable intent, continuation | `sky_tangent`, `galactic_to_equatorial`, `compute_parallax` | **Broken:** `jnp` is inherited and `compute_parallax` omits required `positions` |
| `README.md:145` | Executable | `UnitSystem.to_cgs`, `from_cgs`, `.G`, velocity scale | Public symbols exist; numerical claims have unit-test anchors |
| `README.md:172` | Illustrative | `safe_log`, `safe_exp`, `safe_div` | `x`, `a`, and `b` are placeholders; not standalone |
| `README.md:184` | Executable | `bisect`, `newton`; recover square root of two | Covered by subprocess test |
| `README.md:198` | Continuation | `compensated_sum_array`; recover 2.0 | Compensated result verified; ordinary-sum comment is false on the documented x64 path |
| `README.md:215` | Executable | `Parameterization`, `Exp`, `Sigmoid`; PyTree/vector round trip and gradient | Installed API maps correctly; needs a dedicated executable-doc test in its page slice |
| `README.md:257` | Executable | spatial bin assignment, capacity fill, approximate candidates | Direct submodule imports map correctly; needs a small deterministic docs fixture |
| `README.md:305` | Import-only | coordinate public imports | Imports exist; the parallax summary hides its required positions argument |
| `README.md:322` | Import-only | spatial public imports | Direct `jaxstro.spatial` imports exist |
| `README.md:342` | Import-plus-comments | params public imports | Imports exist; usage is intentionally abbreviated |
| `docs/00-getting-started/index.md:45` | Executable | x64 configuration | Installed API maps correctly |
| `docs/00-getting-started/index.md:68` | Executable | guarded logarithm plus `bisect` | Forward value runs, but the derivative lesson is not a valid bisection contract |
| `docs/00-getting-started/index.md:102` | Continuation | `jax.grad(radius_at_density)` | Numerically matches here only because the target-dependent bracket carries the derivative |
| `docs/40-api/index.md:18` | Executable | top-level public module imports | **Resolved in C6:** `spatial` is an eager top-level attribute and committed member of `jaxstro.__all__`; a clean-process test executes the documented surface |
| `docs/50-howto/quantity-migration.md:16` | Executable | current `UnitSystem` behavior | Installed API maps correctly |
| `docs/50-howto/quantity-migration.md:26` | Continuation | `quantity_units`, `quantity_scales` | Requires the earlier `U` import |
| `docs/50-howto/quantity-migration.md:35` | Continuation/illustrative | quantity boundary conversion | `_escape_speed_cgs` is intentionally supplied by the caller |
| `docs/50-howto/quantity-migration.md:46` | Continuation | quantity-aware `escape_speed` call | Requires the preceding function and import |
| `docs/50-howto/quantity-migration.md:54` | Continuation | parser and scalar serialization | Requires the preceding `q` import |
| `docs/50-howto/quantity-migration.md:67` | Continuation | explicit spectral equivalency | Requires the preceding `q` import |
| `docs/20-architecture/quantity-system.md:23` | Continuation example | `Quantity`, concrete units, basis conversion | `jnp` is not imported in the block |
| `docs/20-architecture/quantity-system.md:62` | Pseudocode | conceptual `Quantity` data model | Ellipsis and unqualified names are intentional |
| `docs/20-architecture/quantity-system.md:68` | Pseudocode | conceptual `Unit` metadata model | Ellipsis and unqualified names are intentional |
| `docs/20-architecture/quantity-system.md:179` | Illustrative | named bases | Requires `q`; expression-only inventory |
| `docs/20-architecture/quantity-system.md:192` | Illustrative | direct unit conversion | `mass` and `radius` are placeholders |
| `docs/20-architecture/quantity-system.md:199` | Illustrative | role-aware basis conversion | domain values are placeholders |
| `docs/20-architecture/quantity-system.md:209` | Illustrative | quantity constants | Requires `q`; expression-only inventory |
| `docs/20-architecture/quantity-system.md:224` | Illustrative | explicit equivalencies | physical values are placeholders |
| `docs/20-architecture/spectra-data-architecture.md` | Interface notation | `AtmosphereParams -> SpectrumResult` boundary | **Resolved in C11f:** rendered as a captioned text code block explicitly identified as interface notation, not Python |
| `docs/20-architecture/spectra-data-architecture.md` | Executable | synthetic `PreparedSpectralGrid`, statuses, JIT, and gradient | **Resolved in C11f:** the standalone artifact-free block is executed verbatim and checks bilinear flux, fail-closed statuses, clamping, and the local derivative |
| `docs/20-architecture/spectra-data-architecture.md` | Local-data example | `AtmosphereLibrary.from_local/select/spectrum` | **Resolved in C11f:** explicitly labelled as requiring local processed artifacts plus the optional data dependencies |
| `docs/20-architecture/spectra-data-architecture.md` | Local-data example | `NewEraBackend.open/prepare`, JIT spectrum evaluation | **Resolved in C11f:** explicitly labelled as requiring a compatible local catalog and Zarr store; opening/preparation remain outside the compiled model |
| `docs/10-theory/bsplines.md` | Executable | fixed-knot basis, evaluation, derivative, and `BSpline1D` parity | **Resolved in C11a:** the standalone block is executed verbatim and asserts shape, partition of unity, parity, and finite derivatives |
| `docs/10-theory/interpolation.md` | Executable | Hermite, natural-cubic, PCHIP, and wrapper shape contracts | **Resolved in C11b:** the standalone block is executed verbatim and asserts measured undershoot, bounds, monotonicity, and parity |
| `docs/10-theory/random.md` | Executable | explicit key ownership, deterministic replay, and three resamplers | **Resolved in C11c:** the standalone block is executed verbatim and checks shapes, ranges, replay, and exact residual counts |
| `docs/10-theory/regular-grid.md` | Executable | generic and bilinear affine interpolation with payload and boundary contracts | **Resolved in C11d:** the standalone block is executed verbatim and checks affine recovery, wrapper parity, clamp, and whole-payload fill |
| `docs/10-theory/linear-algebra.md` | Executable | weighted/unweighted fits, QR/SVD parity, covariance/correlation guard, and diagonal jitter | **Resolved in C11e:** the standalone block is executed verbatim and checks coefficients, finite guarded correlation, selected jitter/success, and positive shifted eigenvalues |
| `docs/10-theory/quantities.md:14` | Executable | basic quantity arithmetic | Installed API maps correctly |
| `docs/10-theory/quantities.md:73` | Illustrative | direct and basis conversion | values come from the earlier example |
| `docs/10-theory/quantities.md:87` | Illustrative | constant metadata/raw-value lookup | Requires the earlier `q` import |
| `docs/10-theory/quantities.md:96` | Illustrative | explicit equivalencies | Requires the earlier `q` import |
| `docs/10-theory/quantities.md:107` | Pseudocode | public-boundary/raw-kernel pattern | `_raw_kernel` intentionally represents caller code |

## Verified findings

| ID | Severity | Finding | Direct evidence | Page-slice action |
| --- | --- | --- | --- | --- |
| C0-001 | P1 | README parallax quick start cannot run | Installed signature is `compute_parallax(positions, distance_pc)`; the README calls only `distance_pc` | **Resolved in C1:** the standalone block passes explicit center-star positions and is executed by `test_readme_coordinate_block_is_standalone_and_executable` |
| C0-002 | P1 | Getting Started teaches an invalid general lesson about differentiating bisection | The dedicated rootfinding page and Slice-A audit classify bisection as branchy/forward-solve-only. The displayed derivative matches only because the analytic, target-dependent upper bracket carries the gradient | **Resolved in C2:** the website now uses a dimensionless Newton solve with a parameter-independent initial guess and checks the root and derivative against analytic and central-FD evidence |
| C0-003 | P2 | Spatial has an inconsistent top-level ownership contract | Before C6, plain `import jaxstro` omitted `spatial` even though direct and `from` imports worked | **Resolved in C6:** approved additive ownership makes `spatial` eager, includes it in `jaxstro.__all__`, and verifies the contract in a clean subprocess |
| C0-004 | P2 | README ordinary-sum output is wrong | With documented x64 setup, `jnp.sum([1e16, 1, -1e16, 1])` is `1.0`, not `0.0`; compensated sum is `2.0` | **Resolved in C1:** the README no longer promises a portable ordinary-reduction value; the exact block requires the compensated result to be 2.0 and observes that the local ordinary result differs |
| C0-005 | P2 | Package-wide JAX/AD claims are overbroad | The audited contracts exclude spatial preprocessing, hard branches, clamp boundaries, poles, origins, and coincident geometries | **Resolved across the primary overview surfaces in C1, C4-C6, and C8:** dedicated numerical pages retain their individual claim-review order |
| C0-006 | P2 | Quantity is simultaneously documented as implemented and planned/missing | The landing, architecture index, and science-general vision formerly disagreed with the implemented package and current ecosystem policy | **Resolved across C3–C5:** all three pages identify `jaxstro.units` as the current contract, `jaxstro.quantity` as implemented, and ecosystem adoption/replacement as deferred |
| C0-007 | P2 | Spatial has no dedicated conceptual chapter | Spatial formerly appeared only in README/API/validation without one place explaining Morton ordering, capacity/overflow, approximate-candidate recall, exact fixed-radius pairs, or discrete preprocessing | **Resolved in C7:** the executable spatial theory chapter, decision table, and JaxtroViz figure distinguish each contract and link back to API/validation |
| C0-008 | P2 | Slice-B provenance cards are not integrated into the teaching/reference narrative | The landing, architecture index, API, validation page, and dedicated provenance architecture now route to generated cards and distinguish their evidence from runtime manifests | **Resolved across C3, C4, C6, C9, and C10:** the final ownership page verifies live APIs, card states, registry routes, and the honest atmosphere gap |
| C0-009 | P2 | API reference contains duplicated interpolation prose | The C6 source check finds exactly one `pchip_slopes(...)` description and one `monotone_cubic_interp(...)` description | **Closed as not reproducible in the C6 starting state;** retain the singleton regression assertion |
| C0-010 | P2 | README mislabels the rounded solar-mass conversion | README calls `MSUN_G` simply “Solar mass [g]”; Slice A/B establish it as a rounded conversion from nominal $(GM)_\odot$ and the selected CODATA $G$, not an IAU nominal solar mass | **Resolved in C1:** README carries the derived-conversion wording without implying that the current units API is retired |
| C0-011 | P2 | Site landing module list and ecosystem status lag the package | The former landing omitted quantity, atmospheres, geometry, and provenance from its API list, called quantity planned, and called Startrax planned | **Resolved in C3:** the import-backed module inventory, active Startrax status, deferred quantity-adoption boundary, and atmosphere in-progress boundary are current |
| C0-012 | P3 | Several completed sections still use stub/future tense | Validation formerly said it “will carry” the table already present | **Resolved in C4 and C9:** architecture and validation now describe their current, evidence-specific surfaces |
| C0-013 | P3 | Python fences do not expose their execution contract | Continuations, placeholders, local-data examples, and interface pseudocode all use the same unlabeled `python` fence as runnable snippets | **Resolved across C11a-C11f:** executable numerical examples run verbatim; the spectra page now distinguishes portable code, interface notation, and artifact-dependent recipes explicitly |
| C0-014 | P3 | The rendered bibliography page contains a duplicate `references` ID | MyST rendered both the authored “References” heading and the generated bibliography section as `id="references"`; the C8 DOM audit found two occurrences on `/index-11` | **Resolved in the reusable C2 gate:** the authored heading is removed, all 59 rendered routes pass duplicate-ID inspection, and the checker retains the regression contract |

## Public-API claim map

| Documented surface | Installed evidence | Currency verdict |
| --- | --- | --- |
| `constants`, `units`, `astrometry`, `coords`, `geometry`, `numerics`, `params`, `quantity`, `provenance`, `testing`, `atmospheres` | Imported from the top-level package; module exports and focused tests exist | Present; page-specific wording still needs the corrections above |
| `spatial` | `jaxstro.spatial` imports directly, eagerly at top level, and has unit/validation coverage | Implemented; C6 ratified eager public ownership and added it to `jaxstro.__all__` |
| Quantity theory/architecture/how-to | `src/jaxstro/quantity/` and quantity test families | Implemented; ecosystem adoption and replacement of the current units contract are deferred |
| Spatial neighbor gathering and exact pairs | `src/jaxstro/spatial/` and `tests/unit/test_spatial.py` | Implemented; dedicated conceptual documentation and an executable public-API figure were added in C7 |
| Provenance-card tooling | `jaxstro.testing.provenance_cards`, registry validation tests, generated pages | Implemented; navigation, API, validation, and the runtime-manifest versus source-card ownership narrative are current through C10 |
| Atmosphere runtime | NewEra/BOSZ backends plus catalog/prepared-grid tests; Sonora/TLUSTY runtime policies absent | Correctly in progress; keep the boundary honest and non-blocking |
| Per-symbol generated API reference | No full signature/parameter generator exists | The “planned” note remains true; provenance cards are evidence pages, not a substitute |

## Proposed page-by-page order

Each row after C0 is a separate approval gate and, once approved, a separate
verified commit. The order prioritizes public breakage before broader pedagogy.

| Order | Page | Learner outcome and bounded work | Proof before prose |
| --- | --- | --- | --- |
| C1 | `README.md` | A new user can run the public quick start; parallax, summation, solar-mass, and transform claims are exact | Extend `test_readme_examples.py` to execute the final blocks verbatim |
| C2 | `docs/00-getting-started/index.md` | First lesson demonstrates a scientifically valid differentiable solve rather than a bisection artifact | Analytic plus independent FD-vs-AD test for the chosen example |
| C3 | `docs/index.md` | **Completed:** landing routes users to implemented quantity, spatial, provenance, and atmosphere boundaries while preserving current-unit and in-progress boundaries | 3 import/currency tests pass; strict 58-page MyST build passes; rendered `/` DOM has the correct anchor and five required resolved routes |
| C4 | `docs/20-architecture/index.md` | **Completed:** architecture describes current ownership, transform/discrete boundaries, provenance surfaces, and the implemented-but-not-adopted quantity layer | 7 architecture/figure tests pass; registered WebP is deterministic and fresh; strict 58-page build and rendered `/index-3` figure/alt/route/xref checks pass |
| C5 | `docs/20-architecture/science-general-vision.md` | **Completed:** vision maps nine delivered modules and applies evidence-based admission criteria to genuinely future work | 4 focused tests import every former candidate; strict 58-page build and rendered 9-row table/11-route/xref checks pass without duplicating the C4 figure |
| C6 | `docs/40-api/index.md` | **Completed:** every advertised import has explicit ownership, execution boundary, status, and evidence route; `spatial` has ratified eager top-level ownership | 5 focused code/docs tests pass; clean-process import and `__all__` agree; strict 58-page build and rendered 13-row/4-column table/route/xref checks pass |
| C7 | new `docs/10-theory/spatial.md` | **Completed:** users can distinguish Morton grouping, fixed capacity, approximate candidates, and exact fixed-radius pairs, including every overflow and preprocessing boundary | 20 focused code/docs/figure tests pass; power-of-two Morton allocation is enforced; JaxtroViz figure uses public API results; strict 59-page build and rendered figure/table/citation/route/xref checks pass |
| C8 | `docs/10-theory/index.md` | **Completed:** the thesis teaches the five live gradient contracts, limits inference-ready claims to verified smooth paths, and distinguishes Newton conditions from branch-selected bisection | 4 focused registry/prose tests pass; strict 59-page build passes; rendered 5-row/4-column table, required routes, unique scoped anchor, and internal-link behavior are verified; the site-wide scan also records pre-existing C0-014 |
| C9 | `docs/60-validation/index.md` | **Completed:** validation states current coordinate, spatial, quantity, card-registry, and atmosphere boundaries with resolvable evidence anchors | 5 focused page tests and 33 bounded cross-cutting tests pass; strict 59-page build passes; rendered 40-row/4-column table, unique anchor, four routes, registry node IDs, and internal-link behavior are verified |
| C10 | `docs/20-architecture/provenance.md` | **Completed:** runtime manifests and source-backed card registries have a clear ownership, composition, status, and non-substitution contract | 4 focused architecture tests and 38 bounded runtime/card/registry tests pass; strict 59-page build passes; rendered 3-row/5-column table, unique anchor, five routes, live states, gap text, and internal-link behavior are verified |
| C11a | `docs/10-theory/bsplines.md` | **Completed:** the public example executes, gradient boundaries are explicit, recurrence/derivative claims cite primary evidence, and local support is visualized from public API results | 5 focused page tests, 13 page/figure tests, and 24 spline unit/gradient tests pass; WebP freshness and strict 59-page build pass; rendered figure/table/citation/code/routes are verified |
| C11b | `docs/10-theory/interpolation.md` | **Completed:** one executable fixture distinguishes smooth natural interpolation from PCHIP shape preservation, with explicit gradient/JIT boundaries, primary sources, and a public-API figure | 5 page tests, 24 page/unit/gradient tests, and 10 figure-registry tests pass; WebP freshness and strict 59-page build pass; rendered figure/table/citations/code/routes are verified |
| C11c | `docs/10-theory/random.md` | **Completed:** explicit key ownership, deterministic replay, resampling input/shape contracts, and discrete differentiation boundaries replace the placeholder; concrete invalid inputs fail closed before private jitted kernels | 36 focused runtime/page tests and 46 API/validation cross-page tests pass; Ruff and mypy pass; strict 59-page build and static HTML render pass; the DOM has one contract-table ID, both primary citations, executable code, and three resolved evidence routes |
| C11d | `docs/10-theory/regular-grid.md` | **Completed:** an executable affine fixture teaches generic/bilinear parity, arbitrary payload axes, boundary policies, and five differentiation/execution contracts; whole-payload fill and scalar-axis validation defects fail closed; a public-API figure visualizes measured corner weights and clamp/fill behavior | 13 runtime/page/gradient tests, 21 API/validation cross-page tests, and all 11 deterministic figure-registry tests pass; Ruff and mypy pass; strict 59-page content/static builds and rendered figure/table/citation/code/route checks pass |
| C11e | `docs/10-theory/linear-algebra.md` | **Completed:** an executable fit/solve/covariance/jitter fixture teaches finite weight and denominator boundaries plus six differentiation contracts; correlation conversion fails closed on invalid concrete covariance inputs; a public-API figure visualizes declared weighting and the selected diagonal shift | 57 runtime/page/gradient tests, 60 API/validation cross-page tests, and all 12 deterministic figure-registry tests pass; Ruff and mypy pass; strict 59-page content/static builds and rendered figure/table/citation/code/route checks pass |
| C11f | `docs/20-architecture/spectra-data-architecture.md` | **Completed:** interface notation, a portable prepared-grid proof, and two local-artifact recipes have distinct execution contracts; the host/JAX/downstream ownership boundary and incomplete backends remain explicit | 28 atmosphere/docs tests and all 13 figure-registry tests pass; Ruff passes; strict 59-page content/static builds pass; rendered DOM has unique figure/table IDs, exact alt text, the hashed WebP, and three resolved evidence routes |
| Reusable C2 gate | `scripts/check_docs.sh`, `scripts/check_docs_site.py`, `docs/route-manifest.json` | **Completed:** strict content rendering, stable root-flat routes, duplicate IDs, internal links/new-tab behavior, and image alt text are one local/CI contract | 12 checker/wiring tests pass; all 59 rendered routes pass in 7.39 s; the bibliography collision is fixed; the gate is parallel in scheduled/manual CI and reused by `scripts/check.sh` |
| Deferred program | `docs/10-theory/quantities.md`, `docs/20-architecture/quantity-system.md`, `docs/50-howto/quantity-migration.md` | Preserve an honest implemented-but-not-adopted status; do not teach an ecosystem cutover yet | Reopen only after explicit cross-package design approval |
| Later, not blocking | atmosphere capability/how-to pages | Preserve accurate in-progress status; revise only when backend/data claims change | Existing data-extra and local-artifact gates where applicable |

The remaining theory pages (`autodiff`, `cumulative-trapz`, `distributions`,
`geometry`, `grids`, `meshes`, `ode`, `operators`, `optimization`, `quadrature`,
`rootfinding`, and `special-functions`) contain no Python fences requiring C0
classification. They still receive a claim/link pass under the reusable docs
gate, but no currency defect found here moves them ahead of the approved
non-quantity page sequence.

## C0 executable evidence

- `tests/integration/test_readme_examples.py` runs two clean subprocesses using
  public imports and x64-before-array ordering.
- Verified outputs: solar escape speed about 617.7 km/s, dynamical
  `G` about 0.00450, sky-tangent output shape `(2, 2)`, finite Galactic transform,
  center-star parallax 10 mas at 100 pc, bisection/Newton roots equal to
  `sqrt(2)`, and compensated sum equal to 2.0.
- Direct reproductions: missing parallax positions, the spatial auto-load versus
  `__all__` mismatch, ordinary-sum mismatch, and target-dependent-bracket
  bisection gradient provenance.

### Rendered-site evidence

- `myst build --html --ci --strict` completed 58 pages with exit status zero in
  3.83 seconds wall time. The Node experimental/deprecation notices were renderer
  runtime notices, not MyST content diagnostics.
- `docs/_build/site/myst.xref.json` contains 58 page references, 58 unique page
  URLs, and no duplicate page URL.
- The final HTML contains 58 rendered routes and every internal route-valued
  `href` resolves to one of them. No internal link opens a new tab.
- External content links were inspected in the HTML and render with
  `target="_blank" rel="noreferrer"`; source/edit links use the theme's explicit
  external-link attributes. No source `.md` path appears as an internal href.
- The rendered landing page still contains “Planning quantity-aware APIs?”, the
  Getting Started DOM contains the bisection lesson, the API DOM contains the
  per-symbol-planning note, and `/index-6` renders the generated “Provenance
  cards” page. These are DOM observations, not source-AST assumptions.
- Eleven generic `index.md` pages currently receive root-flat deduplicated URLs
  `/index-1` through `/index-11`. They are unique today, but a successful build
  does not prove that a future added page will retain the same semantic route;
  C2 must ratchet this mapping or replace it with stable explicit slugs.

The reusable zero-warning/link/xref gate belongs to C2 of the implementation
plan. C0 records the current rendered build state without silently turning an
inventory slice into that later infrastructure change.
