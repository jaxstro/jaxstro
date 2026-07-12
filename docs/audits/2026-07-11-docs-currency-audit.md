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
| `docs/20-architecture/spectra-data-architecture.md:28` | Pseudocode | `AtmosphereParams -> SpectrumResult` boundary | Not Python syntax; should be labelled as an interface diagram |
| `docs/20-architecture/spectra-data-architecture.md:81` | Local-data example | `AtmosphereLibrary.from_local/select/spectrum` | Signatures map; execution requires processed local artifacts |
| `docs/20-architecture/spectra-data-architecture.md:95` | Local-data example | `NewEraBackend.open/prepare`, JIT spectrum evaluation | Signatures map; execution requires processed local artifacts |
| `docs/10-theory/bsplines.md:29` | Illustrative | spline construction, evaluation, fitting | coefficients and samples are placeholders |
| `docs/10-theory/interpolation.md:15` | Illustrative | cubic/PCHIP interpolation surface | grids, values, derivatives, and queries are placeholders |
| `docs/10-theory/random.md:18` | Illustrative | `key_stream` | import and `key` are intentionally omitted |
| `docs/10-theory/regular-grid.md:15` | Illustrative | regular/bilinear/trilinear interpolation | axes, values, and queries are placeholders |
| `docs/10-theory/linear-algebra.md:74` | Pseudocode | `positive_definite_jitter` return shape | names show the return contract, not a runnable call |
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
| C0-005 | P2 | Package-wide JAX/AD claims are overbroad | The audited contracts exclude spatial preprocessing, hard branches, clamp boundaries, poles, origins, and coincident geometries | **Resolved for README in C1, architecture in C4, vision in C5, and API in C6:** remaining site pages retain their individual review order |
| C0-006 | P2 | Quantity is simultaneously documented as implemented and planned/missing | The landing, architecture index, and science-general vision formerly disagreed with the implemented package and current ecosystem policy | **Resolved across C3–C5:** all three pages identify `jaxstro.units` as the current contract, `jaxstro.quantity` as implemented, and ecosystem adoption/replacement as deferred |
| C0-007 | P2 | Spatial has no dedicated conceptual chapter | Spatial formerly appeared only in README/API/validation without one place explaining Morton ordering, capacity/overflow, approximate-candidate recall, exact fixed-radius pairs, or discrete preprocessing | **Resolved in C7:** the executable spatial theory chapter, decision table, and JaxtroViz figure distinguish each contract and link back to API/validation |
| C0-008 | P2 | Slice-B provenance cards are not integrated into the teaching/reference narrative | The landing, architecture index, and API now route to generated cards and distinguish them from runtime manifests, but validation anchors and the dedicated provenance page do not yet describe card validation/rendering or evidence states | **Resolved for C3, C4, and C6;** reconcile the remaining provenance-architecture and validation pages after their individual approvals |
| C0-009 | P2 | API reference contains duplicated interpolation prose | The C6 source check finds exactly one `pchip_slopes(...)` description and one `monotone_cubic_interp(...)` description | **Closed as not reproducible in the C6 starting state;** retain the singleton regression assertion |
| C0-010 | P2 | README mislabels the rounded solar-mass conversion | README calls `MSUN_G` simply “Solar mass [g]”; Slice A/B establish it as a rounded conversion from nominal $(GM)_\odot$ and the selected CODATA $G$, not an IAU nominal solar mass | **Resolved in C1:** README carries the derived-conversion wording without implying that the current units API is retired |
| C0-011 | P2 | Site landing module list and ecosystem status lag the package | The former landing omitted quantity, atmospheres, geometry, and provenance from its API list, called quantity planned, and called Startrax planned | **Resolved in C3:** the import-backed module inventory, active Startrax status, deferred quantity-adoption boundary, and atmosphere in-progress boundary are current |
| C0-012 | P3 | Several completed sections still use stub/future tense | Validation says it “will carry” the table already present | **Resolved for the architecture index in C4;** convert validation to current-tense, evidence-specific prose in its approved page slice |
| C0-013 | P3 | Python fences do not expose their execution contract | Continuations, placeholders, local-data examples, and interface pseudocode all use the same unlabeled `python` fence as runnable snippets | In each page slice, make examples standalone or label the non-executable contract explicitly |

## Public-API claim map

| Documented surface | Installed evidence | Currency verdict |
| --- | --- | --- |
| `constants`, `units`, `astrometry`, `coords`, `geometry`, `numerics`, `params`, `quantity`, `provenance`, `testing`, `atmospheres` | Imported from the top-level package; module exports and focused tests exist | Present; page-specific wording still needs the corrections above |
| `spatial` | `jaxstro.spatial` imports directly, eagerly at top level, and has unit/validation coverage | Implemented; C6 ratified eager public ownership and added it to `jaxstro.__all__` |
| Quantity theory/architecture/how-to | `src/jaxstro/quantity/` and quantity test families | Implemented; ecosystem adoption and replacement of the current units contract are deferred |
| Spatial neighbor gathering and exact pairs | `src/jaxstro/spatial/` and `tests/unit/test_spatial.py` | Implemented; conceptual documentation is missing |
| Provenance-card tooling | `jaxstro.testing.provenance_cards`, registry validation tests, generated pages | Implemented; navigation exists, narrative/API/validation integration is incomplete |
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
| C8 | `docs/10-theory/index.md` | Thesis states differentiability contracts without claiming every primitive/branch is smooth | Link claims to Slice-A gradient contracts |
| C9 | `docs/60-validation/index.md` | Validation includes current coordinate, spatial, quantity, and card-registry anchors in present tense | Resolve every cited node ID and render the table |
| C10 | `docs/20-architecture/provenance.md` | Runtime manifests and model-card registries have a clear ownership split | Registry freshness and reference-resolution tests |
| C11+ | `bsplines.md`, `interpolation.md`, `random.md`, `regular-grid.md`, `linear-algebra.md`, `spectra-data-architecture.md` | Each illustrative block is either completed or explicitly marked | One page and one focused proof at a time |
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
