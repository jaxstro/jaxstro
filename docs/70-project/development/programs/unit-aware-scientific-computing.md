---
title: Unit-aware scientific computing program
description: >-
  Planned Jaxstro quantity adoption checks, starting with Progenax kinematics
  consumed by Informax observing-design examples.
---

# Unit-aware scientific computing program

:::{important} Planned program, 2026-09-25
The first path is selected: check quantities at physical boundaries while
keeping the existing numerical kernels. The preferred long-term direction is
a Jaxstro-owned unit-aware mathematical system. Phase 2 compares architectures
before its detailed design is fixed. The selected public interface is hybrid:
`q.math` defines the tested unit-aware operations, and direct JAX calls are
supported where they preserve the same contract. One public `Quantity`
container is selected for multiplicative, affine, and logarithmic physical
representations, with semantic rules for conversion and operators. Photometry,
dynamics, and learned-surrogate workflows are all selected as independent
program lanes. The intended adoption horizon is the post-v1 development cycle
of every ecosystem package. The phase details below are a design sketch.
Phase 2 includes small feasibility probes for the hardest proposed semantics
before an architecture is selected. Later phase gates require physical results
to agree when the same case is expressed in different compatible units.
This page does not authorize a `jaxstro.units` replacement, change a sibling
package, or qualify an inference result. The existing examples retain their
physics and numerical assumptions.
:::

## Objective and current boundary

Demonstrate that explicit physical units can prevent interface errors in a
differentiable scientific workflow without obscuring numerical scales, derivative
meaning, or JAX compilation cost. Let observed consumer gaps determine any
addition to `jaxstro.quantity`.

Scientific AI here includes model-based inference and observing design driven by
JAX derivatives, differentiable simulation, probabilistic inference, and
learned components when a scientific question needs them. The intended program
is larger than the first quantity-boundary pilot. It should eventually decide
whether Jaxstro can support unit-aware mathematical work across substantial
JAX workflows, with tested physical semantics, transformations, derivatives,
performance, and interoperation. The first cases exercise observing design;
they do not establish that wider capability.

[`jaxstro.units`](../../../30-representations/units-quantities/quantity-system.md)
remains the ecosystem's canonical unit-system contract. `jaxstro.quantity`
already supplies an immutable `Unit`, a JAX PyTree `Quantity`, explicit
conversion, a small math surface, and a scalar-output `q.grad`. The
[parameter bridge](../../../30-representations/parameters-state/parameters-and-transforms.md)
handles selected numerical parameters; it does not choose their physical
interpretation. The initial sibling source scan found no direct
`jaxstro.quantity` import in the inspected Progenax and Informax `src` trees.
Those packages currently use explicit numerical conventions.

The proposed first pass uses quantities at physical entry and exit points. A
conversion to numerical arrays is recorded before an existing JAX kernel. This
preserves the current algorithms while exposing the units of their inputs,
observables, uncertainties, and derivatives. How far quantities should travel
through later computations remains a program decision. The v1 releases of
Jaxstro and its siblings have their own gates; this program does not add a
quantity migration requirement to those gates. The v2 objective is a common
unit-aware physical data contract across the ecosystem: public scientific APIs
require `Quantity` for physical inputs and outputs, including dimensionless
physical values. Numerical kernel boundaries and transform behavior remain
explicit per package.

## What this enables for Scientific AI

The program serves two related workflows. Scientific ML uses simulations,
observations, differentiation, optimization, and learned models to answer a
scientific question. AI coding assistants help researchers build and audit that
software. `Quantity` supplies a common physical contract to both workflows;
it does not choose a model or validate a scientific conclusion.

In a proposed end-to-end path, Progenax supplies physical parameters and
predicts a state, Gravax may evolve that state, and Fluxax maps it to an
observable. A learned surrogate may replace a qualified part of the forward
calculation. Informax constructs a likelihood, sensitivity, or observing
design from the predicted and measured quantities. Each package can state the
units and semantic kind of its physical inputs and outputs. This lets the
workflow reject a mismatched velocity, a missing magnitude reference, or an
uncertainty expressed in the wrong observable before those values enter a
likelihood. The three selected Phase 6 lanes test distinct portions of this
path; this paragraph is a proposed composition, not an implemented pipeline.

A network or optimizer can still use dimensionless numerical coordinates:
convert physical inputs using a declared scale, train or optimize in those
coordinates, then reconstruct physical `Quantity` outputs. The conversion
must be recorded so a Jacobian with respect to network coordinates is not
mistaken for a derivative with respect to mass, radius, or time. The same rule
applies to custom AD and implicit derivatives. Unit correctness can expose
interface and chain-rule mistakes; prediction error, calibration, model
validity, and uncertainty coverage need independent evidence.

## Codex and Claude research workflows

Codex and Claude can use the same source, public unit contracts, tests, and
evidence records to make cross-package work reviewable. For a proposed change,
an agent should identify the scientific owner, write an input/output unit and
semantic-kind ledger, state the expected physical or dimensional behavior,
implement the bounded change, and report the exact parity, rejection, JAX
transform, custom-AD, and performance checks it actually ran. It should record
the conversion into any raw numerical kernel and the reconstruction of its
output. A later agent can inspect those artifacts instead of inferring a unit
convention from array names or a passing test.

Anna chooses physical models, approximations, calibration references,
likelihoods, validity regimes, and scientific acceptance criteria. Agents can
propose alternatives and run an approved test. A green type or unit check is
evidence about the software contract; it does not establish that the model or
inference result describes nature. This follows the
[scientific ML ownership boundary](../../../40-workflows/scientific-ml/ecosystem-boundaries.md)
and the repository's researcher-led validation rules.

## Package responsibilities

| Owner | Responsibility in the first cases |
| --- | --- |
| Jaxstro | Unit and conversion rules; boundary checks; tests of any general-purpose quantity operation; compilation and derivative evidence for that operation. |
| [Progenax](https://github.com/jaxstro/progenax) | Plummer/Osipkov-Merritt projected-dispersion physics, parameter interpretation, and physical-model validity. |
| [Informax](https://github.com/jaxstro/informax) | Jacobian-to-Fisher construction, observing-design criteria, optimization, calibration, and inference claims. |

Jaxstro does not own the mock cluster, its noise model, survey selection, or
an acceptable observing strategy. A unit-consistent result is not evidence
that those choices describe an observed cluster.

## First use case: projected kinematics to observing design

Use the existing [Progenax to Informax Stage 1 example](https://github.com/jaxstro/informax/blob/main/examples/progenax_kinematic_oed/README.md)
as the baseline. Its [forward module](https://github.com/jaxstro/informax/blob/main/examples/progenax_kinematic_oed/forward.py)
passes $(r_a,M,r_h)$ in pc, solar masses, and pc to Progenax. It predicts
line-of-sight and proper-motion velocity dispersions in pc/Myr. Informax
uses a Jacobian and a dispersion variance to form per-cell Fisher blocks,
then compares observing allocations. The current example also converts
km/s and proper-motion errors to its native velocity unit with explicit
numerical factors.

The first unit check should cover the three parameter roles separately, the
radial bins, predicted dispersions, and error scales. A single `Quantity`
array cannot represent a parameter vector whose entries have different units.
The check should leave the existing numerical parameter vector and
`jax.jacrev` path intact.

For positive parameters, the existing Jacobian is multiplied by the parameter
vector to express sensitivity to fractional parameter changes. Before that
multiplication, each Jacobian column has the observable unit divided by the
corresponding parameter unit. After multiplication, each column has velocity
units. Dividing its outer product by a velocity variance yields a dimensionless
Fisher block. This dimensional chain is the decisive first assertion. Compare
the resulting numerical predictions, Jacobian, and Fisher blocks with the
unchanged example, including its current manual conversion factors.

The existing example's mock values, nuisance prior, completeness curve, and
dispersion-noise relation are baseline assumptions for this engineering
comparison. Their scientific adequacy is a separate Progenax and Informax
decision.

## Second use case: survey-depth design

After the first boundary is stable, use the existing [Stage 2 depth example](https://github.com/jaxstro/informax/blob/main/examples/progenax_kinematic_oed/depth.py).
It changes a limiting apparent magnitude, detectable-star supply, and
per-star velocity error while reusing the projected-dispersion Jacobian.
Check units on distance, velocity errors, dispersion variance, and the
Fisher input. Preserve the existing source and design decisions.

Apparent magnitude is logarithmic and outside the current `Quantity` contract.
The domain owner keeps its selection and photometric conventions. Phase 3 will
define how a magnitude can inhabit the approved single `Quantity` container
without treating it as an ordinary dimensionless value or choosing a
photometric zero point for the domain.

## Selected first step and long-term architecture choices

| Approach | Quantity path | What it can establish | Main cost |
| --- | --- | --- | --- |
| Boundary checks first (recommended starting design) | Validate physical inputs and outputs, then use the existing numerical forward model and Informax calculation. | Whether explicit units catch interface errors with little migration. | Unit rules inside the numerical kernel remain documented and independently checked. |
| Selected in-kernel operations | Carry `Quantity` through a specific calculation that needs unit-aware math or a physical Jacobian. | Whether one additional operation materially improves the scientific contract. | More wrappers, transform tests, and unit-specialized compilations. |
| Full quantity path | Carry units through the forward model, likelihood, Fisher construction, and design objective. | Whether broad unit propagation is useful for this workflow. | A large migration and a wide mathematical API before the consumer benefit is known. |

The boundary-first approach is selected for Phase 1 and follows the current
[](../../../30-representations/units-quantities/quantity-system.md) contract.
It is an entry point, not the program's end state. The preferred end state is
a Jaxstro-owned system. The architecture comparison should inform its design
and test whether it can meet the intended contracts; an external unit engine
is an evaluated alternative, not the default implementation path.
[ADR 0006](../../decisions/0006-build-own-quantity-not-unxt.md) already chose
an opt-in Jaxstro `Quantity` with explicit numerical boundaries for the
original scope. Expanding that contract or adopting another engine requires an
explicit decision. No architecture choice changes the source model, noise law,
prior, or design objective by itself.

The architecture comparison will examine [SaiUnit](https://github.com/chaobrain/saiunit)
for broad unit-aware math and AD and [Unxt](https://github.com/GalacticDynamics/unxt)
for JAX operator interoperation. JAX's [PyTree static metadata rules](https://docs.jax.dev/en/latest/pytrees.html)
constrain every option. These are comparison candidates, not approved runtime
dependencies or replacements for the current `UnitSystem` contract.

## Proposed phases

### Phase 0 — charter, baseline, and success claims

Record the exact Progenax, Informax, and Jaxstro revisions and the selected
Stage 1 configuration. Make a compact ledger for $(r_a,M,r_h)$, radial bins,
velocity channels, measurement errors, variance, Jacobian columns, and Fisher
blocks. Record the existing numerical outputs, tests, transform path, and
known science limitations. State separately what the program aims to prove
about unit safety, mathematical coverage, derivative meaning, runtime cost,
and scientific inference. Set parity, unit-representation invariance, and
derivative comparison criteria before a qualification run; this design
supplies no new tolerance or physical assumption.

**Exit:** a reviewable baseline and unit ledger. A missing unit convention or
unresolved physical assumption is assigned to its owner before Phase 1.

### Phase 1 — test the physical boundary

Use the existing `jaxstro.quantity` conversion and compatibility rules to
check heterogeneous parameter roles and velocity inputs in a small consumer
experiment. Convert once into the numerical units expected by
`project_dispersion`. Compare with the existing `STELLAR` and manual conversion
path, including km/s to pc/Myr. Reject dimensionally or semantically invalid
inputs. Keep the old example as the comparison reference during this phase.

**Exit:** executable parity and rejection checks with an explicit account of
every conversion. If a factor differs, identify its convention and source
before changing the calculation. No new public Quantity API is presumed.

### Phase 2 — compare unit-system architectures

Keep the frozen Stage 1 workload as the consumer baseline. Compare the current
Jaxstro path with isolated SaiUnit and Unxt paths where each can express the
required operations. Stage 1 retains a raw parameter vector and raw
`jax.jacrev`, so success there does not establish the proposed in-kernel
quantity, affine, logarithmic, or mixed-unit AD contracts. Before selecting an
architecture, run bounded feasibility probes for those contracts:

| Probe | Discriminating result |
| --- | --- |
| Affine point and difference | Convert a temperature point between Celsius and Kelvin; test that its JVP tangent is a temperature difference, check VJP pairing, and state supported second derivatives. |
| Logarithmic representation | Differentiate a specified flux-to-magnitude conversion with respect to both flux and its reference; test valid, zero, negative, and batched mixed-validity inputs without hidden clipping. The domain owner supplies the reference and convention. |
| Mixed-unit linear algebra | Use two parameter roles and two observables with different units; transform the Jacobian and covariance when each basis changes independently, then check a dimensionless quadratic form. |
| Compiled physical loop | Carry a fixed-unit state through a short `scan` containing one supported custom AD rule; test `jit`, `vmap`, and the declared forward and reverse directions. |
| Direct JAX interoperation | For selected operations, compare `q.math` with direct `jax.numpy` and `jax.lax` calls; record correct results, explicit rejection, and unsupported dispatch separately. |

These probes test feasibility, not scientific validity or full API coverage.
Record the provisional semantic assumptions each uses for Phase 3 review. For
every candidate and probe, record supported, rejected, or unsupported behavior,
including incorrect-unit rejection, numerical conversion, PyTree behavior,
transform composition, and error diagnostics. Compare semantic correctness,
coverage, debugging, serialization, dependency and licensing costs, upgrade
maintenance, and migration impact at pinned library and JAX revisions. Include
a bounded examination of [JAX's experimental custom types](https://docs.jax.dev/en/latest/hijax_types.html)
for a tangent kind distinct from its primal; this does not select that API or
make it a dependency. Measure trace, lowering, backend compilation, warm
execution, memory, and recompilation on matched shapes, dtypes, units, and
hardware. Do not infer a performance winner from one warm call or an
unsupported operation.
Operation count and the existence of unit-aware gradient wrappers are comparison
baselines, not sufficient evidence for a state-of-the-art claim. The program
tests whether Jaxstro's admitted semantics are correct across the stated
transforms and workflows, with useful diagnostics and an acceptable measured
cost. Record what each pinned reference already provides and what remains
unsupported; do not label an untested surface as superior.

**Exit:** a Jaxstro-native architecture proposal with recorded outcomes for
every feasibility probe, known coverage gaps, a supported JAX version range,
an interoperation strategy, and ownership for wrapper or primitive updates.
If the native design cannot satisfy a required contract at an acceptable
maintenance or runtime cost, present that evidence and a proposed alternative for a separate
decision. The comparison does not silently change ADR 0006 or migrate sibling
packages.

### Phase 3 — specify physical quantity semantics

Specify one public `Quantity` container with a static semantic kind. Its
contract must distinguish multiplicative quantities, points on affine scales,
their differences, and logarithmic representations. Define the permitted
conversions and operator results for every admitted pair of kinds: for example,
subtracting two temperature points yields a temperature difference, while
adding a difference to a point yields a point. A Celsius coordinate must not
be treated as a scale-only unit. Logarithmic conversion must retain its
reference and convention; adding two magnitudes must not silently stand for
adding their underlying fluxes.

Write the rules for dimensions, scales, angle tags, conversion contexts,
equivalencies, structured parameters with different units, and serialization.
The angle algebra must state how angular rates, inverse angles, powers, solid
angles, and angular versus cyclic frequency compose. In particular, test
whether an angle recovered from an angular rate times elapsed time is accepted
by trigonometric operations, and check the scale of its derivative. The
current rule drops the angle tag in products with dimensional units; any
change to that rule requires an explicit semantic decision.

The selected conversion policy is hybrid:

| Conversion | Required information | Policy |
| --- | --- | --- |
| Compatible multiplicative units | Dimension and scale | Direct conversion. |
| Affine or logarithmic representation | Kind, offset or reference, and convention fully specified by the value and target | Direct conversion using the specified definition. |
| Physical equivalency or external calibration | A relation, reference, or calibration that must be chosen | Explicit caller-supplied context; no implicit registry selection. |

Keep unit and semantic kind static across JAX transforms; one array has one
static unit and kind. The semantic contract must distinguish static definitions
from calibration values that may be dynamic JAX data. Domain owners supply
scientific conventions such as photometric zero points. Explicitly define
failures for incompatible units, missing conversion context, undefined
nonlinear conversions, and branches returning different unit or kind
structures. Specify each nonlinear conversion's direction and numerical
domain, eager and traced failure behavior, and treatment of invalid members
of a batch without hidden clipping. State how multiple supplied equivalencies
are resolved or rejected when more than one applies. Identify which reference
and calibration fields are static definitions and which are dynamic
differentiable leaves. Provide a comprehensive conversion and operator matrix
rather than assuming that every pair of representations has a meaningful
conversion or arithmetic operation.

**Exit:** a reviewed single-container semantic contract and adversarial tests
for every admitted representation and operator, including angular-rate
round trips, nonlinear domains, and explicit-context ambiguity. A dimensionally
consistent expression is still subject to physical-model and numerical-domain
checks.

### Phase 4 — build the unit-aware mathematical surface

Inventory the operations used by real Jaxstro ecosystem workloads. Cover
coherent families rather than isolated aliases: array creation and shape,
arithmetic and comparison, reductions and selection, transcendental and
special functions, linear algebra and contractions, transformations,
quadrature and differential-equation boundaries, and control flow. Consider
FFT, sparse, random, and multi-device operations when a workload uses them.
Specify how a calculation represents a Jacobian or covariance whose entries
have different units. Use heterogeneous parameter and observable PyTrees with
explicit coordinate bases first; add a general operator representation only
if the consumer probes show that these cannot express the required contraction,
transpose, solve, or reconstruction. For parameter units $u_i$, covariance
entry $C_{ij}$ has units $u_i u_j$ and a physical-coordinate Fisher entry
$F_{ij}$ has units $1/(u_i u_j)$.

For the selected hybrid interface, `q.math` is the supported unit-aware
mathematical contract. Add direct `jax.numpy` or `jax.lax` use on `Quantity`
only where it passes the same unit, scale, transform, and failure checks.
An unsupported direct JAX call must not silently discard units. Each operation
needs an input-unit rule, output-unit rule, scale behavior, JAX transform
contract, and failure case. Record unsupported operations in a coverage matrix
so a broad API claim is testable.

**Exit:** the selected workload runs with the intended quantity path, and
the public coverage matrix matches executable tests. This phase does not
promise arbitrary `jax.numpy` compatibility without an architecture that
demonstrates it.

### Phase 5 — unit-aware calculus and inference contracts

Check the units of each column of $J_\theta=\partial\sigma/\partial\theta$.
Check that $J_\theta\,\mathrm{diag}(\theta)$ has velocity units and that its
variance-normalized Fisher blocks are dimensionless. Compare values and
derivatives with the unchanged Stage 1 path and an independent analytic or
central-difference reference on its stated smooth domain. Keep the existing
Informax criterion, allocation, and calibration gates as separate evidence.

Extend the design beyond the current scalar `q.grad`: specify the unit rules
for `value_and_grad`, JVP/VJP, Jacobian, Hessian, and mixed-unit parameter
PyTrees. The selected hybrid AD interface makes quantity-aware functions the
source of physical derivative units; direct JAX AD receives that claim only
on paths with verified unit behavior. Raw `jax.grad` on the current `Quantity`
PyTree does not by itself provide the physical output/input unit.

Define the primal, tangent, and cotangent objects for each semantic kind and
the coordinate map used by each derivative. An affine point's tangent is a
difference, while a logarithmic coordinate tangent and a physical-flux tangent
are related by the derivative of the conversion. A VJP cotangent belongs to
the dual of the stated output coordinate. A raw-value derivative may be
reconstructed into a physical result only after these maps are checked.
Direct JAX AD on a `Quantity` receives no physical-unit claim merely because
its PyTree transformation succeeds.

Treat custom AD as a first-class path. A custom JVP/VJP, implicit derivative,
or replay rule must declare its primal, tangent, cotangent, and derivative-unit
relationships, active parameter coordinates, supported transform directions,
and derivative order. Check its numerical rule against an independent analytic
or finite-difference reference on its stated smooth domain. Preserve any
fail-closed limitation of the underlying rule: a reverse-only custom VJP does
not acquire forward-mode support from a quantity wrapper. See the
[JAX custom-derivative contract](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html)
and Jaxstro's existing [adaptive quadrature replay](../../../20-methods/approximation-integration/adaptive-quadrature.md)
as examples of transform-specific behavior.

Build an adversarial AD matrix for ordinary, scaled dimensionless, and tagged
dimensionless inputs; dimensioned outputs from dimensionless inputs; equal-unit
ratios; mixed-unit parameter trees; zero tangents and nondifferentiated
arguments; affine differences and logarithmic coordinates; conversion scales;
unit-changing branches; and custom-rule compositions under `jit`, `vmap`,
forward/reverse AD, and higher derivatives where claimed. For example,
$\mathrm{d}(x^2)/\mathrm{d}x$ is dimensionless when $x$ is untagged
dimensionless, while $\mathrm{d}(Lx)/\mathrm{d}x$ has length units when $L$
has length units. Dimensionless input does not imply dimensionless output or
derivative. Define dimensionless residuals and likelihood inputs, physical
covariance and Fisher units, and the mapping between physical parameters and
optimizer coordinates. Test smoothness and derivative meaning separately
from syntactic JAX compatibility.

Specify the measure used by a likelihood density. A density carries inverse
observable units and changes its numerical value under a change of data
coordinates: $p_y(y)=p_x(x)|\mathrm{d}x/\mathrm{d}y|$. Its logarithm needs a
declared reference measure or an equivalent dimensionless probability ratio.
An event probability and a correctly transformed physical inference result
must remain invariant. A normalized Gaussian with data and uncertainty
converted together is a proposed diagnostic; Informax owns the likelihood
and its scientific acceptance criterion. Test the same physical prediction,
gradient, and inference calculation in independently chosen compatible input
and output units. Check the chain rule through optimizer coordinates and any
surrogate normalization; numerical parity at one unit choice is insufficient.

**Exit:** a physical derivative ledger and executable comparisons for each
claimed standard or custom AD transform, including dimensionless edge cases
and explicit unsupported directions or orders. Record the primal/tangent/
cotangent maps, likelihood measure, and unit-representation invariance results.
Informax retains inference and observing-design policy; Jaxstro owns only
reusable mathematical and unit contracts.

### Phase 6 — exercise independent ecosystem workflows

Apply the accepted architecture to the Stage 2 survey-depth design. Trace distance,
proper-motion conversion, per-star velocity error, dispersion variance, and
the Fisher input. Keep apparent magnitude and its selection semantics with
the domain owner. Confirm that the added checks preserve the existing
depth-design calculation and its derivatives before considering a new unit
operation.

Exercise all three selected lanes beyond the kinematic example, with separate
owner-approved scientific specifications and acceptance gates:

| Lane | Proposed package path | Distinct quantity and AD stress |
| --- | --- | --- |
| Photometry | Progenax to Fluxax to Informax | Physical parameters, fluxes, logarithmic magnitudes with references, uncertainty, and likelihood. |
| Dynamics | Progenax initial conditions through Gravax evolution to Informax | Evolving state units, time and code-unit conversion, integrator and custom AD boundaries, and inferred observables. |
| Learned surrogate | A qualified physical forward model through a learned approximation to Informax | Input/output normalization, dimensionless network coordinates, uncertainty and domain validity, gradient fidelity, and inference calibration. |

These paths are proposals, not claims that the integrations or learned model
already exist. Each lane needs an explicit data model, normalization,
likelihood, transform boundary, independent reference, and performance record.
Its accepted physical case must also be rerun in compatible units; compare
physical outputs, derivative maps, and the inference quantity under the
declared coordinate and likelihood-measure changes.

The domain owners set the physical approximation, training target, noise model,
and scientific acceptance criterion before an experiment. A lane may use raw
numerical arrays inside a compiled kernel if its `Quantity` boundary and
derivative mapping are explicit and verified.

**Exit:** separate records for all three lanes: implementation parity,
unit-representation invariance, derivative evidence, calibration,
physical-model limits, performance, and a bounded scientific claim. Stage 2
alone is a second use case, not an
independent package consumer or proof of survey realism.

### Phase 7 — qualify performance and the shared contract

The selected policy is to set performance limits per representative workload,
before its qualification run. Each limit names the workload, shape and batch
size, units and semantic kinds, precision, hardware, JAX configuration,
baseline revision, and whether compilation and transfers are included. Record
separate limits for tracing, lowering, backend compilation, warm execution,
memory, and the AD directions that workload uses. Record input magnitude and
conditioning ranges along with dtype, and compare output and derivative
accuracy across those ranges. The numerical limits remain
to be proposed and approved; this page supplies none.

Use matched raw-array and `Quantity` paths that compute the same physical
result and perform the same necessary conversions. Also record the cost of
the public conversion boundary as part of an end-to-end call. Measure repeated
calls with fixed units, calls that vary unit metadata, and representative
`jit`, `vmap`, custom-AD, and multi-device paths actually used. Count traces
and compiled specializations, inspect lowered numerical operations, and block
until device work finishes for warm timings. Separate Python entry and
validation cost from compiled execution. Define nonoverlapping timing
boundaries and cache states for trace, lowering, compilation, and warm calls.
Enumerate the unit sequence before measuring specialization count, including
alternate symbols and algebraic constructions of semantically equivalent
units. A cache-key change may share compilations only after it is shown to
preserve every conversion and semantic rule. Compare the chosen external
unit-library baseline on the same scope. JAX's
[PyTree metadata](https://docs.jax.dev/en/latest/101/pytrees.html),
[JIT specialization](https://docs.jax.dev/en/latest/201/jit.html), and
[benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
motivate these separate measurements.

The first representative workload is the frozen Stage 1 projected-dispersion
prediction, its parameter Jacobian, and the Informax Fisher calculation.
Measure repeated calls at one fixed unit choice and a separate sequence that
changes compatible input units. Use the actual example shapes and numerical
conventions; later repeat this design for the photometry, dynamics, and
surrogate lanes. The owner will set numerical acceptance limits against the
recorded baseline before each qualification run.

The performance hypothesis is that static unit and kind rules may add tracing
or compilation work and new unit specializations, while a fixed-unit warm
kernel may approach raw-array execution if it lowers to the same numerical
operations. Explicit affine, logarithmic, or physical conversions can add
runtime operations when the raw baseline does not already perform them. This
is a prediction, not a measured Jaxstro result. Audit error handling,
serialization, documentation, versioning, downstream migration, and
reproducible evidence. Remove a superseded manual conversion only after
parity is established in its owner package.

**Exit:** a qualified shared contract limited to the operations, transforms,
platforms, and scientific workflows actually tested. A general Scientific AI
claim needs evidence across the independent lanes and an explicit coverage
matrix; passing one lane cannot qualify another. The performance record
includes expected and observed specialization counts, numerical accuracy,
and explicitly delimited timing stages.

### Phase 8 — adopt across the ecosystem in the post-v1 cycle

Inventory every package in the current ecosystem map in `README.md`
and classify its physical public inputs, outputs, state, observables, numerical
kernels, data formats, and AD rules. Set a v2 adoption contract for each package
with its owner. Migrate in dependency order: stabilize Jaxstro semantics first,
then physical producers and observable packages, then inference and consuming
workflows. Keep package releases independently qualified; coordinated v2
adoption does not require a simultaneous release date or identical version
numbers.
State supported producer/consumer version combinations and dependency bounds
for each transition. An unsupported combination must fail clearly before a
physical value is misread. Define an offline artifact migration for saved
arrays that preserves dtype, shape, unit, semantic kind, and any calibration
reference. This converter is distinct from a retained raw physical public API.

For v2, public scientific APIs require `Quantity` for physical values,
including dimensionless physical results and tagged angles. Indices, masks,
random keys, status values, and deliberately normalized optimizer coordinates
are classified by their roles; they are not assigned physical units merely
because they appear beside quantities. `Quantity` is the default representation
through physical calculations, including compiled kernels. A low-level
numerical kernel may use raw arrays when an explicit, tested conversion and
reconstruction boundary is justified. Each package must identify that boundary
and demonstrate the mapping of values, scales, and derivatives, including
custom AD. The program does not require identical internal representations in
every solver or learned component.

For each package, demonstrate unit-correct public interchange, explicit raw
kernel boundaries where used, parity against its v1 numerical path, JAX
transform and custom-AD behavior where claimed, serialization compatibility,
performance, and migration guidance. Migrate one saved v1 scientific artifact
and exercise one producer-v2/consumer-v2 boundary in a clean environment.
Domain packages retain their scientific
models and validity criteria. Release gates for v1 remain unchanged by this
planned migration.

The selected migration policy is a clean major-version cutover. After its
owner's v2 qualification gate passes, a package removes its raw physical
public entry points rather than retaining parallel or deprecated entry points
in that major release. Its migration guide maps each removed signature to the
`Quantity` signature, names the required units and semantic kinds, and records
changes to outputs, AD, and serialized data. Explicit low-level numerical
kernels remain available only under their documented numerical contract. The
v1 release line and its scientific claims remain independently scoped.

**Exit:** an owner-reviewed adoption and qualification record for every
ecosystem package, with exceptions or unsupported operations named. Each
package's v2 physical public API uses the accepted `Quantity` contract and
has a reviewed migration guide with no residual raw public physical path.
Its supported dependency combinations and artifact migration are verified.
Ecosystem coverage is a package-by-package claim, not a conclusion from the
three pilot lanes alone.

## Performance engineering and cost controls

The target is unit-correct computation within an approved cost envelope. A
unit-bearing array does not need a unit object for every element: the current
`Quantity` PyTree carries its value as the dynamic leaf and its unit as static
metadata. The v2 design keeps unit and semantic-kind decisions static during a
compiled call. This can move unit-rule work to tracing, but it can also create
another compiled specialization when the static unit or kind changes. Actual
costs depend on the workload and must be measured.

For a campaign with compiled specializations $s$, the planning model is

```{math}
T_{\mathrm{campaign}}
\approx
\sum_s \left(T_{\mathrm{trace},s}+T_{\mathrm{lower},s}
+T_{\mathrm{compile},s}\right)
+\sum_s N_s\left(T_{\mathrm{entry},s}+T_{\mathrm{warm},s}\right)
+T_{\mathrm{transfer}}.
```

$N_s$ is the number of uses of one compiled specialization. Record peak memory
separately. This model shows why a long simulation reused at fixed units and an
interactive workflow that frequently changes units can have different costs.
Compute spending follows measured billed machine time and the owner's actual
hardware rate; this page assumes no price or acceptable overhead.

The implementation rules to test are:

1. **Bound specialization count.** Use a declared, stable computation basis
   within a workflow. Convert compatible inputs once where possible, then
   carry `Quantity` through the physical calculation. Preserve exact stored
   unit-scale identity and semantic distinctions; floating conversion factors
   still have finite precision. Separate display or provenance metadata from
   execution identity only after proving that it changes no conversion or
   operator rule. A cheaper cache key must never conflate physically distinct
   units or kinds. Measure fixed-unit, changed-unit, and equivalent-unit
   construction paths.
2. **Keep host work out of hot loops.** Resolve parsing and registry lookup
   before tracing, and resolve unit rules at trace time. JIT the useful outer
   calculation and use JAX batching or compiled control flow for repeated
   numerical work.
   Preserve stable compiled function identity rather than recreating a jitted
   callable for every call. Hoist an invariant conversion out of a time step
   or particle loop only after checking that values and derivatives agree.
3. **Keep the compiled graph lean.** `q.math` rules should lower to the
   necessary numerical operations. Inspect jaxpr and lowered operations for
   repeated scale multiplications, host transfers, extra allocations, or
   materialized intermediates. Affine and logarithmic conversions may require
   real numerical work; compare against a raw path that performs the same
   physical conversion.
4. **Treat AD as part of performance.** Measure primal, JVP, VJP, and any
   custom derivative used by the workload. A custom rule or raw-array kernel
   is a performance option only after its physical derivative units, numerical
   agreement, and transform limits pass the same gates as the unitful path.
   Record checkpoint, memory, and compile effects separately from warm speed.
5. **Price the whole workflow.** Include public conversion and validation,
   compiled calls, repeated unit specializations, device transfer, and
   serialization where the workflow actually uses them. A warm inner-kernel
   speedup does not establish lower campaign time or lower compute cost.

Enforce correctness and performance with separate gates. Ordinary CI checks
unit rules, numerical parity, supported transforms, and expected specialization
behavior without a noisy wall-clock threshold. A controlled performance lane
uses pinned revisions, precision, shapes, unit patterns, hardware, and JAX
configuration; it reports trace, lower, compile, blocked warm execution, peak
memory, AD costs, specialization counts, and accuracy by magnitude and dtype
separately. Anna and the package owner set each workload's
numerical limits before qualification. A regression is investigated at the
stage that grew; a cost limit is not relaxed after seeing the result. These
controls follow JAX's [JIT caching guidance](https://docs.jax.dev/en/latest/201/jit.html)
and [benchmarking and profiling guidance](https://docs.jax.dev/en/latest/201/profiling.html).

## Next design decision

The Phase 1 boundary-first choice is settled under ADR 0006. The preferred
end state is Jaxstro-owned; Phase 2 will use external comparisons to inform
that design and identify any reason to revisit it. The hybrid mathematical
interface is selected for Phase 4. Phase 3 targets one public `Quantity`
container with explicit semantic kinds, including affine and logarithmic
representations. Compatible and fully specified conversions are direct;
physical equivalencies and external calibrations require explicit context.
Phase 5 uses a verified hybrid AD interface, including custom derivative
rules and dimensionless edge cases. Phase 6 includes photometry, dynamics,
and learned-surrogate lanes, and Phase 8 targets post-v1 adoption across the
full ecosystem. The v2 public scientific API requires `Quantity` for physical
values, and physical calculations carry it by default. A raw compiled kernel
requires a justified and tested conversion boundary. The v2 release uses a
clean major-version cutover for physical public APIs. The next design choice
is the numerical performance limits for the frozen Stage 1 workload after its
baseline and expected usage pattern are recorded. The selected policy uses
predeclared, workload-specific limits for separate trace, lower, compile,
warm, memory, and AD costs. The Phase 2 architecture decision additionally
requires the affine, logarithmic, mixed-unit, compiled-loop, and direct-JAX
feasibility outcomes. The next executable action remains the Phase 0 baseline
and unit ledger.

For the current representation and its limitations, see
[](../../../30-representations/units-quantities/quantities.md). For the wider
scientific-ML ownership rule, see
[](../../../40-workflows/scientific-ml/ecosystem-boundaries.md).
