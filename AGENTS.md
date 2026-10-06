# AGENTS.md (Codex) - jaxstro

Guidance for Codex when working in `jaxstro`.

## Writing

Remove all mannered prose. State the rule, the measurement and the mechanism in plain
sentences. No rhetorical emphasis, no asides, no metaphors, no dramatised failure stories;
the number, the date and the mechanism carry the point. This applies to chat, docstrings,
commit messages, `STATUS.md` and everything under `docs/`.

## Read First
- `CLAUDE.md`
- `README.md`
- `pyproject.toml`

## Units Policy
- `DEFAULT_UNITS`: `CGS` — jaxstro is the domain-agnostic foundation, so its default is the
  physics-pure base all `UnitSystem`s are built from. Downstream *domain* packages set their own
  `DEFAULT_UNITS` (e.g. gravax/progenax use `STELLAR` = `ASTRO_DYNAMICAL`).
- Core APIs require explicit units or explicit `G`.
- Convenience wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`.

## JAX Rules
- Use `jax.numpy`, `jax.lax.scan`, `jax.jit`, `jax.vmap`.
- Keep utilities domain-agnostic and differentiable.

## Testing
- `pytest`
- `ruff check src/ && ruff format src/`
- `mypy`

## Brain hub - this repo is a spoke of ~/brain (read-only from here)

- Never edit `~/brain` from this session; it is a pull-only hub.
- Capture durable notes with `brain "what happened - short, factual"`.
- Capture cross-project insight with `brain "xref: <insight> - touches <other project / paper>"`.
- For focused work, pull context with `/brain-pack jaxstro`.
- For full protocol, read `~/brain/AGENTS.md` and `~/brain/guide/`; keep this file as the short spoke reminder.

<!-- brain-handshake: keep in sync with ~/brain/guide/how-to/set-up-a-project.md#spoke-stanza -->

<!-- brain-status-convention -->
## Brain status updates
When you make notable progress, hit a blocker, or set the next action, update this repo's `STATUS.md` (`next:` / `blocker:` / `due:` lines) — the brain pulls it into the portfolio dashboard + standup via `federate.py` (see `~/brain/work/meta/status-convention.md`). Brain stays pull-only: never hand-edit `~/brain`; capture events with `brain "…"`.

## Test design: quality over quantity (Anna, 2026-10-05)

This rule is identical in the jaxstro-dev root and in every package's `CLAUDE.md` and `AGENTS.md`; keep the
wording the same everywhere.

A test exists to catch a named failure. A few strong tests are better than many weak ones. Inefficient test
design is not allowed in any jaxstro package, and each package's suite is to be brought under this rule file by
file.

**Every test**

- names in its docstring the failure it catches, and fails when that failure is introduced; check this once by
  mutation when it is not obvious;
- asserts against an independent value: an analytic or limiting solution, a vendored reference fixture, a
  conservation law, or a bound whose derivation is stated. A rerun of the same code path is not a reference;
- sets its tolerance from the quantity its error scales with, a few times the measured error. A near-zero
  difference is compared against the scale its roundoff comes from, not against itself (gravax: the contact gap
  against the contact radius, 2026-10-05);
- runs in seconds. A test that needs longer is marked `slow` and says why.

**Compile cost (JAX)**

- Do not call a function that contains `lax.scan`, `fori_loop`, `while_loop` or `cond` eagerly in a test. Call it
  through one module-level `jax.jit` with its static arguments named. An eager call rebuilds the loop body and
  compiles a new program on every call: gravax `test_reach.py` made 575 XLA compiles, 235 after the change, and
  its cold-cache wall went from 17.0 s to 9.2 s (2026-10-05).
- Values that vary between cases (thresholds, spans, seeds, eta, distances) are traced inputs of one compiled
  program: sweep them with `vmap` or `lax.map` inside one `jit`, or parametrize over traced values. A static
  field that differs between tests costs one compile per test.
- Tests in a file share static configurations. Build each distinct configuration once (a module-scoped fixture
  or a shared configuration module) and reuse it. A new static configuration needs a stated reason. Count the
  distinct compiled programs of a file when adding tests to it.
- An expensive result (an integration, a routed state, a replay) is computed once in a module-scoped fixture and
  checked by several tests, not recomputed in each.
- The persistent compilation cache stores every program: `jax_persistent_cache_min_compile_time_secs = 0` and
  `jax_persistent_cache_min_entry_size_bytes = 0` in `tests/conftest.py`. With a 0.5 s floor the small programs
  were recompiled on every run (gravax `test_interaction_policy.py`: 24.0 s to 7.1 s of wall for 4.6 MB of
  cache, 2026-10-05).

**Removal and hardening**

- Delete a test that can no longer fail: its scenario is unreachable, it duplicates another test, or it pins an
  implementation detail with no failure mode. Removing a weak test is progress, not lost coverage.
- Harden a loose test: replace "runs", "is finite" or "changed" with the quantitative claim and its tolerance.
- Never weaken a test to make it pass. Fix the code, or fix the test's premise and say why.

**Measure before changing a slow file.** Record per test the trace, lowering and XLA compile seconds, the cache
hits and the programs compiled (`jax.monitoring` duration listeners and `jax_log_compiles`). Gravax, 2026-10-05,
563 hierarchy unit tests with the cache off: XLA compile was 73% of the wall and execution 12%; 232 distinct
programs took over 0.3 s each to compile.
