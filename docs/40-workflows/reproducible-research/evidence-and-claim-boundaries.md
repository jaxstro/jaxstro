# Evidence and claim boundaries

Use this page when a passing test, generated artifact, provenance record, or
investigation must be translated into a claim without overstating what it
proves.

## Evidence classes

The terms below answer different questions and must not be collapsed.

| Object | Question answered | What it cannot prove |
| --- | --- | --- |
| Scientific contract | What behavior, assumptions, and failure boundary does a callable claim? | That every implementation or downstream use satisfies the contract |
| Validation target | Which executable check probes a named part of the contract? | That untested domains or models are valid |
| Evidence artifact | What metrics and comparisons were emitted under a recorded environment? | That the scientific model is adequate |
| Provenance | Which sources, inputs, methods, hashes, and environment produced or support a record? | Authority, correctness, or sufficiency by itself |
| Investigation | How do prediction, computation, audit, and limitation connect for a bounded example? | General validity outside the fixture |
| Warranted scientific claim | What conclusion is supported after assumptions and evidence are considered? | Any broader conclusion not named by that chain |

## From contract to claim

Let $C$ be a contract, $V$ a set of validation results, $P$ provenance, and $A$
the scientific assumptions. A claim is warranted only within their intersection:

```{math}
:label: eq-workflow-warranted-claim

\mathcal{W}
\subseteq
\mathcal{C}\cap\mathcal{V}\cap\mathcal{P}\cap\mathcal{A}.
```

This is a reasoning relation, not a numerical formula. More passing tests do not
compensate for a false scientific assumption, and detailed provenance does not
turn a weak comparison into strong validation.

## Fail-closed reasoning

Missing evidence remains visible. An unresolved contract ID, stale generated
artifact, failed comparison, absent source, or invalid environment gate must not
be converted into a positive claim. `No standalone indexed artifact` is more
informative than silently inferring evidence from a passing example.

Thresholds belong to the method-specific contract. Tightening or weakening a
tolerance without an error model changes the claim. Informational telemetry
must be distinguished from pass/fail comparisons so that a finite number is not
mistaken for acceptance.

## Audit procedure

1. State the proposed claim in one bounded sentence.
2. Link every noun in the claim to a contract or declared scientific assumption.
3. Identify independent validation targets and their tested domain.
4. Inspect artifact freshness, metric definitions, units, thresholds, and
   environment policy.
5. Inspect runtime and source provenance separately.
6. List known limitations and negative results.
7. Narrow the claim until every remaining clause is supported.

## Worked boundary

A rootfinding artifact can show residual, bracket width, status, and evaluation
count for benchmark fixtures. A gradient artifact can compare an implicit rule
with analytic and finite-difference sensitivities. Together with a uniqueness
and smoothness argument they may support the fixture's local derivative. They
do not prove that an arbitrary physical residual has one branch or that a
downstream parameter is identifiable.

## Where the claim stops

Evidence infrastructure makes reasoning inspectable. It does not automate the
scientist's responsibility to judge source adequacy, model assumptions, or
external validity.

## Connected ideas

Use [](../../10-foundations/models-and-computation/models-inference-information.md),
[](../../20-methods/change-constraints-evolution/rootfinding.md),
[](./provenance.md), and [](../investigations/investigations.md).
