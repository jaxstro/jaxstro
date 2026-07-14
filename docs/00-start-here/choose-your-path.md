---
title: Choose your foundations path
description: A task-routed refresher for connected mathematical and scientific ideas.
---

# Choose your foundations path

This is a route finder, not a gate. Use it to locate the idea that would make
today's research work easier, then return whenever you need it.

## If you are computation-first

You may already write Python and JAX but want the mathematical meaning behind
the code. Start with
[](../10-foundations/mathematical-objects/linear-algebra-language-of-change.md)
and [](../10-foundations/mathematical-objects/what-is-a-derivative.md), then
connect control flow to the derivative of the executed program. Ask: *what
mathematical object does this array represent, and which claim does this
transform support?*

## If astronomy is your strongest language

Start with
[](../10-foundations/mathematical-objects/functions-units-scales.md) and
[](../10-foundations/models-and-computation/what-is-a-model.md).
Newtonian gravity, luminosity,
parallax, spectra, and stellar-mass distributions supply familiar physical
questions while we make parameter space, probability, and sensitivity explicit.
The astronomy is teaching context; the reusable numerical primitive remains
science-general.

## If statistics and inference are your strongest language

Start with
[](../10-foundations/mathematical-objects/probability-and-distributions.md),
then
[](../10-foundations/models-and-computation/models-inference-information.md).
Connect a gradient of a log likelihood to the broader meaning of a derivative
as a local linear sensitivity. Keep the physical model, measurement model, and
inferential assumptions separate.

## If you want the complete first-principles path

You do not need to identify with a strongest-language category. Read
[](../10-foundations/foundations.md), then follow its sequence from functions
and models through linear algebra, derivatives, probability, inference,
conditioning, and programs.

## If you are a returning researcher

Choose by the failure you are facing:

- units or scales feel unstable ->
  [](../10-foundations/mathematical-objects/functions-units-scales.md);
- too many parameters behave alike ->
  [](../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md);
- AD and finite differences disagree ->
  [](../10-foundations/mathematical-objects/what-is-a-derivative.md);
- a precise fit feels scientifically wrong ->
  [](../10-foundations/models-and-computation/models-inference-information.md);
- JAX returns a surprising gradient ->
  [](../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md).

## A five-minute self-check

For your current problem, can you name:

1. the observable and its units;
2. the model inputs, parameters, and state;
3. one limiting case or invariant;
4. the derivative you actually want;
5. an independent audit;
6. the strongest claim those checks would justify?

Any unanswered item is a route, not a verdict. Continue through the complete
path above or to [](./first-research-calculation.md).
