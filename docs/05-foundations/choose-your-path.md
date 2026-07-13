---
title: Choose your foundations path
description: An ungraded, task-routed refresher for uneven preparation.
---

# Choose your foundations path

This is **ungraded** and **not a placement test**. It does not decide whether you
belong in a course or research project. Use it to locate the idea that would
make today's work easier, then return whenever you need it.

## If you are computation-first

You may already write Python and JAX but want the mathematical meaning behind
the code. Start with [](../10-theory/linear-algebra.md) and
[](../10-theory/autodiff.md), then connect control flow to the derivative of the
executed program. Ask: *what mathematical object does this array represent, and
which claim does this transform support?*

## If astronomy is your strongest language

Start with [](../10-theory/quantities.md) and
[](../10-theory/science-patterns.md). Newtonian gravity, luminosity,
parallax, spectra, and stellar-mass distributions supply familiar physical
questions while we make parameter space, probability, and sensitivity explicit.
The astronomy is teaching context; the reusable numerical primitive remains
science-general.

## If statistics and inference are your strongest language

Start with [](../10-theory/distributions.md), then use
[](../20-architecture/quantity-system.md) as an example of separating a model
contract from adoption evidence. Connect a gradient of a log likelihood to the
broader meaning of a derivative as a local linear sensitivity. Keep the physical
model, measurement model, and inferential assumptions separate.

## If you want the complete first-principles path

You do not need to identify with a strongest-language category. Read
[](./foundations.md), then follow the current sequence through
[](../10-theory/quantities.md), [](../10-theory/linear-algebra.md),
[](../10-theory/autodiff.md), and [](../10-theory/distributions.md). The dedicated
foundation pages will make these connections progressively; these existing
module pages are the resolvable route while that sequence is being built.

## If you are a returning researcher

Choose by the failure you are facing:

- units or scales feel unstable → [](../10-theory/quantities.md);
- too many parameters behave alike → [](../10-theory/linear-algebra.md);
- AD and finite differences disagree → [](../10-theory/autodiff.md);
- a precise fit feels scientifically wrong → [](../10-theory/distributions.md);
- JAX returns a surprising gradient → [](../10-theory/rootfinding.md).

## A five-minute self-check

For your current problem, can you name:

1. the observable and its units;
2. the model inputs, parameters, and state;
3. one limiting case or invariant;
4. the derivative you actually want;
5. an independent audit;
6. the strongest claim those checks would justify?

Any unanswered item is a route, not a verdict. Continue through the complete
path above or to [](../00-getting-started/how-to-learn.md).
