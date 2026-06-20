"""Shared AD-vs-FD gradient-audit engine (single source of truth for sibling gates).

``audit_entry_point(case)`` returns an ``AuditResult`` whose ``status`` is COMPUTED from
(expect, finite, |ratio-1|<tol, |ad|>eps) -- never hand-set. The same results feed each
sibling's pytest gate (``tests/validation/test_grad_audit.py``) and its
``scripts/audit_gradients.py`` -> results.json emitter. Reverse-mode ``jax.grad`` only
(ODE custom_vjp-safe).

This engine is the deduplicated core extracted verbatim (no behaviour change) from the
byte-identical copies that lived in fluxax and progenax under
``tests/validation/grad_audit/core.py``. The two origins differed only in docstrings and in
the per-package ``Direction`` label literal (fluxax: photometry/image; progenax: IC/summary);
``direction`` is never enforced by the engine -- it is stored as a free ``str`` on the result --
so the shared engine keeps ``Direction`` as a generic string alias and each package supplies
its own label strings in its local case registry.

Dependency-light by design: jax + stdlib only (no pytest at import time), so it ships with the
installed package and siblings import it at test time.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Tuple, cast

import jax
import jax.numpy as jnp

from jaxstro.testing.contracts import (
    GradContract,
    default_contract_for_expect,
    is_grad_contract,
)

# direction is a free-form, package-specific label (e.g. fluxax "params->photometry",
# progenax "params->IC"); the engine never validates it, so it is a generic string here.
Direction = str
Expect = Literal["consistent", "known_zero", "known_blocked"]


@dataclass(frozen=True)
class EdgeConfig:
    """A curated boundary probe for a Case (e.g. a low-mass end or grid-node crossing)."""

    label: str  # appears in the case id, e.g. "M=0.1"
    theta0: float  # the edge parameter value
    hazard_id: str | None = None  # links to the hazard map; set if it probes a suspect
    tol: float | None = None  # per-edge tolerance override
    expect: Expect | None = None  # per-edge expect override


@dataclass(frozen=True)
class Case:
    id: str
    direction: Direction
    fn: Callable[[jax.Array], jax.Array]  # theta (scalar) -> output array
    param: str
    theta0: float
    reduce: Callable[[jax.Array], jax.Array] = jnp.sum  # output -> scalar
    expect: Expect = "consistent"
    tol: float = 1e-3
    h_rel: float = 1e-4
    eps: float = 1e-9  # |AD| silent-zero threshold
    edges: Tuple[EdgeConfig, ...] = ()
    hazard_id: str | None = None  # set when confirmed-but-unfixed -> strict-xfail
    grad_contract: GradContract | None = None
    allowed_claim: str = ""
    forbidden_claims: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        contract = self.grad_contract
        if contract is None:
            contract = default_contract_for_expect(self.expect)
        elif not is_grad_contract(contract):
            raise ValueError(f"unknown gradient contract: {contract!r}")
        object.__setattr__(self, "grad_contract", contract)
        object.__setattr__(self, "forbidden_claims", tuple(self.forbidden_claims))


@dataclass(frozen=True)
class AuditResult:
    id: str
    direction: str
    param: str
    theta: float
    finite: bool
    ad: float
    fd: float
    ratio: float
    abs_ad: float
    expect: str
    tol: float
    status: str  # clean | known-limitation | hazard
    grad_contract: str = ""
    allowed_claim: str = ""
    forbidden_claims: Tuple[str, ...] = ()


def _scalar(case: Case, theta: jax.Array) -> jax.Array:
    return case.reduce(case.fn(theta))


def _classify(expect, finite, ad, fd, ratio, tol, eps, grad_contract) -> str:
    if grad_contract == "surrogate":
        if expect == "known_zero":
            return "known-limitation" if (abs(ad) < eps and abs(fd) < eps) else "hazard"
        if expect == "known_blocked":
            return "known-limitation" if finite else "hazard"
        return "clean" if (finite and abs(ad) > eps) else "hazard"
    if expect == "known_zero":
        # BOTH |AD|~0 AND |FD|~0 for a genuine known-limitation. AD~0 with a LIVE FD means
        # the value genuinely moves with the param while the gradient is silently zero -> a
        # hazard (the unannounced-blocked-gradient detector). known_zero cases must pick
        # theta0 off any grid-node crossing so FD~0 holds for the genuinely-constant quantity.
        return "known-limitation" if (abs(ad) < eps and abs(fd) < eps) else "hazard"
    if expect == "known_blocked":
        return "known-limitation" if finite else "hazard"
    # consistent
    if finite and abs(ad) > eps and abs(ratio - 1.0) < tol:
        return "clean"
    return "hazard"


def _effective_grad_contract(case: Case, expect: str) -> GradContract:
    case_contract = cast(GradContract, case.grad_contract)
    if expect != case.expect and case_contract == default_contract_for_expect(
        case.expect
    ):
        return default_contract_for_expect(expect)
    return case_contract


def audit_entry_point(
    case: Case,
    theta: float | None = None,
    tol: float | None = None,
    expect: str | None = None,
) -> AuditResult:
    theta = case.theta0 if theta is None else theta
    tol = case.tol if tol is None else tol
    expect = case.expect if expect is None else expect

    from typing import get_args

    assert (expect if expect is not None else case.expect) in get_args(Expect), (
        f"unknown expect class: {expect!r}"
    )
    grad_contract = _effective_grad_contract(case, expect)

    t = jnp.asarray(theta, dtype=jnp.float64)
    ad = float(jax.grad(lambda x: _scalar(case, x))(t))

    h = case.h_rel * max(abs(float(theta)), 1.0)

    def g(x):
        return float(_scalar(case, jnp.asarray(x, dtype=jnp.float64)))

    fd = (g(float(theta) + h) - g(float(theta) - h)) / (2.0 * h)

    finite = bool(jnp.isfinite(jnp.asarray(ad)))
    if fd != 0.0:
        ratio = ad / fd
    else:
        ratio = 1.0 if ad == 0.0 else float("inf")
    status = _classify(expect, finite, ad, fd, ratio, tol, case.eps, grad_contract)
    return AuditResult(
        id=case.id,
        direction=case.direction,
        param=case.param,
        theta=float(theta),
        finite=finite,
        ad=ad,
        fd=fd,
        ratio=ratio,
        abs_ad=abs(ad),
        expect=expect,
        tol=tol,
        status=status,
        grad_contract=grad_contract,
        allowed_claim=case.allowed_claim,
        forbidden_claims=case.forbidden_claims,
    )
