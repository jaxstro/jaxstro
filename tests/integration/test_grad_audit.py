"""The shared grad-audit engine (jaxstro.testing) must CATCH broken gradients
(NaN / silent-zero / wrong-by-2x) and correctly classify the intentional-limitation
expect-classes. These toy functions are the engine's proof — they touch no domain physics.

Ported verbatim (engine semantics) from the byte-identical fluxax/progenax
``grad_audit/core.py`` engine that this module deduplicates.
"""

from typing import get_args

import jax
import jax.numpy as jnp

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # float64: FD/AD ratios must be tight

from jaxstro.testing import (  # noqa: E402  (after float64 enable)
    RESERVED_GRAD_CONTRACTS,
    AuditResult,
    Case,
    EdgeConfig,
    GradContract,
    audit_entry_point,
    contract_requires_fd,
    default_contract_for_expect,
    is_inference_ready,
)


def _case(
    fn,
    expect="consistent",
    tol=1e-5,
    reduce=jnp.sum,
    grad_contract=None,
    allowed_claim="",
    forbidden_claims=(),
):
    return Case(
        id="toy",
        direction="params->summary",
        fn=fn,
        param="x",
        theta0=2.0,
        reduce=reduce,
        expect=expect,
        tol=tol,
        grad_contract=grad_contract,
        allowed_claim=allowed_claim,
        forbidden_claims=forbidden_claims,
    )


def test_exports_present():
    # The public surface the siblings depend on.
    assert callable(audit_entry_point)
    assert Case is not None and AuditResult is not None and EdgeConfig is not None
    assert callable(is_inference_ready)
    assert "smooth_pathwise" in get_args(GradContract)


def test_clean_linear_is_clean():
    r = audit_entry_point(_case(lambda x: jnp.array([3.0 * x, 5.0 * x])))
    assert r.finite and r.status == "clean"
    assert abs(r.ratio - 1.0) < 1e-6 and r.abs_ad > 1e-9
    assert r.grad_contract == "smooth_pathwise"
    assert is_inference_ready(r)


def test_legacy_case_construction_defaults_contract_metadata():
    c = _case(lambda x: jnp.array([x]))
    assert c.grad_contract == "smooth_pathwise"
    assert c.allowed_claim == ""
    assert c.forbidden_claims == ()

    r = audit_entry_point(c)
    assert r.grad_contract == "smooth_pathwise"
    assert r.allowed_claim == ""
    assert r.forbidden_claims == ()


def test_legacy_audit_result_positional_construction_stays_supported():
    r = AuditResult(
        "toy",
        "params->summary",
        "x",
        2.0,
        True,
        1.0,
        1.0,
        1.0,
        1.0,
        "consistent",
        1e-3,
        "clean",
    )
    assert r.grad_contract == ""
    assert not is_inference_ready(r)


def test_expect_defaults_map_to_contracts():
    assert default_contract_for_expect("consistent") == "smooth_pathwise"
    assert default_contract_for_expect("known_zero") == "known_zero"
    assert default_contract_for_expect("known_blocked") == "known_blocked"
    assert _case(lambda x: x, expect="consistent").grad_contract == "smooth_pathwise"
    assert _case(lambda x: x, expect="known_zero").grad_contract == "known_zero"
    assert _case(lambda x: x, expect="known_blocked").grad_contract == "known_blocked"


def test_explicit_surrogate_and_validation_only_overrides_work():
    surrogate = _case(lambda x: jnp.array([x]), expect="consistent")
    surrogate = Case(
        id=surrogate.id,
        direction=surrogate.direction,
        fn=surrogate.fn,
        param=surrogate.param,
        theta0=surrogate.theta0,
        reduce=surrogate.reduce,
        expect=surrogate.expect,
        tol=surrogate.tol,
        h_rel=surrogate.h_rel,
        eps=surrogate.eps,
        grad_contract="surrogate",
        allowed_claim="live surrogate sensitivity",
        forbidden_claims=("FD equality",),
    )
    validation = Case(
        id="validation",
        direction="params->summary",
        fn=lambda x: jnp.array([x]),
        param="x",
        theta0=2.0,
        grad_contract="validation_only",
    )

    assert surrogate.grad_contract == "surrogate"
    assert surrogate.forbidden_claims == ("FD equality",)
    assert validation.grad_contract == "validation_only"


def test_silent_zero_is_hazard():
    # stop_gradient -> AD is 0 but FD is non-zero: the headline failure mode.
    r = audit_entry_point(_case(lambda x: jax.lax.stop_gradient(3.0 * x) * jnp.ones(2)))
    assert r.status == "hazard" and r.abs_ad < 1e-12
    assert not is_inference_ready(r)


def test_nan_grad_is_hazard():
    # sqrt(x - x) has a 0/0 grad -> NaN; the engine must not call it clean.
    r = audit_entry_point(_case(lambda x: jnp.sqrt(x - x) * x * jnp.ones(1)))
    assert (not r.finite) and r.status == "hazard"


def test_wrong_by_two_is_hazard():
    # AD double-counts: a custom_jvp that lies about the derivative.
    @jax.custom_jvp
    def f(x):
        return x * jnp.ones(1)

    f.defjvp(lambda p, t: (f(p[0]), 2.0 * t[0] * jnp.ones(1)))  # claims 2x
    r = audit_entry_point(_case(f))
    assert r.status == "hazard" and abs(r.ratio - 2.0) < 1e-5


def test_surrogate_requires_live_gradient_but_not_ad_fd_equality():
    @jax.custom_jvp
    def f(x):
        return x * jnp.ones(1)

    f.defjvp(lambda p, t: (f(p[0]), 2.0 * t[0] * jnp.ones(1)))
    r = audit_entry_point(_case(f, tol=1e-5, grad_contract="surrogate"))
    assert r.status == "clean"
    assert r.grad_contract == "surrogate"
    assert abs(r.ratio - 2.0) < 1e-5
    assert r.finite and r.abs_ad > 1e-9
    assert not is_inference_ready(r)


def test_surrogate_silent_zero_is_hazard():
    r = audit_entry_point(
        _case(
            lambda x: jax.lax.stop_gradient(3.0 * x) * jnp.ones(2),
            grad_contract="surrogate",
        )
    )
    assert r.status == "hazard" and r.abs_ad < 1e-12


def test_known_zero_pins_zero_gradient():
    # An intentionally constant-in-x output: AD=0, FD=0 -> known-limitation, NOT hazard.
    r = audit_entry_point(_case(lambda x: jnp.ones(2), expect="known_zero"))
    assert r.status == "known-limitation"
    assert r.grad_contract == "known_zero"
    assert not is_inference_ready(r)


def test_known_zero_flags_if_gradient_appears():
    # If a 'known_zero' case SUDDENLY has a gradient, that is a hazard (unannounced change).
    r = audit_entry_point(_case(lambda x: 3.0 * x * jnp.ones(2), expect="known_zero"))
    assert r.status == "hazard"


def test_known_zero_with_live_fd_is_hazard():
    # AD blocked at 0 but FD non-zero => the value genuinely depends on x while the
    # gradient is silently zero. Requires |ad|<eps AND |fd|<eps for a known-limitation;
    # a live FD must be a HAZARD (unannounced change).
    r = audit_entry_point(
        _case(
            lambda x: jax.lax.stop_gradient(3.0 * x) * jnp.ones(2), expect="known_zero"
        )
    )
    assert r.status == "hazard" and r.abs_ad < 1e-12


def test_known_blocked_requires_only_finite():
    r = audit_entry_point(
        _case(lambda x: jax.lax.stop_gradient(x) * jnp.ones(2), expect="known_blocked")
    )
    assert (
        r.status == "known-limitation"
    )  # finite AD (0) is acceptable for a blocked site
    assert r.grad_contract == "known_blocked"
    assert not is_inference_ready(r)


def test_validation_only_is_not_inference_ready():
    r = audit_entry_point(
        _case(lambda x: jnp.array([3.0 * x]), grad_contract="validation_only")
    )
    assert r.status == "clean"
    assert r.grad_contract == "validation_only"
    assert not is_inference_ready(r)


def test_inference_ready_fails_closed_for_unknown_reserved_and_non_clean():
    clean = audit_entry_point(_case(lambda x: jnp.array([3.0 * x])))
    assert is_inference_ready(clean)

    unknown = AuditResult(
        id="unknown",
        direction="params->summary",
        param="x",
        theta=2.0,
        finite=True,
        ad=1.0,
        fd=1.0,
        ratio=1.0,
        abs_ad=1.0,
        expect="consistent",
        tol=1e-3,
        status="clean",
        grad_contract="stochastic_reparam",
    )
    missing = object()
    non_clean = audit_entry_point(
        _case(lambda x: jax.lax.stop_gradient(3.0 * x) * jnp.ones(1))
    )

    assert not is_inference_ready(unknown)
    assert not is_inference_ready(missing)
    assert not is_inference_ready(non_clean)
    for reserved in RESERVED_GRAD_CONTRACTS:
        reserved_result = AuditResult(
            id=reserved,
            direction="params->summary",
            param="x",
            theta=2.0,
            finite=True,
            ad=1.0,
            fd=1.0,
            ratio=1.0,
            abs_ad=1.0,
            expect="consistent",
            tol=1e-3,
            status="clean",
            grad_contract=reserved,
        )
        assert not is_inference_ready(reserved_result)


def test_reserved_future_names_are_not_live_contracts():
    live = set(get_args(GradContract))
    assert live == {
        "smooth_pathwise",
        "known_zero",
        "known_blocked",
        "surrogate",
        "validation_only",
    }
    assert not (live & set(RESERVED_GRAD_CONTRACTS))
    for reserved in RESERVED_GRAD_CONTRACTS:
        assert not contract_requires_fd(reserved)


def test_testing_subpackage_is_top_level_discoverable():
    # jaxstro.testing is committed public surface (the shared grad-audit gate the
    # siblings import). It must be bound on the top-level namespace + __all__ like
    # every other public subpackage, so `import jaxstro; jaxstro.testing` works.
    import jaxstro

    assert hasattr(jaxstro, "testing")
    assert "testing" in jaxstro.__all__
    assert jaxstro.testing.audit_entry_point is audit_entry_point
