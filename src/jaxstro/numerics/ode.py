"""Fixed-step ODE helpers built from JAX scans."""

from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

Rhs = Callable[[Float[Array, "..."], Float[Array, ""]], Float[Array, "..."]]


class ODEResult(NamedTuple):
    """State history returned by first-order fixed-step solvers."""

    t: Float[Array, " steps"]
    y: Float[Array, " steps ..."]


class VerletResult(NamedTuple):
    """Position and velocity history from velocity-Verlet integration."""

    t: Float[Array, " steps"]
    q: Float[Array, " steps ..."]
    v: Float[Array, " steps ..."]


def euler_step(
    rhs: Rhs,
    y: Float[Array, "..."],
    t: Float[Array, ""],
    dt: float,
) -> Float[Array, "..."]:
    """Take one explicit Euler step."""
    return y + dt * rhs(y, t)


def midpoint_step(
    rhs: Rhs,
    y: Float[Array, "..."],
    t: Float[Array, ""],
    dt: float,
) -> Float[Array, "..."]:
    """Take one explicit midpoint / RK2 step."""
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    return y + dt * k2


def rk4_step(
    rhs: Rhs,
    y: Float[Array, "..."],
    t: Float[Array, ""],
    dt: float,
) -> Float[Array, "..."]:
    """Take one classical fourth-order Runge-Kutta step."""
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _scan_first_order(
    step: Callable[
        [Rhs, Float[Array, "..."], Float[Array, ""], float], Float[Array, "..."]
    ],
    rhs: Rhs,
    y0: Float[Array, "..."],
    t0: float,
    dt: float,
    num_steps: int,
) -> ODEResult:
    y0 = jnp.asarray(y0)
    t0_arr = jnp.asarray(t0, dtype=y0.dtype)

    def body(carry, _):
        y, t = carry
        y_next = step(rhs, y, t, dt)
        t_next = t + dt
        return (y_next, t_next), y_next

    (_, _), ys = jax.lax.scan(body, (y0, t0_arr), None, length=num_steps)
    y_hist = jnp.concatenate([y0[jnp.newaxis, ...], ys], axis=0)
    t_hist = t0_arr + dt * jnp.arange(num_steps + 1, dtype=y0.dtype)
    return ODEResult(t=t_hist, y=y_hist)


def euler(
    rhs: Rhs,
    *,
    y0: Float[Array, "..."],
    t0: float,
    dt: float,
    num_steps: int,
) -> ODEResult:
    """Integrate a first-order ODE with fixed-step explicit Euler."""
    return _scan_first_order(euler_step, rhs, y0, t0, dt, num_steps)


def midpoint(
    rhs: Rhs,
    *,
    y0: Float[Array, "..."],
    t0: float,
    dt: float,
    num_steps: int,
) -> ODEResult:
    """Integrate a first-order ODE with fixed-step explicit midpoint / RK2."""
    return _scan_first_order(midpoint_step, rhs, y0, t0, dt, num_steps)


def rk4(
    rhs: Rhs,
    *,
    y0: Float[Array, "..."],
    t0: float,
    dt: float,
    num_steps: int,
) -> ODEResult:
    """Integrate a first-order ODE with fixed-step classical RK4."""
    return _scan_first_order(rk4_step, rhs, y0, t0, dt, num_steps)


def solve_fixed_step(
    rhs: Rhs,
    *,
    y0: Float[Array, "..."],
    t0: float,
    dt: float,
    num_steps: int,
    method: Literal["euler", "midpoint", "rk2", "rk4"] = "rk4",
) -> ODEResult:
    """Dispatch to a named fixed-step first-order ODE method."""
    if method == "euler":
        return euler(rhs, y0=y0, t0=t0, dt=dt, num_steps=num_steps)
    if method in {"midpoint", "rk2"}:
        return midpoint(rhs, y0=y0, t0=t0, dt=dt, num_steps=num_steps)
    if method == "rk4":
        return rk4(rhs, y0=y0, t0=t0, dt=dt, num_steps=num_steps)
    msg = f"unknown fixed-step ODE method: {method!r}"
    raise ValueError(msg)


def velocity_verlet(
    acceleration: Rhs,
    *,
    q0: Float[Array, "..."],
    v0: Float[Array, "..."],
    t0: float,
    dt: float,
    num_steps: int,
) -> VerletResult:
    """Integrate a separable second-order system with velocity-Verlet."""
    q0 = jnp.asarray(q0)
    v0 = jnp.asarray(v0)
    t0_arr = jnp.asarray(t0, dtype=q0.dtype)

    def body(carry, _):
        q, v, t = carry
        a0 = acceleration(q, t)
        q_next = q + dt * v + 0.5 * dt**2 * a0
        t_next = t + dt
        a1 = acceleration(q_next, t_next)
        v_next = v + 0.5 * dt * (a0 + a1)
        return (q_next, v_next, t_next), (q_next, v_next)

    (_, _, _), (qs, vs) = jax.lax.scan(
        body,
        (q0, v0, t0_arr),
        None,
        length=num_steps,
    )
    q_hist = jnp.concatenate([q0[jnp.newaxis, ...], qs], axis=0)
    v_hist = jnp.concatenate([v0[jnp.newaxis, ...], vs], axis=0)
    t_hist = t0_arr + dt * jnp.arange(num_steps + 1, dtype=q0.dtype)
    return VerletResult(t=t_hist, q=q_hist, v=v_hist)


__all__ = [
    "ODEResult",
    "VerletResult",
    "euler_step",
    "midpoint_step",
    "rk4_step",
    "euler",
    "midpoint",
    "rk4",
    "solve_fixed_step",
    "velocity_verlet",
]
