"""Exact structured transport: the isotropic exponential--quadratic family.

For ``r(theta) ∝ exp(a^T theta - kappa/2 ||theta||^2)`` with ``1 + kappa tau > 0``,
completing the square against the VE kernel gives, for any base clean posterior,

    D = 1 + kappa tau,   rho = tau / D,   m = (z + tau a) / D,

    s_q(z, tau, x) = (a - kappa z) / D + s_p(m, rho, x) / D.

Each reverse step therefore needs one base-score query at a transformed state
and noise level, and neither a reverse-conditional reconstruction nor a score
Jacobian.  ``T0/E1`` is ``kappa = 0`` and ``E2`` is
``a = mu / lam, kappa = 1 / lam``; both are covered by :func:`tq_target_score`
and are checked against their own closed forms in ``tests/test_transport.py``.

Nothing here clamps: when the transformed effective noise leaves the range the
score model was trained on, :func:`effective_noise_report` records it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TransportQuery:
    """The transformed query at one noise level, plus what has to be logged."""

    D: jnp.ndarray
    rho: jnp.ndarray
    m: jnp.ndarray


def tq_pieces(z: jnp.ndarray, tau, a: jnp.ndarray, kappa) -> TransportQuery:
    D = 1.0 + kappa * tau
    return TransportQuery(D=D, rho=tau / D, m=(z + tau * a) / D)


def tq_target_score(score_fn, z: jnp.ndarray, tau, a: jnp.ndarray, kappa):
    """``s_q`` for the exponential--quadratic family.

    ``score_fn(m, rho)`` must return the base score at state ``m`` and noise
    variance ``rho``; the caller decides how ``x_o`` is bound.
    """
    q = tq_pieces(z, tau, a, kappa)
    return (a - kappa * z) / q.D + score_fn(q.m, q.rho) / q.D, q


def e2_params(mu: jnp.ndarray, lam) -> tuple[jnp.ndarray, float]:
    """``r ∝ N(theta; mu, lam I)`` expressed in the exponential--quadratic family."""
    return mu / lam, 1.0 / lam


def e2_target_score(score_fn, z: jnp.ndarray, tau, mu: jnp.ndarray, lam):
    """The isotropic localisation written out directly, for cross-checking."""
    rho = tau * lam / (tau + lam)
    m = (lam * z + tau * mu) / (tau + lam)
    return (mu - z) / (tau + lam) + (lam / (tau + lam)) * score_fn(m, rho)


def t0_target_score(score_fn, z: jnp.ndarray, tau, a: jnp.ndarray):
    """Pure exponential tilt, the ``kappa = 0`` boundary of the family."""
    return a + score_fn(z + tau * a, tau)


def guard_ok(tau, kappa) -> bool:
    return bool(np.all(1.0 + np.asarray(kappa) * np.asarray(tau) > 0.0))


def min_guard_D(schedule, t_range, kappa) -> float:
    """Smallest ``D = 1 + kappa tau`` over the interval the sampler will use.

    ``D`` is affine in ``tau`` and ``sigma`` is increasing in ``t``, so over
    ``t_range`` the minimum sits at one endpoint: at ``t_min`` when ``kappa > 0``
    and at ``t_max`` when ``kappa < 0``.  Both endpoints are evaluated, so a
    negative ``kappa`` is handled correctly.
    """
    t_min, t_max = float(t_range[0]), float(t_range[1])
    taus = np.array([float(schedule.sigma(t_min) ** 2), float(schedule.sigma(t_max) ** 2)])
    return float(np.min(1.0 + np.asarray(kappa, float)[..., None] * taus))


def require_guard(schedule, t_range, kappa, what: str) -> float:
    """``min_guard_D``, refusing an interval on which the identity does not hold.

    ``D <= 0`` makes ``rho = tau / D`` non-positive and ``t_rho`` undefined, so
    the run would produce NaN rather than a wrong-but-finite answer.  Checked
    once when the hook is built, where ``kappa`` and the schedule are concrete;
    nothing is added inside the traced step.

    This is the mathematical domain, computed in double precision.  It does not
    guarantee that every accepted ``kappa`` is numerically safe: within roughly
    float32 epsilon of the boundary, ``D`` can round to zero inside the sampler
    and the hook still returns non-finite values.  Every prior shipped with this
    package has ``kappa > 0``, far from that edge.
    """
    d = min_guard_D(schedule, t_range, kappa)
    if not d > 0.0:
        raise ValueError(
            f"{what}: the exact-transport guard fails on this schedule "
            f"(min D = 1 + kappa*tau = {d:.6g} <= 0 over t in "
            f"[{float(t_range[0]):.6g}, {float(t_range[1]):.6g}]).  The "
            f"exponential--quadratic identity needs D > 0 at every noise level "
            f"the sampler visits; kappa must exceed -1/max(tau).")
    return d


def effective_noise_report(taus, kappa, schedule) -> dict:
    """Where the transformed queries land relative to the trained noise range.

    ``taus`` is the sampler's own noise grid.  Returned in diffusion-time units
    as well, because that is what the network is conditioned on.
    """
    taus = np.asarray(taus, float)
    rhos = taus / (1.0 + float(kappa) * taus)
    t_rho = np.asarray(schedule.t_of_tau(jnp.asarray(rhos)), float)
    return {
        "rho_min": float(rhos.min()),
        "rho_median": float(np.median(rhos)),
        "rho_max": float(rhos.max()),
        "t_rho_min": float(t_rho.min()),
        "t_rho_max": float(t_rho.max()),
        "t_nominal_min": float(schedule.t_min),
        "t_nominal_max": float(schedule.t_max),
        "n_below_t_min": int((t_rho < schedule.t_min).sum()),
        "n_above_t_max": int((t_rho > schedule.t_max).sum()),
        "frac_out_of_range": float(
            ((t_rho < schedule.t_min) | (t_rho > schedule.t_max)).mean()
        ),
    }
