"""Sampler-consistent path-space atom evidence, with an oracle-score control.

The estimator writes the likelihood ratio of two finite-step reverse chains and
reweights.  It does not assume that the learned score is a gradient field, does
not integrate it along a path in parameter space, and builds no intermediate
ensemble.

Setting.  Protocol A's reverse step from ``t_hi`` down to ``t_lo`` is

    z' = z + dt d2 s(z, t_hi) + sqrt(dt d2) eps,
    dt = t_hi - t_lo,   d2 = sigma(t_hi)^2 2 log(sigma_max / sigma_min),

so ``P`` (base score) and ``Q_k`` (atom-``k`` transported score) are Gaussian
kernels with the same covariance and different means.  Their log ratio along a
trajectory drawn from ``Q_k`` telescopes into

    log dP/dQ_k = log p_T(z_T) / q_{k,T}(z_T)
                + sum_n [ sqrt(dt d2) eps_n . (s_P - s_Q) - (dt d2 / 2) ||s_P - s_Q||^2 ],

and because Protocol A starts every method from the base model's own terminal
law, the terminal term is zero when the proposal starts there, as the sampler
does.  The general terminal term (``log_terminal_ratio``) is kept for callers
that start the proposal elsewhere.

The estimator is then importance sampling for the sampler-induced atom evidence

    Z_k^sampler = E_P[phi_k(theta_clean)] = E_{Q_k}[ phi_k(theta_clean) dP/dQ_k ],

whose variance is small when Spectra's transport is a good proposal for the
tilted chain.  A high ESS indicates that the transported chain tracks
``p_0 phi_k``; a collapsed ESS indicates that path importance sampling cannot
estimate the mass reliably at this step count and dimension.

Oracle control.  On Gaussian Linear the noisy base score is analytic, so the
same chain can be run with ``s_P`` replaced by the exact score and ``s_Q`` by
the exact score pushed through the same transport formula.  Grid,
discretisation, terminal convention, weight formula and budget are unchanged;
only the score differs.  Four outcomes are distinguished:

* oracle also collapses -> path-IS scaling barrier;
* oracle fine and learned collapses -> learned proposal mismatch;
* both fine but learned answer biased -> learned global-mass bias;
* neither stable -> the representation is wrong for this setting.
"""

from __future__ import annotations

import functools
import time
import weakref
from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from spectra import atoms as atoms_mod
from spectra import e2e


@dataclass
class PathResult:
    """One ``(atom, seed)`` path-space run."""

    atom: int
    seed: int
    num_trajectories: int
    num_steps: int
    log_weight: np.ndarray  # (N,) log dP/dQ_k along each trajectory
    log_phi: np.ndarray  # (N,) log phi_k(theta_clean)
    samples: np.ndarray  # (N, d) clean draws of the proposal chain
    log_z: float  # log Zhat_k
    diagnostics: dict = field(default_factory=dict)
    wall_clock_s: float = float("nan")


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


def weight_diagnostics(log_u: np.ndarray) -> dict:
    """Every weight diagnostic, from one log-weight vector."""
    log_u = np.asarray(log_u, float)
    finite = np.isfinite(log_u)
    n = int(log_u.shape[0])
    out = {
        "n": n,
        "n_finite": int(finite.sum()),
        "frac_finite": float(finite.mean()),
    }
    if finite.sum() == 0:
        out.update({"log_w_mean": float("nan"), "log_w_var": float("nan"),
                    "ess": 0.0, "ess_normalized": 0.0,
                    "max_normalized_weight": float("nan"), "status": "nonfinite"})
        return out
    lu = log_u[finite]
    lse1 = _logsumexp(lu)
    lse2 = _logsumexp(2.0 * lu)
    ess = float(np.exp(2.0 * lse1 - lse2))
    w = np.exp(lu - lse1)
    out.update({
        "log_w_mean": float(np.mean(lu)),
        "log_w_var": float(np.var(lu, ddof=1)) if lu.size > 1 else 0.0,
        "log_w_q01": float(np.quantile(lu, 0.01)),
        "log_w_q25": float(np.quantile(lu, 0.25)),
        "log_w_median": float(np.median(lu)),
        "log_w_q75": float(np.quantile(lu, 0.75)),
        "log_w_q99": float(np.quantile(lu, 0.99)),
        "log_w_max": float(np.max(lu)),
        "log_w_min": float(np.min(lu)),
        "ess": ess,
        "ess_normalized": ess / n,
        "max_normalized_weight": float(np.max(w)),
        "top10_weight_share": float(np.sort(w)[-10:].sum()) if n >= 10 else 1.0,
        "status": "ok",
    })
    return out


# Compiled proposal-chain kernels, keyed by the two score functions.  As in
# ``e2e.make_sampler``: ``lax.scan`` caches its traced jaxpr on a weak reference
# to the body function object, so a ``step`` closure rebuilt on every call would
# miss the cache and XLA would recompile inside every timed repeat.
# ``make_atom_chain`` builds the body once per ``(score_p, score_q, schedule,
# cfg)`` and caches it.
_CHAIN_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _chain_key(score_q, schedule, cfg: e2e.SamplerConfig) -> tuple:
    try:
        sched = hash(schedule)
    except TypeError:  # pragma: no cover - schedules are frozen dataclasses
        sched = id(schedule)
    return (id(score_q), sched, cfg)


def make_atom_chain(score_p: Callable, score_q: Callable, *, schedule,
                    cfg: e2e.SamplerConfig):
    """The compiled proposal-chain kernel for one atom.

    Returns ``kernel(x_init, diffusion_noise, lw0) -> (z, log_w)`` on device.
    Repeated calls return the same callable holding the same scan body, so JAX
    reuses its traced jaxpr and compiled executable.  ``score_q`` must be the
    same object across calls for the cache to hit; ``learned_scores`` and
    ``oracle_scores`` memoise their per-atom closures for this reason.
    """
    if cfg.langevin_steps:
        raise ValueError("path-space weights are only defined for the "
                         "corrector-free Protocol A chain (langevin_steps=0)")
    key = _chain_key(score_q, schedule, cfg)
    try:
        by_key = _CHAIN_CACHE.setdefault(score_p, {})
    except TypeError:  # unhashable / non-weakref-able callable: no caching
        by_key = {}
    if key in by_key:
        kernel = by_key[key]
        kernel.reused = True
        return kernel

    ts = e2e.time_grid(schedule, cfg)
    log_ratio = schedule.log_ratio
    t_hi = ts[::-1][:-1]
    t_lo = ts[::-1][1:]

    def step(carry, payload):
        z, lw = carry
        th, tl, eps = payload
        dt = th - tl
        d2 = schedule.sigma(th) ** 2 * (2.0 * log_ratio)
        sp = score_p(z, th)
        sq = score_q(z, th)
        diff = sp - sq
        sd = jnp.sqrt(dt * d2)
        lw = lw + sd * jnp.sum(eps * diff, axis=-1) - 0.5 * dt * d2 * jnp.sum(
            diff * diff, axis=-1)
        z = z + dt * d2 * sq + sd * eps
        return (z, lw), None

    def run(x_init, diffusion_noise, lw0):
        (z, lw), _ = jax.lax.scan(step, (x_init, lw0), (t_hi, t_lo, diffusion_noise))
        return z, lw

    run.reused = False
    by_key[key] = run
    return run


def run_atom_chain(
    score_p: Callable,
    score_q: Callable,
    *,
    schedule,
    cfg: e2e.SamplerConfig,
    x_init: jnp.ndarray,
    diffusion_noise: jnp.ndarray,
    log_terminal_ratio: Optional[jnp.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``Q`` and accumulate ``log dP/dQ``; returns ``(samples, log_w)``.

    ``score_p`` and ``score_q`` are both evaluated at the proposal's state, as
    the change of measure requires.  ``cfg.langevin_steps`` must be zero: a
    Langevin corrector is not a Markov kernel of the form assumed above, so the
    chain raises instead of accumulating a wrong weight.

    The numerical work runs inside the cached kernel of :func:`make_atom_chain`;
    this wrapper only pulls the result back to the host, so a caller that times
    it still pays compilation on a cache miss.  Timing probes call the kernel.
    """
    kernel = make_atom_chain(score_p, score_q, schedule=schedule, cfg=cfg)
    lw0 = (jnp.zeros(x_init.shape[0]) if log_terminal_ratio is None
           else jnp.asarray(log_terminal_ratio))
    z, lw = kernel(x_init, diffusion_noise, lw0)
    z.block_until_ready()
    return np.asarray(jax.device_get(z)), np.asarray(jax.device_get(lw))


def pathspace_evidence(
    score_p: Callable,
    make_score_q: Callable,
    *,
    schedule,
    cfg: e2e.SamplerConfig,
    atoms,
    atom: int,
    terminal_mean,
    terminal_std,
    num_trajectories: int,
    seed: int,
    log_terminal_ratio_fn: Optional[Callable] = None,
) -> PathResult:
    """``log Zhat_k`` from one proposal chain, with its full weight diagnostics.

    ``make_score_q(k)`` returns the transported score of atom ``k``; it is a
    factory rather than a bound function so the caller cannot accidentally pair
    atom ``i``'s weights with atom ``j``'s chain.
    """
    key = jax.random.PRNGKey(seed)
    k_init, k_noise = jax.random.split(key)
    x_init = e2e.draw_terminal(k_init, num_trajectories, terminal_mean, terminal_std)
    diff, _ = e2e.make_noise(k_noise, cfg.num_steps, num_trajectories,
                             int(np.asarray(terminal_mean).shape[-1]), 0)
    ltr = None if log_terminal_ratio_fn is None else log_terminal_ratio_fn(x_init)

    t0 = time.perf_counter()
    samples, log_w = run_atom_chain(
        score_p, make_score_q(atom), schedule=schedule, cfg=cfg, x_init=x_init,
        diffusion_noise=diff, log_terminal_ratio=ltr)
    wall = time.perf_counter() - t0

    lphi = atoms_mod.log_phi(samples, atoms)[:, atom]
    log_u = lphi + log_w
    good = np.isfinite(log_u)
    log_z = (_logsumexp(log_u[good]) - np.log(num_trajectories)
             if good.any() else float("nan"))

    res = PathResult(
        atom=atom, seed=seed, num_trajectories=num_trajectories,
        num_steps=cfg.num_steps, log_weight=log_w, log_phi=lphi,
        samples=samples, log_z=log_z, wall_clock_s=wall,
    )
    # two weight vectors are diagnosed: the raw change of measure, which says
    # whether the two chains stay close, and the product with phi_k, which is
    # what the estimator averages.
    res.diagnostics["weight"] = weight_diagnostics(log_w)
    res.diagnostics["integrand"] = weight_diagnostics(log_u)
    res.diagnostics["log_phi_mean"] = float(np.mean(lphi[np.isfinite(lphi)]))
    res.diagnostics["frac_finite_samples"] = float(
        np.isfinite(samples).all(axis=-1).mean())
    return res


def combine_log_odds(results: dict, atoms, i: int = 0, j: int = 1) -> dict:
    """``log(b_i Z_i) - log(b_j Z_j)`` from two independent proposal chains."""
    zi, zj = results[i].log_z, results[j].log_z
    lo = float(atoms.log_b[i] + zi - atoms.log_b[j] - zj)
    return {
        "log_z_i": zi, "log_z_j": zj, "log_odds": lo,
        "delta": float(zi - zj),
        "ess_norm_i": results[i].diagnostics["integrand"]["ess_normalized"],
        "ess_norm_j": results[j].diagnostics["integrand"]["ess_normalized"],
    }


# ------------------------------------------------------- score factories ----


def learned_scores(backbone, x_o, atoms):
    """``(s_P, make_s_Q)`` from the frozen model, with the Spectra hook's transport."""
    sch = backbone.schedule
    a_all = jnp.asarray(atoms.a)
    kap_all = jnp.asarray(atoms.kappa)

    def score_p(z, t):
        return backbone.score_theta_t(z, t, x_o)

    @functools.lru_cache(maxsize=None)
    def make_score_q(k: int):
        """Memoised so repeated calls return the same closure.

        ``make_atom_chain`` keys its compiled-kernel cache on this object; a
        fresh closure per call would miss the cache and recompile.
        """
        a = a_all[k]
        kap = kap_all[k]

        def score_q(z, t):
            tau = sch.sigma(t) ** 2
            D = 1.0 + kap * tau
            m = (z + tau * a) / D
            t_rho = t - 0.5 * jnp.log(D) / sch.log_ratio
            return (a - kap * z) / D + backbone.score_theta_t(m, t_rho, x_o) / D

        return score_q

    return score_p, make_score_q


def oracle_scores(schedule, atoms, base_mean, base_var):
    """``(s_P, make_s_Q)`` for a Gaussian base posterior, analytic.

    ``s_Q`` pushes the analytic base score through the same transport formula
    the learned run uses, rather than using the tilted Gaussian's score
    directly, so the control differs from the learned run only in the score
    field.  That the two forms agree is checked separately by
    :func:`atom_score_parity`.
    """
    m_b = jnp.asarray(np.asarray(base_mean, float))
    v_b = float(base_var)
    a_all = jnp.asarray(atoms.a)
    kap_all = jnp.asarray(atoms.kappa)

    def base_score_tau(z, tau):
        return -(z - m_b) / (v_b + tau)

    def score_p(z, t):
        return base_score_tau(z, schedule.sigma(t) ** 2)

    @functools.lru_cache(maxsize=None)
    def make_score_q(k: int):
        """Memoised for the same reason as in :func:`learned_scores`."""
        a = a_all[k]
        kap = kap_all[k]

        def score_q(z, t):
            tau = schedule.sigma(t) ** 2
            D = 1.0 + kap * tau
            m = (z + tau * a) / D
            rho = tau / D
            return (a - kap * z) / D + base_score_tau(m, rho) / D

        return score_q

    return score_p, make_score_q


def atom_score_parity(schedule, atoms, base_mean, base_var, *, taus, rng,
                      num: int = 64) -> dict:
    """Transported-oracle score vs the tilted Gaussian's own score.

    ``q_k ∝ N(m_b, v_b I) N(nu_k, I / kappa_k)`` is Gaussian with
    ``1 / v_k = 1 / v_b + kappa_k``, so its noisy score is available directly.
    Agreement to machine precision checks the exact-atom transport identity in
    the same formula the path-space run uses.
    """
    m_b = np.asarray(base_mean, float)
    v_b = float(base_var)
    d = m_b.shape[0]
    _, make_q = oracle_scores(schedule, atoms, m_b, v_b)
    worst = 0.0
    for k in range(atoms.num_atoms):
        v_k = 1.0 / (1.0 / v_b + atoms.kappa[k])
        m_k = v_k * (m_b / v_b + atoms.kappa[k] * atoms.nu[k])
        sq = make_q(k)
        for tau in taus:
            t = float(schedule.t_of_tau(jnp.asarray(tau)))
            z = jnp.asarray(m_k + np.sqrt(v_k + tau) * rng.standard_normal((num, d)))
            got = np.asarray(sq(z, t), float)
            want = np.asarray(-(z - jnp.asarray(m_k)) / (v_k + tau), float)
            worst = max(worst, float(np.max(np.abs(got - want))))
    return {"max_abs_score_gap": worst, "num_taus": len(list(taus))}
