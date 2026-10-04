"""SLCP as a native task of this repo, in Simformer working coordinates.

Upstream dispatches ``get_task("slcp")`` to ``SBIBMTask``, whose constructor
accepts only ``two_moons`` / ``gaussian_linear`` / ``gaussian_linear_high``, and
ships no SLCP priors, observations, checkpoints or references, so SLCP is
defined here.  Its posterior is strongly non-Gaussian for a structural reason:
the likelihood depends on ``theta_3`` and ``theta_4`` only through their
squares, so the base posterior has four sign-symmetric modes and a test prior
that is not sign-symmetric has to move mass between them.

Definition (sbibm ``slcp``, no distractors), in physical coordinates:

    theta ~ U(-3, 3)^5
    m     = (theta_1, theta_2)
    s1    = theta_3^2,  s2 = theta_4^2,  rho = tanh(theta_5)
    S     = [[s1^2 + eps, rho s1 s2], [rho s1 s2, s2^2 + eps]],   eps = 1e-6
    x     = 4 i.i.d. draws from N(m, S), flattened to 8 numbers.

Working coordinates.  The diffusion runs on ``y = theta_phys / 3``, so the
model-space training prior is ``U([-1, 1]^5)`` and the whole uniform-task
convention of this repo (isotropic Gaussian test priors with
``sigma = 0.2 (high - low) / sqrt(12) = 0.11547``, atoms defined in model space)
transfers unchanged.  The map is affine with a single common scale, so isotropy
survives it and the exact-atom family is intact; the coordinate check in
:mod:`spectra.coordinates` records this.

The likelihood is written here in closed form, so SLCP references never need a
surrogate: they are produced by tempered SMC against the exact density.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THETA_DIM = 5
X_DIM = 8
NUM_DATA = 4
PHYS_LOW, PHYS_HIGH = -3.0, 3.0
COV_EPS = 1e-6
MODEL_SCALE = 3.0  # theta_phys = MODEL_SCALE * y

# Standardisation of x, computed once from simulations under the training prior
# and stored in ``slcp_x_moments.json``.  It follows the upstream pattern of a
# constant affine map on the conditioned variable; it does not touch the theta
# coordinate and so does not affect the atom algebra.  Filled in by
# :func:`_load_moments` at import time so the constants live in one place.
X_MEAN = np.zeros(X_DIM)
X_STD = np.ones(X_DIM)
_MOMENTS_PATH = Path(__file__).with_name("slcp_x_moments.json")


def _load_moments() -> None:
    global X_MEAN, X_STD
    if _MOMENTS_PATH.is_file():
        blob = json.loads(_MOMENTS_PATH.read_text())
        X_MEAN = np.asarray(blob["x_mean"], float)
        X_STD = np.asarray(blob["x_std"], float)


_load_moments()


def to_physical(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, float) * MODEL_SCALE


def to_model(theta_phys: np.ndarray) -> np.ndarray:
    return np.asarray(theta_phys, float) / MODEL_SCALE


def covariance(theta_phys: np.ndarray) -> np.ndarray:
    """``S(theta)``, shape ``(N, 2, 2)``, exactly as sbibm builds it."""
    t = np.atleast_2d(np.asarray(theta_phys, float))
    s1 = t[:, 2] ** 2
    s2 = t[:, 3] ** 2
    rho = np.tanh(t[:, 4])
    S = np.empty((t.shape[0], 2, 2))
    S[:, 0, 0] = s1**2 + COV_EPS
    S[:, 1, 1] = s2**2 + COV_EPS
    S[:, 0, 1] = rho * s1 * s2
    S[:, 1, 0] = S[:, 0, 1]
    return S


def simulate(theta_model: np.ndarray, rng) -> np.ndarray:
    """Raw (unstandardised) ``x``, shape ``(N, 8)``.

    The four data points are flattened in row-major order, matching sbibm's
    ``reshape((num_samples, 8))``.
    """
    phys = to_physical(np.atleast_2d(theta_model))
    S = covariance(phys)
    L = np.linalg.cholesky(S)
    n = phys.shape[0]
    eps = rng.standard_normal((n, NUM_DATA, 2))
    pts = phys[:, None, :2] + np.einsum("nij,ndj->ndi", L, eps)
    return pts.reshape(n, X_DIM)


def standardise(x_raw: np.ndarray) -> np.ndarray:
    return (np.asarray(x_raw, float) - X_MEAN) / X_STD


def unstandardise(x_std: np.ndarray) -> np.ndarray:
    return np.asarray(x_std, float) * X_STD + X_MEAN


def log_likelihood(theta_model: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """Exact ``log p(x | theta)``, up to a ``theta``-independent constant.

    ``x`` is four i.i.d. bivariate normals with the same mean and covariance, so
    the log-likelihood is ``-2 log|S| - (1/2) sum_d (x_d - m)^T S^{-1} (x_d - m)``
    plus the standardisation Jacobian, which is constant in ``theta``.
    """
    phys = to_physical(np.atleast_2d(theta_model))
    x = unstandardise(np.asarray(x_std, float).reshape(-1)).reshape(NUM_DATA, 2)
    S = covariance(phys)
    a, b, c = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
    det = a * c - b * b
    bad = ~(det > 0)
    det = np.where(bad, 1.0, det)
    resid = x[None, :, :] - phys[:, None, :2]  # (N, 4, 2)
    r0, r1 = resid[:, :, 0], resid[:, :, 1]
    # inverse of a 2x2, applied without forming it
    quad = (c[:, None] * r0**2 - 2.0 * b[:, None] * r0 * r1
            + a[:, None] * r1**2) / det[:, None]
    out = -0.5 * quad.sum(axis=1) - 0.5 * NUM_DATA * np.log(det)
    return np.where(bad, -np.inf, out)


# ------------------------------------------------------- priors / obs ----


def training_prior_json() -> dict:
    return {
        "task": "slcp", "dist": "uniform", "type": "training",
        "low_ori": [PHYS_LOW] * THETA_DIM, "high_ori": [PHYS_HIGH] * THETA_DIM,
        "low": [-1.0] * THETA_DIM, "high": [1.0] * THETA_DIM,
        "model_coordinate_note": (
            "theta_phys = 3 * y; the diffusion, the atoms and every mu/sigma "
            "below are in the y (model) coordinate"),
    }


def generate_priors(seed: int = 0, num: int = 10) -> dict:
    """Test priors under the upstream uniform-task recipe, in model space.

    Follows ``experiments/data/priors/gen_priors_uniform.py``
    (``sigma_mixture = 0.2 (high - low) / sqrt(12)``, component means uniform on
    ``[low + 3 sigma, high - 3 sigma]``, ``pi ~ U(0.2, 0.8)``), so SLCP's priors
    are drawn from the same family as Two Moons' and OUP's.
    """
    import jax
    import jax.numpy as jnp

    low = jnp.full((THETA_DIM,), -1.0)
    high = jnp.full((THETA_DIM,), 1.0)
    key = jax.random.PRNGKey(seed)
    out = {}
    for i in range(num):
        key, key_mild, key_strong, key_mixture = jax.random.split(key, 4)
        sigma_mild = 0.5 * (high - low) / jnp.sqrt(12)
        mu_mild = jax.random.uniform(key=key_mild, shape=sigma_mild.shape,
                                     minval=low + 3 * sigma_mild,
                                     maxval=high - 3 * sigma_mild)
        sigma_strong = 0.2 * (high - low) / jnp.sqrt(12)
        mu_strong = jax.random.uniform(key=key_strong, shape=sigma_strong.shape,
                                       minval=low + 3 * sigma_strong,
                                       maxval=high - 3 * sigma_strong)
        sigma_mixture = 0.2 * (high - low) / jnp.sqrt(12)
        k1, k2 = jax.random.split(key_mixture)
        mu1 = jax.random.uniform(key=k1, shape=sigma_mixture.shape,
                                 minval=low + 3 * sigma_mixture,
                                 maxval=high - 3 * sigma_mixture)
        mu2 = jax.random.uniform(key=k2, shape=sigma_mixture.shape,
                                 minval=low + 3 * sigma_mixture,
                                 maxval=high - 3 * sigma_mixture)
        pi = jax.random.uniform(key=key_mixture, minval=0.2, maxval=0.8)
        out[f"mild_{i}"] = {"task": "slcp", "dist": "gaussian", "type": "mild",
                            "mu": mu_mild.tolist(), "sigma": sigma_mild.tolist()}
        out[f"strong_{i}"] = {"task": "slcp", "dist": "gaussian", "type": "strong",
                              "mu": mu_strong.tolist(),
                              "sigma": sigma_strong.tolist()}
        out[f"mixture_{i}"] = {
            "task": "slcp", "dist": "mixture", "type": "mixture",
            "mu": jnp.array([mu1, mu2]).tolist(),
            "sigma": jnp.array([sigma_mixture, sigma_mixture]).tolist(),
            "pi": jnp.array([pi, 1 - pi]).tolist()}
    return out


OBSERVATION_SEEDS = (1000000, 1000001, 1000002, 1000003, 1000004,
                     1000005, 1000010, 1000012, 1000008, 1000009)


def sample_truncated_mixture(prior: dict, rng, num: int) -> np.ndarray:
    """Draw from the test prior truncated to the training box, in model space.

    Upstream's observation generator uses a truncated normal / truncated mixture
    for this reason: an observation whose ``theta_true`` sits outside the
    training support would be unreachable for every method by construction.
    """
    mu = np.atleast_2d(np.asarray(prior["mu"], float))
    sigma = np.atleast_2d(np.asarray(prior["sigma"], float))
    pi = np.asarray(prior.get("pi", [1.0]), float)
    out = np.empty((num, THETA_DIM))
    filled = 0
    while filled < num:
        k = rng.choice(len(pi), size=num, p=pi / pi.sum())
        cand = mu[k] + sigma[k] * rng.standard_normal((num, THETA_DIM))
        keep = cand[np.all((cand >= -1.0) & (cand <= 1.0), axis=-1)]
        take = min(num - filled, keep.shape[0])
        out[filled:filled + take] = keep[:take]
        filled += take
    return out


def generate_observation(prior: dict, obs_seed: int) -> dict:
    """One ``(theta_true, x)`` pair, mirroring upstream's per-seed convention."""
    rng = np.random.default_rng(obs_seed)
    theta = sample_truncated_mixture(prior, rng, 1)[0]
    x_raw = simulate(theta[None, :], rng)[0]
    return {"theta": theta.tolist(), "x": standardise(x_raw).tolist(),
            "x_raw": x_raw.tolist(), "obs_seed": int(obs_seed)}


# ------------------------------------------------------------ tempered SMC ----


@dataclass
class SMCResult:
    samples: np.ndarray
    log_evidence: float
    stages: list
    num_particles: int
    seed: int
    diagnostics: dict


SIGN_FLIPS = ((1.0, 1.0, -1.0, 1.0, 1.0),
              (1.0, 1.0, 1.0, -1.0, 1.0),
              (1.0, 1.0, -1.0, -1.0, 1.0))


def sign_flip_move(theta, log_lik, beta, rng, log_prior_fn):
    """Exact mode-jumping move from a symmetry of the SLCP likelihood.

    ``s_1 = theta_3^2`` and ``s_2 = theta_4^2`` enter the density only through
    their squares, so flipping the sign of either parameter leaves
    ``p(x | theta)`` unchanged.  A Metropolis proposal that flips signs
    therefore has an acceptance ratio of ``pi(theta') / pi(theta)`` with no
    likelihood term, and under the symmetric uniform training prior it is
    accepted with probability one.

    Without this move a random-walk chain has to cross a likelihood valley to
    reach the mirror mode and in practice does not: base posteriors then put up
    to 97% of their mass on one sign where the exact answer is 50%.
    """
    # One flip per particle.  A single flip for the whole ensemble would be
    # accepted for every particle under the symmetric training prior, so the
    # ensemble would reflect as a block and any sign imbalance would remain
    # (base posteriors up to 0.44 away from the exact 0.5 occupancy);
    # independent flips remove it.
    table = np.asarray(SIGN_FLIPS, float)
    flip = table[rng.integers(table.shape[0], size=theta.shape[0])]
    prop = theta * flip
    lp_prop = log_prior_fn(prop)
    lp_cur = log_prior_fn(theta)
    # both endpoints can be -inf for a particle already outside the support;
    # -inf - -inf is NaN, so those cases are handled separately
    with np.errstate(invalid="ignore"):
        log_acc = np.where(np.isfinite(lp_prop) & np.isfinite(lp_cur),
                           lp_prop - lp_cur,
                           np.where(np.isfinite(lp_prop), np.inf, -np.inf))
    take = np.log(rng.random(theta.shape[0])) < log_acc
    return np.where(take[:, None], prop, theta), log_lik, float(take.mean())


def tempered_smc(log_lik_fn, log_prior_fn, sample_prior, *, num_particles: int,
                 rng, target_ess: float = 0.5, num_mcmc: int = 8,
                 step_scale: float = 0.5, max_stages: int = 400,
                 symmetry_move=None) -> SMCResult:
    """Adaptive tempered SMC from the prior to ``prior * lik``.

    Chosen over rejection sampling because the SLCP likelihood is unbounded
    (``s_1 = theta_3^2 -> 0`` sends the density to infinity), so no valid
    envelope exists, and over a single MCMC chain because the posterior has four
    sign-symmetric modes that a random-walk chain will not cross.  The
    temperature ladder is adapted to hold the particle ESS at ``target_ess``,
    each stage resamples systematically and rejuvenates with a random-walk
    Metropolis kernel whose scale is adapted to the current particle covariance.

    Returns the normalising constant as well, which is what makes an independent
    check of the atom evidences possible on this task.
    """
    theta = sample_prior(num_particles)
    log_lik = log_lik_fn(theta)
    beta = 0.0
    log_w = np.zeros(num_particles)
    log_evidence = 0.0
    stages = []
    while beta < 1.0 and len(stages) < max_stages:
        lo, hi = beta, 1.0
        for _ in range(60):  # bisect on the next temperature
            mid = 0.5 * (lo + hi)
            inc = (mid - beta) * log_lik
            m = np.max(inc)
            w = np.exp(inc - m)
            ess = (w.sum() ** 2) / np.sum(w**2)
            if ess < target_ess * num_particles:
                hi = mid
            else:
                lo = mid
        next_beta = min(1.0, hi if hi < 1.0 else 1.0)
        if next_beta <= beta:
            next_beta = min(1.0, beta + 1e-4)
        inc = (next_beta - beta) * log_lik
        m = np.max(inc)
        w = np.exp(inc - m)
        log_evidence += m + math.log(w.mean())
        ess = (w.sum() ** 2) / np.sum(w**2)
        # systematic resampling
        p = w / w.sum()
        pos = (rng.random() + np.arange(num_particles)) / num_particles
        idx = np.searchsorted(np.cumsum(p), pos)
        theta = theta[idx]
        log_lik = log_lik[idx]
        beta = next_beta

        cov = np.cov(theta.T) + 1e-10 * np.eye(theta.shape[1])
        L = np.linalg.cholesky(cov) * step_scale
        acc_total = 0.0
        for _ in range(num_mcmc):
            prop = theta + rng.standard_normal(theta.shape) @ L.T
            lp_prop = log_prior_fn(prop)
            ok = np.isfinite(lp_prop)
            ll_prop = np.full(num_particles, -np.inf)
            if ok.any():
                ll_prop[ok] = log_lik_fn(prop[ok])
            log_acc = (beta * ll_prop + lp_prop) - (beta * log_lik
                                                    + log_prior_fn(theta))
            take = np.log(rng.random(num_particles)) < log_acc
            theta = np.where(take[:, None], prop, theta)
            log_lik = np.where(take, ll_prop, log_lik)
            acc_total += float(take.mean())
        flip_acc = float("nan")
        if symmetry_move is not None:
            theta, log_lik, flip_acc = symmetry_move(theta, log_lik, beta, rng,
                                                     log_prior_fn)
        stages.append({"beta": float(beta), "ess": float(ess),
                       "acceptance": acc_total / num_mcmc,
                       "flip_acceptance": flip_acc})
    return SMCResult(
        samples=theta, log_evidence=float(log_evidence), stages=stages,
        num_particles=num_particles, seed=-1,
        diagnostics={
            "num_stages": len(stages),
            "min_stage_ess": float(min(s["ess"] for s in stages)) if stages else float("nan"),
            "mean_acceptance": float(np.mean([s["acceptance"] for s in stages]))
            if stages else float("nan"),
            "final_beta": float(beta),
            "unique_particles": int(len(np.unique(theta, axis=0))),
            "symmetry_move": symmetry_move is not None,
            "mean_flip_acceptance": (
                float(np.nanmean([s["flip_acceptance"] for s in stages]))
                if stages and symmetry_move is not None else float("nan")),
        })


def box_log_prior(low=-1.0, high=1.0):
    def fn(theta):
        inside = np.all((theta >= low) & (theta <= high), axis=-1)
        return np.where(inside, 0.0, -np.inf)
    return fn


def mixture_log_prior(prior: dict, box=(-1.0, 1.0)):
    """``log pi_new`` on all of R^d, without restriction to the box (target convention)."""
    mu = np.atleast_2d(np.asarray(prior["mu"], float))
    sigma = np.atleast_2d(np.asarray(prior["sigma"], float))
    pi = np.asarray(prior.get("pi", [1.0]), float)
    log_pi = np.log(pi / pi.sum())

    def fn(theta):
        theta = np.atleast_2d(theta)
        parts = []
        for k in range(mu.shape[0]):
            z = (theta - mu[k]) / sigma[k]
            parts.append(log_pi[k] - 0.5 * np.sum(z**2, axis=-1)
                         - np.sum(np.log(sigma[k]))
                         - 0.5 * THETA_DIM * math.log(2 * math.pi))
        parts = np.stack(parts, 0)
        m = parts.max(0)
        return m + np.log(np.exp(parts - m).sum(0))

    return fn
