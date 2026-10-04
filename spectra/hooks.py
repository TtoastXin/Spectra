"""Target-score hooks: one per method, all plugged into the same reverse driver.

Each factory returns ``(hook, ops, meta)``:

``hook(z, t)``
    ``z`` is ``(B, theta_dim)`` and ``t`` a scalar diffusion time; it returns
    ``(s_q, aux)`` with ``aux`` a dict of per-step scalars that the driver stacks.
``ops``
    how many of each expensive operator one hook call issues, for the compute
    accounting (score forwards alone do not describe PG-FullCov).
``meta``
    what has to reach the run manifest.

The guidance algebra itself is reused from :mod:`spectra.methods` rather than
rewritten batched.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from spectra import methods
from spectra.learned_eval import PSD_EPS, psd_project
from spectra.transport import require_guard, tq_pieces

UPSTREAM_X0_CLIP = 50.0


def _norm_mean(v):
    return jnp.mean(jnp.linalg.norm(v, axis=-1))


# ----------------------------------------------------------------- base ----


def make_base_hook(backbone, x_o):
    """Base Simformer: ``s_q = s_p``.  The adaptation-gain reference."""

    def hook(z, t):
        s = backbone.score_theta_t(z, t, x_o)
        return s, {"score_norm": _norm_mean(s)}

    return hook, {"base_score": 1}, {"method": "base"}


# ------------------------------------------------- exact structured (TQ) ---


def make_tq_hook(backbone, x_o, a, kappa):
    """Exact transport for ``r ∝ exp(a^T theta - kappa/2 ||theta||^2)``.

    One transformed base-score query per step; no guidance is formed and no
    second query at the current state is made.
    """
    a = jnp.asarray(a)
    kappa = float(kappa)
    sch = backbone.schedule
    t_min, t_max = backbone.nominal_t_range()
    guard_min_D = require_guard(sch, (t_min, t_max), kappa, "exact transport (tq)")

    def hook(z, t):
        tau = sch.sigma(t) ** 2
        q = tq_pieces(z, tau, a, kappa)
        # sigma(t_rho) = sqrt(tau / D) = sigma(t) / sqrt(D), and sigma is
        # log-linear in t, so the transformed query time is an exact shift:
        #     t_rho = t - log(D) / (2 log(sigma_max / sigma_min)).
        # Going through t_of_tau(sigma(t)**2) instead would round-trip through
        # log/exp and lose the r == 1 identity in float32.
        t_rho = t - 0.5 * jnp.log(q.D) / sch.log_ratio
        s = (a - kappa * z) / q.D + backbone.score_theta_t(q.m, t_rho, x_o) / q.D
        aux = {
            "rho": q.rho,
            "t_rho": t_rho,
            "t": t,
            "m_norm": _norm_mean(q.m),
            "z_norm": _norm_mean(z),
            "displacement": _norm_mean(q.m - z),
            "t_rho_below_min": (t_rho < t_min).astype(jnp.float32),
            "t_rho_above_max": (t_rho > t_max).astype(jnp.float32),
            "score_norm": _norm_mean(s),
        }
        return s, aux

    return hook, {"base_score": 1}, {
        "method": "tq", "a": np.asarray(a).tolist(), "kappa": kappa,
        "guard_min_D": guard_min_D,
    }


# ------------------------------------------------------------ PriorGuide ---


def make_pg_hook(backbone, x_o, comps, *, reverse_cov: jnp.ndarray,
                 clip_x0: Optional[float] = UPSTREAM_X0_CLIP):
    """PriorGuide's reverse-Gaussian closure, ``g = J_m^T u``.

    ``reverse_cov`` is ``Sigma_post``.  Upstream builds it from an empirical
    scalar variance of base Simformer samples
    (``inv(inv(total_var I) + I / tau)``); the analytic training covariance is
    also accepted.  The raw (unsymmetrised) learned Jacobian is used, because
    that is what upstream's ``jacfwd`` returns.
    """
    sch = backbone.schedule
    closure = jax.vmap(
        lambda m, jac, cov: jac.T @ methods.gaussian_product_moments(m, cov, comps)[1]
    , in_axes=(0, 0, None))

    def hook(z, t):
        tau = sch.sigma(t) ** 2
        s = backbone.score_theta_t(z, t, x_o)
        jac = backbone.denoiser_jacobian_t(z, t, x_o)
        m = z + tau * s
        cov = reverse_cov(tau) if callable(reverse_cov) else reverse_cov
        g = closure(m, jac, cov)
        if clip_x0 is not None:
            x0_new = jnp.clip(m + tau * g, -clip_x0, clip_x0)
            hit = jnp.mean((jnp.abs(m + tau * g) > clip_x0).astype(jnp.float32))
            s_q = (x0_new - z) / tau
        else:
            hit = jnp.zeros(())
            s_q = s + g
        asym = jnp.linalg.norm(jac - jnp.swapaxes(jac, -1, -2), axis=(-2, -1))
        return s_q, {
            "guidance_norm": _norm_mean(g),
            "score_norm": _norm_mean(s_q),
            "jac_asym": jnp.mean(asym / jnp.linalg.norm(jac, axis=(-2, -1))),
            "clip_hit_frac": hit,
        }

    return hook, {"base_score": 1, "denoiser_jacobian": 1}, {
        "method": "pg", "clip_x0": clip_x0,
    }


def make_pg_vjp_hook(backbone, x_o, comps, *, reverse_cov: jnp.ndarray,
                     clip_x0: Optional[float] = UPSTREAM_X0_CLIP):
    """PriorGuide's ``g = J_m^T u`` as one vector-Jacobian product.

    Same algebra as :func:`make_pg_hook` (the same ``u`` from the Gaussian
    product, reverse covariance and x0 clip), but the Jacobian is not formed.
    With ``m = z + tau s_p``, ``J_m^T u = u + tau J_s^T u``, and ``jax.vjp`` of the
    batched score gives ``J_s^T u`` in one backward pass instead of the
    ``theta_dim`` passes of ``jacrev``.  ``u`` is evaluated at the primal ``m`` and
    handed to the pullback as a cotangent, so no ``du/dm`` term enters (taking
    ``grad`` of ``u(m)^T m`` would add one).  The score network acts row by row,
    so the batched pullback is the per-row ``J_s(z_i)^T u_i``.  There is no
    ``jac_asym`` diagnostic because it would need the Jacobian.
    """
    sch = backbone.schedule
    weighted_residual = jax.vmap(
        lambda m, cov: methods.gaussian_product_moments(m, cov, comps)[1], in_axes=(0, None)
    )

    def hook(z, t):
        tau = sch.sigma(t) ** 2
        s, pullback = jax.vjp(lambda zz: backbone.score_theta_t(zz, t, x_o), z)
        m = z + tau * s
        cov = reverse_cov(tau) if callable(reverse_cov) else reverse_cov
        u = weighted_residual(m, cov)
        g = u + tau * pullback(u)[0]
        if clip_x0 is not None:
            x0_new = jnp.clip(m + tau * g, -clip_x0, clip_x0)
            hit = jnp.mean((jnp.abs(m + tau * g) > clip_x0).astype(jnp.float32))
            s_q = (x0_new - z) / tau
        else:
            hit = jnp.zeros(())
            s_q = s + g
        return s_q, {
            "guidance_norm": _norm_mean(g),
            "score_norm": _norm_mean(s_q),
            "clip_hit_frac": hit,
        }

    return hook, {"base_score": 1, "score_vjp": 1}, {
        "method": "pg_vjp", "clip_x0": clip_x0,
    }


def upstream_reverse_cov(total_variance: float, dim: int):
    """Upstream's ``Sigma_post = inv(inv(total_var I) + I / tau)``.

    ``total_variance`` is ``jnp.var(basic_diffusion_samples)``, a single scalar
    pooled over all base Simformer samples and all theta coordinates, as
    upstream's ``run_prior_guide_*.py`` computes it.
    """
    eye = jnp.eye(dim)

    def cov(tau):
        return jnp.linalg.inv(eye / total_variance + eye / tau)

    return cov


# ----------------------------------------------------------- PG-FullCov ----


def make_a_full_hook(backbone, x_o, comps, *, psd_eps: float = PSD_EPS):
    """Moment-consistent twisted mean with the Tweedie full covariance ``C_T = tau J``.

    The learned Jacobian is symmetrised and PSD-projected with a fixed
    eigenvalue floor; the raw asymmetry, the non-PSD rate and the projection size
    are reported every step.
    """
    sch = backbone.schedule
    twist = jax.vmap(
        lambda m, cov: methods.gaussian_product_moments(m, cov, comps)[2],
        in_axes=(0, 0),
    )

    def hook(z, t):
        tau = sch.sigma(t) ** 2
        s = backbone.score_theta_t(z, t, x_o)
        jac = backbone.denoiser_jacobian_t(z, t, x_o)
        m = z + tau * s
        j_sym, j_psd, w, w_clip = psd_project(jac, psd_eps)
        m_q = twist(m, tau * j_psd)
        g = (m_q - m) / tau
        asym = jnp.linalg.norm(jac - jnp.swapaxes(jac, -1, -2), axis=(-2, -1))
        return s + g, {
            "guidance_norm": _norm_mean(g),
            "score_norm": _norm_mean(s + g),
            "jac_asym": jnp.mean(asym / jnp.linalg.norm(jac, axis=(-2, -1))),
            "min_eig_sym": jnp.mean(jnp.min(w, axis=-1)),
            "frac_points_nonpsd": jnp.mean((jnp.min(w, axis=-1) <= 0.0).astype(jnp.float32)),
            "frac_eigs_clipped": jnp.mean((w < psd_eps).astype(jnp.float32)),
        }

    return hook, {"base_score": 1, "denoiser_jacobian": 1, "eigh": 1}, {
        "method": "a_full", "psd_eps": psd_eps,
    }


# --------------------------------------------------------------- Spectra ----


def make_spectra_hook(backbone, x_o, atoms, labels):
    """Exact per-atom transport with a trajectory-fixed atom label.

    ``labels`` assigns one atom to each walker for the whole run, drawn once from
    ``Categorical(pihat)``.  Different walkers therefore query different
    transformed states and different transformed noise levels, but the network
    still sees one batched forward per step: ``score_theta_t`` broadcasts a
    per-row time vector, so the per-atom ``t_rho`` needs no per-atom forward.

    The hook does no online field estimation; the only estimated quantities are
    the atom weights, which are fixed before sampling.
    """
    a_all = jnp.asarray(atoms.a)  # (K, d)
    kap_all = jnp.asarray(atoms.kappa)  # (K,)
    labels = jnp.asarray(labels)
    a = a_all[labels]  # (B, d)
    kap = kap_all[labels]  # (B,)
    sch = backbone.schedule
    t_min, t_max = backbone.nominal_t_range()
    guard_min_D = require_guard(sch, (t_min, t_max), np.asarray(atoms.kappa, float),
                                "exact per-atom transport")

    def hook(z, t):
        tau = sch.sigma(t) ** 2
        D = 1.0 + kap * tau  # (B,)
        m = (z + tau * a) / D[:, None]
        # same exact time shift as make_tq_hook, per walker
        t_rho = t - 0.5 * jnp.log(D) / sch.log_ratio
        s = (a - kap[:, None] * z) / D[:, None] + backbone.score_theta_t(
            m, t_rho, x_o) / D[:, None]
        return s, {
            "rho_median": jnp.median(tau / D),
            "t_rho_median": jnp.median(t_rho),
            "t_rho_below_min": jnp.mean((t_rho < t_min).astype(jnp.float32)),
            "t_rho_above_max": jnp.mean((t_rho > t_max).astype(jnp.float32)),
            "m_norm": _norm_mean(m),
            "z_norm": _norm_mean(z),
            "displacement": _norm_mean(m - z),
            "score_norm": _norm_mean(s),
            "frac_finite": jnp.mean(jnp.isfinite(s).all(axis=-1).astype(jnp.float32)),
        }

    return hook, {"base_score": 1}, {
        "method": "eamt",
        "num_atoms": int(a_all.shape[0]),
        "kappa": np.asarray(atoms.kappa).tolist(),
        "guard_min_D": guard_min_D,
    }


def draw_atom_labels(key, log_pi, num_samples: int):
    """One trajectory-fixed atom per walker, from its own PRNG key.

    Kept out of the sampler's key stream so that the terminal draw and the
    Brownian/Langevin increments stay element-wise paired with every other
    method in the cell.
    """
    return jax.random.categorical(key, jnp.asarray(log_pi), shape=(num_samples,))
