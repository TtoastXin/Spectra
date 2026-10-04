"""Guidance estimators for the single-factor comparison.

All estimators return the guidance ``g`` such that ``s_q = s_p + g``; the exact
target is ``g_star = grad log q_tau - grad log p_tau = (m_q - m) / tau``.

The estimators kept here are the ones the tests use as independent references:
  * ``pg_closure``: PriorGuide's reverse-Gaussian closure fed with an exact
    generalized-Gaussian expansion of ``r`` (no ratio fitting error).
  * ``a_full``: PG-FullCov, the twisted-mean closure with the Tweedie full
    covariance ``C_T = tau J_m`` in place of PriorGuide's scalar covariance.
  * ``pg_reverse_cov``: upstream's scalar ``Sigma_post``.
The batched versions used by the sampler live in :mod:`spectra.hooks`.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from spectra.ratios import GaussianComponents


def pg_reverse_cov(tau: float, dim: int, train_prior_cov: jnp.ndarray | None = None) -> jnp.ndarray:
    """PriorGuide's reverse-conditional covariance.

    Default is the isotropic ``C_PG = tau / (1 + tau) * I``, which is what
    upstream uses when ``theta_original_prior_cov`` is not supplied.  With a
    training-prior covariance it becomes ``(Sigma_train^{-1} + I / tau)^{-1}``,
    matching the optional upstream branch.
    """
    if train_prior_cov is None:
        return (tau / (1.0 + tau)) * jnp.eye(dim)
    return jnp.linalg.inv(jnp.linalg.inv(train_prior_cov) + jnp.eye(dim) / tau)


def gaussian_product_moments(
    m: jnp.ndarray, cov: jnp.ndarray, comps: GaussianComponents
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gaussian-product algebra shared by PriorGuide and PG-FullCov.

    With ``Z_k = N(mu_k; m, cov + Sigma_k)`` and ``rho_k ∝ w_k Z_k`` returns
    ``(rho, u, m_q_tilde)`` where

        u          = sum_k rho_k (cov + Sigma_k)^{-1} (mu_k - m)
        m_q_tilde  = sum_k rho_k nu_k = m + cov @ u.
    """
    combined = cov[None] + comps.covs  # (K, d, d)
    resid = comps.means - m[None]  # (K, d)
    sol = jnp.linalg.solve(combined, resid[..., None])[..., 0]  # (K, d)
    quad = jnp.einsum("ki,ki->k", resid, sol)
    _, logdet = jnp.linalg.slogdet(combined)
    log_z = -0.5 * quad - 0.5 * logdet
    log_rho = comps.log_weights + log_z
    rho = _softmax(log_rho)
    u = jnp.einsum("k,ki->i", rho, sol)
    m_q = m + cov @ u
    return rho, u, m_q


def _softmax(log_x: jnp.ndarray) -> jnp.ndarray:
    shifted = log_x - jnp.max(log_x)
    w = jnp.exp(shifted)
    return w / jnp.sum(w)


def pg_closure(m: jnp.ndarray, jac_m: jnp.ndarray, tau: float, comps: GaussianComponents,
               train_prior_cov: jnp.ndarray | None = None) -> jnp.ndarray:
    """g_PG = J_m^T sum_k rho_k (C_PG + Sigma_k)^{-1} (mu_k - m)."""
    cov = pg_reverse_cov(tau, m.shape[-1], train_prior_cov)
    _, u, _ = gaussian_product_moments(m, cov, comps)
    return jac_m.T @ u


def a_full(m: jnp.ndarray, tweedie_cov: jnp.ndarray, tau: float,
           comps: GaussianComponents) -> jnp.ndarray:
    """Twisted-mean closure with the Tweedie full covariance ``C_T = tau J_m``."""
    _, _, m_q = gaussian_product_moments(m, tweedie_cov, comps)
    return (m_q - m) / tau


