"""Analytic VE oracles built on Gaussian mixtures.

A Gaussian mixture base posterior is closed under VE noising, under an
exponential tilt and under multiplication by a Gaussian factor, so a single
mixture class covers every oracle configuration (single Gaussian == K=1) and
delivers ``p_tau``, ``q_tau``, their scores and Hessians to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GaussianMixture:
    """Mixture ``sum_j w_j N(theta; means_j, covs_j)`` of clean densities.

    ``log_weights`` need not be normalised; every quantity used downstream is
    invariant to a global additive constant.
    """

    log_weights: jnp.ndarray  # (K,)
    means: jnp.ndarray  # (K, d)
    covs: jnp.ndarray  # (K, d, d)

    @property
    def dim(self) -> int:
        return int(self.means.shape[-1])

    @property
    def num_components(self) -> int:
        return int(self.means.shape[0])

    # -- noised density -------------------------------------------------
    def log_prob_tau(self, z: jnp.ndarray, tau: float) -> jnp.ndarray:
        """log p_tau(z) up to an additive constant, with p_tau = G_tau * p_0."""
        cov = self.covs + tau * jnp.eye(self.dim)  # (K, d, d)
        resid = z[None, :] - self.means  # (K, d)
        chol = jnp.linalg.cholesky(cov)
        sol = jax.scipy.linalg.solve_triangular(chol, resid[..., None], lower=True)
        quad = jnp.sum(sol[..., 0] ** 2, axis=-1)  # (K,)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
        log_comp = -0.5 * quad - 0.5 * logdet
        return jax.scipy.special.logsumexp(self.log_weights + log_comp)

    def score(self, z: jnp.ndarray, tau: float) -> jnp.ndarray:
        return jax.grad(self.log_prob_tau)(z, tau)

    def hess_log_prob(self, z: jnp.ndarray, tau: float) -> jnp.ndarray:
        return jax.hessian(self.log_prob_tau)(z, tau)

    # -- Tweedie quantities ---------------------------------------------
    def denoiser(self, z: jnp.ndarray, tau: float) -> jnp.ndarray:
        return z + tau * self.score(z, tau)

    def denoiser_jacobian(self, z: jnp.ndarray, tau: float) -> jnp.ndarray:
        """J_m = dm/dy = I + tau * grad^2 log p_tau."""
        return jnp.eye(self.dim) + tau * self.hess_log_prob(z, tau)

    def tweedie_cov(self, z: jnp.ndarray, tau: float) -> jnp.ndarray:
        """C_T = Cov_p[theta_0 | z] = tau * J_m."""
        return tau * self.denoiser_jacobian(z, tau)

    # -- sampling --------------------------------------------------------
    def sample_tau(self, key: jax.Array, tau: float, num: int) -> jnp.ndarray:
        key_c, key_z = jax.random.split(key)
        idx = jax.random.categorical(key_c, self.log_weights, shape=(num,))
        cov = self.covs[idx] + tau * jnp.eye(self.dim)
        chol = jnp.linalg.cholesky(cov)
        eps = jax.random.normal(key_z, (num, self.dim))
        return self.means[idx] + jnp.einsum("nij,nj->ni", chol, eps)


def tilt_mixture(mix: GaussianMixture, b: jnp.ndarray, mat: jnp.ndarray) -> GaussianMixture:
    """Multiply each component by ``exp(b^T z - 0.5 z^T mat z)`` and renormalise.

    Covers the ratio families used with these oracles: exponential tilt
    (``mat = 0``), a Gaussian factor and a Gaussian covariance shift.  Global constants are dropped.
    """
    dim = mix.dim
    prec = jnp.linalg.inv(mix.covs) + mat[None]  # (K, d, d)
    new_covs = jnp.linalg.inv(prec)
    nat = jnp.einsum("kij,kj->ki", jnp.linalg.inv(mix.covs), mix.means) + b[None]
    new_means = jnp.einsum("kij,kj->ki", new_covs, nat)

    _, logdet_new = jnp.linalg.slogdet(new_covs)
    _, logdet_old = jnp.linalg.slogdet(mix.covs)
    quad_new = jnp.einsum("ki,kij,kj->k", new_means, prec, new_means)
    quad_old = jnp.einsum("ki,kij,kj->k", mix.means, jnp.linalg.inv(mix.covs), mix.means)
    offset = 0.5 * (logdet_new - logdet_old) + 0.5 * (quad_new - quad_old)

    log_w = mix.log_weights + offset
    log_w = log_w - jax.scipy.special.logsumexp(log_w)
    return GaussianMixture(log_weights=log_w, means=new_means, covs=new_covs)


def single_gaussian(mean: jnp.ndarray, cov: jnp.ndarray) -> GaussianMixture:
    return GaussianMixture(
        log_weights=jnp.zeros((1,)),
        means=mean[None, :],
        covs=cov[None, :, :],
    )


class GaussianOracleBackbone:
    """Analytic stand-in for :class:`spectra.backbone.Backbone`.

    Wraps a :class:`GaussianMixture` clean posterior and exposes the subset of
    the ``Backbone`` API the hooks and the reverse driver consume
    (``schedule``, ``theta_dim``, ``score_theta_t``, ``denoiser_jacobian_t``,
    ``nominal_t_range`` and ``terminal_moments``), with every quantity exact.
    ``x_o`` is accepted and ignored: the mixture is the posterior at the fixed
    observation.  Per-row diffusion times are supported, as the learned backbone
    does for Spectra's per-walker transformed times.

    The Langevin tests use it to check the transported Spectra score, the
    PriorGuide / PG-FullCov guidance and the corrector against closed forms
    without learned-score error.
    """

    def __init__(self, mixture: GaussianMixture, schedule):
        self.mixture = mixture
        self.schedule = schedule
        self.theta_dim = mixture.dim
        self.mode = "conditional"
        self._score = jax.vmap(mixture.score, in_axes=(0, 0))
        self._jac = jax.vmap(mixture.denoiser_jacobian, in_axes=(0, 0))

    def _tau_rows(self, z, t):
        tau = self.schedule.sigma(jnp.asarray(t, dtype=z.dtype)) ** 2
        return jnp.broadcast_to(tau, (z.shape[0],))

    def score_theta_t(self, z, t, x_o=None):
        z = jnp.atleast_2d(jnp.asarray(z))
        return self._score(z, self._tau_rows(z, t))

    def denoiser_jacobian_t(self, z, t, x_o=None):
        z = jnp.atleast_2d(jnp.asarray(z))
        return self._jac(z, self._tau_rows(z, t))

    def nominal_t_range(self):
        return float(self.schedule.t_min), float(self.schedule.t_max)

    def terminal_moments(self):
        """The mixture's own marginal pushed to ``T_max``, as ``training.terminal_moments``.

        Mean of the clean law; per-coordinate standard deviation
        ``sqrt(var_clean + sigma(T_max)^2)``, the same construction
        ``training.terminal_moments`` applies to ``Empirical(data)``.
        """
        import numpy as _np

        w = _np.exp(_np.asarray(self.mixture.log_weights, float))
        w = w / w.sum()
        mu = _np.asarray(self.mixture.means, float)
        cov = _np.asarray(self.mixture.covs, float)
        mean = (w[:, None] * mu).sum(0)
        second = (w[:, None] * (_np.einsum("kii->ki", cov) + mu**2)).sum(0)
        var = second - mean**2
        std = _np.sqrt(var + float(self.schedule.sigma(self.schedule.t_max)) ** 2)
        return mean, std
