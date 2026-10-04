"""Exact representations of the test-time prior ratio ``r = pi_new / pi_tr``.

Every upstream benchmark prior is isotropic, so the ratio is available in closed
form and nothing here is fitted.  Two representations are produced:

``GaussianComponents``
    ``r(theta) ∝ sum_k w_k N(theta; mu_k, Sigma_k)`` with positive weights.  This
    is what PriorGuide's closure and PG-FullCov consume.  For a Gaussian training
    prior upstream instead runs ``fit_gmm(..., num_components=20)`` on
    ``log q - log p``; here the exact components are used for the PriorGuide
    (``pg_exact``) and PG-FullCov rows.

``ExpQuadRatio``
    ``r(theta) ∝ exp(a^T theta - kappa/2 ||theta||^2)``, the isotropic
    exponential--quadratic family that exact structured transport (TQ) solves.
    Only the single-component (mild / strong) shifts land in this family; the
    two-component mixture priors do not, and are left to PG / PG-FullCov.

Both representations are checked against ``log pi_new - log pi_tr`` up to a
global constant in ``tests/test_prior_shift.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from spectra.benchmark import load_prior_json
from spectra.ratios import GaussianComponents


@dataclass(frozen=True)
class ExpQuadRatio:
    """``log r(theta) = a^T theta - kappa/2 ||theta||^2 + const``.

    ``a`` is kept in float64 so the exactness tests are not limited by JAX's
    default float32; :attr:`a_jnp` is the view the sampler queries.
    """

    a: np.ndarray  # (d,) float64
    kappa: float

    @property
    def a_jnp(self) -> jnp.ndarray:
        return jnp.asarray(self.a)

    def log_r_unnorm(self, theta) -> np.ndarray:
        """float64 evaluation straight from the stored parameters."""
        theta = np.atleast_2d(np.asarray(theta, float))
        return theta @ self.a - 0.5 * self.kappa * np.sum(theta**2, axis=-1)

    def guard(self, tau: float) -> bool:
        """The transport family requires ``1 + kappa tau > 0`` at every noise level."""
        return bool(1.0 + self.kappa * tau > 0.0)


@dataclass(frozen=True)
class IsotropicGaussianPrior:
    """``N(mu, sigma^2 I)`` (a single component of a test prior)."""

    mu: np.ndarray
    sigma: np.ndarray  # per-dim; upstream always writes these isotropic

    @property
    def var(self) -> np.ndarray:
        return self.sigma**2

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        z = (theta - self.mu) / self.sigma
        return -0.5 * np.sum(z**2, axis=-1) - np.sum(np.log(self.sigma)) - 0.5 * len(
            self.mu
        ) * math.log(2 * math.pi)


@dataclass(frozen=True)
class PriorShift:
    """One (task, prior_type, prior_id) test-time shift with all exact objects."""

    task: str
    prior_type: str
    prior_id: int
    theta_dim: int
    train_kind: str  # "gaussian" | "uniform"
    comp_log_weights: np.ndarray  # (K,) float64, exact
    comp_means: np.ndarray  # (K, d) float64, exact
    comp_covs: np.ndarray  # (K, d, d) float64, exact
    expquad: Optional[ExpQuadRatio]
    train_prior_cov: Optional[jnp.ndarray]  # PG's Sigma_train branch, Gaussian tasks
    target_mu: np.ndarray  # (K, d)
    target_sigma: np.ndarray  # (K, d)
    target_pi: np.ndarray  # (K,)
    train_mu: Optional[np.ndarray]
    train_sigma: Optional[np.ndarray]
    train_box: Optional[tuple]

    @property
    def is_mixture(self) -> bool:
        return self.target_pi.shape[0] > 1

    @property
    def components(self) -> GaussianComponents:
        """The jnp view consumed by PriorGuide's closure and by PG-FullCov."""
        return GaussianComponents(
            log_weights=jnp.asarray(self.comp_log_weights),
            means=jnp.asarray(self.comp_means),
            covs=jnp.asarray(self.comp_covs),
        )

    # ---- exact log ratio, used only by tests and diagnostics -----------------

    def log_r(self, theta: np.ndarray) -> np.ndarray:
        """``log pi_new(theta) - log pi_tr(theta)`` up to a constant.

        For a uniform training prior the (constant) training log-density is
        dropped, which is the ``+ const`` freedom the guidance uses.
        Outside a uniform training box the value returned is still the analytic
        continuation of ``log pi_new``; support handling is the caller's job.
        """
        theta = np.atleast_2d(np.asarray(theta, float))
        logs = np.stack(
            [
                IsotropicGaussianPrior(self.target_mu[k], self.target_sigma[k]).log_prob(
                    theta
                )
                + math.log(self.target_pi[k])
                for k in range(self.target_pi.shape[0])
            ],
            axis=0,
        )
        out = _logsumexp(logs, axis=0)
        if self.train_kind == "gaussian":
            out = out - IsotropicGaussianPrior(self.train_mu, self.train_sigma).log_prob(
                theta
            )
        return out

    def grad_log_r(self, theta: np.ndarray) -> np.ndarray:
        """``grad_theta log r(theta)`` in float64: the exact clean-time boundary.

        Differentiating the exact Gaussian expansion rather than ``log_r``
        itself is legitimate because the two differ by a global constant, and it
        keeps a single code path for the Gaussian and the uniform training prior.
        For a uniform training prior this is the analytic continuation of
        ``grad log pi_new`` outside the box as well; the caller records how much
        mass ends up there (:meth:`target_mass_outside_training_box`) rather than
        clamping the field, which would make it discontinuous.

        The jnp counterpart is
        :meth:`spectra.ratios.GaussianComponents.grad_log_r`.
        """
        theta = np.atleast_2d(np.asarray(theta, float))
        lw, mus, covs = self.comp_log_weights, self.comp_means, self.comp_covs
        prec = np.linalg.inv(covs)  # (K, d, d)
        _, logdet = np.linalg.slogdet(covs)  # (K,)
        diff = theta[:, None, :] - mus[None, :, :]  # (n, K, d)
        quad = np.einsum("bkd,kde,bke->bk", diff, prec, diff)
        log_c = lw[None, :] - 0.5 * quad - 0.5 * logdet[None, :]
        rho = np.exp(log_c - _logsumexp(log_c, axis=1)[:, None])  # (n, K)
        grads = -np.einsum("kde,bke->bkd", prec, diff)  # (n, K, d)
        return np.einsum("bk,bkd->bd", rho, grads)

    def log_components(self, theta: np.ndarray) -> np.ndarray:
        """``log sum_k w_k N(theta; mu_k, Sigma_k)`` of the exact expansion."""
        theta = np.atleast_2d(np.asarray(theta, float))
        lw, mus, covs = self.comp_log_weights, self.comp_means, self.comp_covs
        logs = []
        for k in range(lw.shape[0]):
            sd = np.sqrt(np.diag(covs[k]))
            logs.append(
                lw[k]
                + IsotropicGaussianPrior(mus[k], sd).log_prob(theta)
            )
        return _logsumexp(np.stack(logs, axis=0), axis=0)

    # ------------------------------------------------------ support caveat --

    def target_mass_outside_training_box(self) -> Optional[float]:
        """``P_{pi_new}(theta outside supp(pi_tr))``, analytic for the box case."""
        if self.train_box is None:
            return None
        low, high = self.train_box
        total = 0.0
        for k in range(self.target_pi.shape[0]):
            mu, sd = self.target_mu[k], self.target_sigma[k]
            inside = np.prod(norm.cdf((high - mu) / sd) - norm.cdf((low - mu) / sd))
            total += self.target_pi[k] * (1.0 - inside)
        return float(total)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def _target_components(js: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mu, sigma, pi)`` with a leading component axis for both prior kinds."""
    mu = np.asarray(js["mu"], float)
    sigma = np.asarray(js["sigma"], float)
    if js["dist"] == "mixture":
        return mu, sigma, np.asarray(js["pi"], float)
    return mu[None, :], sigma[None, :], np.ones(1)


def build_prior_shift(task: str, prior_type: str, prior_id: int, root=None) -> PriorShift:
    """Assemble every exact object for one shipped prior JSON."""
    train = load_prior_json(task, "training", root=root)
    target = load_prior_json(task, prior_type, prior_id, root=root)
    mu_q, sd_q, pi_q = _target_components(target)
    d = mu_q.shape[-1]

    if train["dist"] == "gaussian":
        mu_p = np.asarray(train["mu"], float)
        sd_p = np.asarray(train["sigma"], float)
        box = None
        _require_isotropic(sd_p, f"{task} training prior")
        comps, expquad = _gaussian_training_ratio(mu_q, sd_q, pi_q, mu_p, sd_p)
        train_cov = jnp.asarray(np.diag(sd_p**2))
    elif train["dist"] == "uniform":
        mu_p = sd_p = None
        box = (np.asarray(train["low"], float), np.asarray(train["high"], float))
        comps, expquad = _uniform_training_ratio(mu_q, sd_q, pi_q)
        train_cov = None
    else:
        raise ValueError(f"unsupported training prior dist {train['dist']!r}")

    log_w, means, covs = comps
    return PriorShift(
        task=task, prior_type=prior_type, prior_id=prior_id, theta_dim=d,
        train_kind=train["dist"], comp_log_weights=log_w, comp_means=means,
        comp_covs=covs, expquad=expquad, train_prior_cov=train_cov,
        target_mu=mu_q, target_sigma=sd_q, target_pi=pi_q,
        train_mu=mu_p, train_sigma=sd_p, train_box=box,
    )


def _require_isotropic(sigma: np.ndarray, what: str) -> None:
    if not np.allclose(sigma, sigma[0]):
        raise ValueError(
            f"{what} is not isotropic ({sigma}); the exact family and the "
            "VE kernel are only compatible in an isotropic parameterisation"
        )


def _uniform_training_ratio(mu_q, sd_q, pi_q):
    """Uniform training prior: ``r ∝ pi_new`` on the box, already a Gaussian mixture.

    This is what upstream's ``run_prior_guide_uniform.py`` passes to the
    guidance; nothing is fitted.
    """
    # Every component is checked, not just the single-component case: the
    # mixture components become Spectra's transport atoms, whose exact
    # transformed-query form is isotropic-only (spectra.atoms).
    for k in range(pi_q.shape[0]):
        _require_isotropic(sd_q[k], f"target prior component {k}")
    covs = np.stack([np.diag(sd_q[k] ** 2) for k in range(pi_q.shape[0])])
    comps = (np.log(pi_q), np.asarray(mu_q, float), covs)
    expquad = None
    if pi_q.shape[0] == 1:
        lam = float(sd_q[0, 0] ** 2)
        expquad = ExpQuadRatio(a=np.asarray(mu_q[0] / lam, float), kappa=1.0 / lam)
    return comps, expquad


def _gaussian_training_ratio(mu_q, sd_q, pi_q, mu_p, sd_p):
    """Gaussian training prior: each ratio term is an exponential-quadratic.

    For component ``k``,

        N(theta; mu_k, s_k^2 I) / N(theta; mu_p, s_p^2 I)
            = C_k exp(a_k^T theta - kappa_k/2 ||theta||^2)

    and, whenever ``kappa_k > 0`` (always true here because the test priors are
    narrower than the training prior), that is a positive multiple of
    ``N(theta; a_k / kappa_k, I / kappa_k)``.  The resulting component weights are
    exact, so the expansion reproduces ``log r`` up to one global constant.
    """
    K, d = mu_q.shape
    var_p = sd_p**2
    log_w, means, covs = [], [], []
    expquad = None
    for k in range(K):
        _require_isotropic(sd_q[k], f"target prior component {k}")
        var_k = sd_q[k] ** 2
        a_k = mu_q[k] / var_k - mu_p / var_p
        kappa_k = float(1.0 / var_k[0] - 1.0 / var_p[0])
        if kappa_k <= 0:
            raise ValueError(
                f"component {k} has kappa={kappa_k:.6g} <= 0: the ratio is not a "
                "normalisable Gaussian, use the exponential-tilt path instead"
            )
        # log of the multiplicative constant in front of N(theta; a/kappa, I/kappa)
        log_c = (
            float(np.sum(np.log(sd_p) - np.log(sd_q[k])))
            + 0.5 * d * math.log(2 * math.pi / kappa_k)
            + float(np.sum(a_k**2) / (2 * kappa_k))
            - float(np.sum(mu_q[k] ** 2 / (2 * var_k)))
            + float(np.sum(mu_p**2 / (2 * var_p)))
        )
        log_w.append(math.log(pi_q[k]) + log_c)
        means.append(a_k / kappa_k)
        covs.append(np.eye(d) / kappa_k)
        if K == 1:
            expquad = ExpQuadRatio(a=np.asarray(a_k, float), kappa=kappa_k)
    comps = (np.asarray(log_w), np.stack(means), np.stack(covs))
    return comps, expquad
