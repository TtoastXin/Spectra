"""Out-of-native prior compilation: a correlated Gaussian as positive atoms.

A correlated Gaussian localisation is not in the exact-atom family of
isotropic Gaussian mixtures.  It is compiled into that family by an identity,
not a fit.  For any ``0 < lambda < lambda_min(Sigma)``,

    N(y; mu, Sigma) = E_{U ~ N(mu, Sigma_c)}[ N(y; U, lambda I) ],
    Sigma_c = Sigma - lambda I  (PSD by the choice of lambda),

so an anisotropic Gaussian is an exact continuous mixture of isotropic
common-bandwidth atoms.  Discretising that expectation with a tensor
Gauss--Hermite rule keeps every weight strictly positive, as the exact-atom
theorem requires, and turns the identity into a finite dictionary

    r~(y) = sum_k b_k N(y; u_k, lambda I),   b_k > 0,   u_k = mu + L z_k.

Everything that fixes the dictionary depends only on the prior and is
deterministic: the covariance, the ratio ``c = lambda / lambda_min(Sigma)``, the
quadrature order and the resulting centres.  No observation, posterior,
evidence or C2ST enters, so one compilation is reused across observations.

Two capacities are set by one prior-only criterion (how many atom widths each
principal axis of ``Sigma_c`` spans), giving ``K = 27`` and ``K = 64`` here.
They are not tuned.  The one-order-for-all-axes tensor rule is available as a
baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelatedGaussianPrior:
    """The prior-only target of the compilation, in model coordinates."""

    mu: np.ndarray  # (d,)
    cov: np.ndarray  # (d, d), correlated
    box: tuple  # training support, for the mass check
    rule: str  # how mu/cov were fixed

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        y = np.atleast_2d(np.asarray(y, float))
        d = self.dim
        chol = np.linalg.cholesky(self.cov)
        sol = np.linalg.solve(chol, (y - self.mu).T)
        _, logdet = np.linalg.slogdet(self.cov)
        return -0.5 * np.sum(sol**2, axis=0) - 0.5 * logdet - 0.5 * d * math.log(
            2 * math.pi)

    def mass_inside_box(self, num: int = 200_000, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(self.cov)
        y = self.mu + rng.standard_normal((num, self.dim)) @ chol.T
        lo, hi = self.box
        return float(np.mean(np.all((y >= lo) & (y <= hi), axis=-1)))

    def as_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "cov": self.cov.tolist(),
                "eigenvalues": np.linalg.eigvalsh(self.cov).tolist(),
                "condition_number": float(
                    np.linalg.cond(self.cov)), "rule": self.rule}


def design_prior(box, *, offset_frac: float = 0.3, major_sd: float = 0.2,
                 condition_number: float = 6.0, rotation_deg: float = 45.0,
                 min_mass_inside: float = 0.99, shrink: float = 0.8,
                 max_shrinks: int = 4) -> CorrelatedGaussianPrior:
    """The fixed, observation-independent construction rule.

    ``mu`` sits at a fixed off-centre interior point (so the prior is not
    trivially symmetric about the box), the covariance is rotated by a fixed
    angle with a fixed condition number, and the only adaptive element is a
    fixed shrink applied if more than ``1 - min_mass_inside`` of the prior mass
    falls outside the training box.  The shrink depends only on the prior and
    the box, and its use is recorded in ``rule``.
    """
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    d = lo.shape[0]
    if d != 2:
        raise ValueError("design_prior is defined for two-dimensional parameters only")
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    mu = center + offset_frac * half * np.array([1.0, -1.0])
    th = math.radians(rotation_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    shrinks = 0
    sd_major = major_sd
    while True:
        sd_minor = sd_major / math.sqrt(condition_number)
        cov = rot @ np.diag([sd_major**2, sd_minor**2]) @ rot.T
        prior = CorrelatedGaussianPrior(
            mu=mu, cov=cov, box=(lo, hi),
            rule=(f"mu = box_center + {offset_frac} * half_width * (+1, -1); "
                  f"cov = R({rotation_deg} deg) diag(sd^2, sd^2 / "
                  f"{condition_number}) R^T with sd = {sd_major:.6f}"
                  + (f"; shrunk {shrinks}x by {shrink}" if shrinks else "")))
        if prior.mass_inside_box() >= min_mass_inside or shrinks >= max_shrinks:
            return prior
        sd_major *= shrink
        shrinks += 1


@dataclass(frozen=True)
class AtomDictionary:
    centers: np.ndarray  # (K, d)
    log_b: np.ndarray  # (K,) normalised log weights
    lam: float
    order: int
    c_ratio: float
    prior: CorrelatedGaussianPrior

    @property
    def num_atoms(self) -> int:
        return int(self.centers.shape[0])

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        y = np.atleast_2d(np.asarray(y, float))
        d = y.shape[-1]
        parts = []
        for k in range(self.num_atoms):
            r2 = np.sum((y - self.centers[k]) ** 2, axis=-1)
            parts.append(self.log_b[k] - 0.5 * r2 / self.lam
                         - 0.5 * d * math.log(2 * math.pi * self.lam))
        parts = np.stack(parts, 0)
        m = parts.max(0)
        return m + np.log(np.exp(parts - m).sum(0))

    def as_prior_json(self, task: str) -> dict:
        """The dictionary written as an ordinary isotropic mixture prior.

        In this form ``build_prior_shift``, ``atoms_from_shift``, the Spectra
        hook, PriorGuide's closure and PG-FullCov consume it without any
        special case, and the ``K > 2`` bookkeeping is exercised as well.
        """
        sd = math.sqrt(self.lam)
        return {
            "task": task, "dist": "mixture", "type": "mixture",
            "mu": self.centers.tolist(),
            "sigma": [[sd] * self.centers.shape[1]] * self.num_atoms,
            "pi": np.exp(self.log_b).tolist(),
            "compiled_from": self.prior.as_dict(),
            "compilation": {"lambda": self.lam, "gauss_hermite_max_order": self.order,
                            "c_ratio_lambda_over_lambda_min": self.c_ratio,
                            "num_atoms": self.num_atoms},
        }

    def as_dict(self) -> dict:
        return {"num_atoms": self.num_atoms, "lambda": self.lam,
                "sqrt_lambda": math.sqrt(self.lam), "order": self.order,
                "c_ratio": self.c_ratio,
                "centers": self.centers.tolist(),
                "log_b": self.log_b.tolist(),
                "prior": self.prior.as_dict()}


# Capacities.  ``spread_factor`` is the only capacity knob: the number of
# quadrature nodes along each principal axis of ``Sigma_c`` is set by how many
# atom widths that axis spans, i.e. the mixing density is resolved at the
# kernel scale.  Chosen from the prior alone and not tuned.
CAPACITIES = {"small": 3.0, "medium": 6.0}
C_RATIO = 0.9  # lambda = C_RATIO * lambda_min(Sigma); keeps Sigma_c PSD with margin


def compile_dictionary(prior: CorrelatedGaussianPrior, *, spread_factor: float,
                       c_ratio: float = C_RATIO, max_nodes_per_axis: int = 31,
                       isotropic_tensor_order: int | None = None
                       ) -> AtomDictionary:
    """Gauss--Hermite discretisation of the exact mixture identity.

    Gauss--Hermite weights are strictly positive, so the dictionary lands inside
    the positive-atom family by construction rather than by a check afterwards.

    The order differs per axis.  ``Sigma_c`` here has principal standard
    deviations of about ``0.19`` and ``0.03`` while the atom width is ``0.078``,
    so a tensor rule with one order for both axes either wastes nodes on the
    short axis or leaves the long axis unresolved: the ``K = 9`` and ``K = 36``
    isotropic tensor rules give TV bounds of 0.47 and 0.19.  Allocating nodes in
    proportion to ``sd_i / sqrt(lambda)`` gives ``K = 27`` and ``K = 64`` with
    bounds of 0.078 and 0.009.  Both the criterion and the numbers depend only
    on the prior.

    ``isotropic_tensor_order`` gives the one-order-for-all-axes rule instead.
    """
    lam_min = float(np.linalg.eigvalsh(prior.cov).min())
    lam = c_ratio * lam_min
    cov_c = prior.cov - lam * np.eye(prior.dim)
    ev, axes = np.linalg.eigh(cov_c)
    if ev.min() < -1e-12:
        raise ValueError(f"Sigma_c is not PSD (min eig {ev.min():.3g}); "
                         "lambda must be below lambda_min(Sigma)")
    sd = np.sqrt(np.maximum(ev, 0.0))

    if isotropic_tensor_order is not None:
        orders = [int(isotropic_tensor_order)] * prior.dim
    else:
        orders = [int(min(max_nodes_per_axis,
                          max(1, math.ceil(spread_factor * s / math.sqrt(lam)) + 1)))
                  for s in sd]

    nodes, weights = [], []
    for i in range(prior.dim):
        xi, wi = np.polynomial.hermite_e.hermegauss(orders[i])
        nodes.append(xi * sd[i])
        weights.append(wi / wi.sum())
    grids = np.meshgrid(*nodes, indexing="ij")
    z = np.stack([g.reshape(-1) for g in grids], axis=-1)
    wt = np.ones(z.shape[0])
    idx = np.indices(orders)
    for i in range(prior.dim):
        wt = wt * weights[i][idx[i].reshape(-1)]
    centers = prior.mu + z @ axes.T
    log_b = np.log(wt / wt.sum())
    return AtomDictionary(centers=centers, log_b=log_b, lam=lam,
                          order=int(max(orders)), c_ratio=c_ratio, prior=prior)


def approximation_error(dict_: AtomDictionary, *, num: int = 200_000,
                        seed: int = 0) -> dict:
    """Prior-only accuracy of the compiled ratio.

    ``eps_r = E_{pi_train}|r - r~|`` is the quantity that bounds
    ``TV(q, q~) <= eps_r / Z`` for any observation, so it is measured under the
    training prior, not under a posterior; a posterior-weighted number would
    make the dictionary depend on the observation.
    """
    rng = np.random.default_rng(seed)
    lo, hi = dict_.prior.box
    y = rng.uniform(lo, hi, size=(num, dict_.prior.dim))
    lr = dict_.prior.log_prob(y)
    lrt = dict_.log_prob(y)
    r, rt = np.exp(lr), np.exp(lrt)
    vol = float(np.prod(np.asarray(hi, float) - np.asarray(lo, float)))
    eps_abs = float(np.mean(np.abs(r - rt)))  # E_uniform |r - r~|
    mass = float(np.mean(r))
    # relative error where the ratio carries mass
    heavy = r > np.quantile(r, 0.9)
    return {
        "num_points": int(num),
        "E_uniform_abs_diff": eps_abs,
        "E_uniform_r": mass,
        "eps_r_over_Z_bound_uniform": eps_abs / mass if mass > 0 else float("inf"),
        "max_abs_log_ratio_gap_heavy_decile": float(
            np.max(np.abs(lr[heavy] - lrt[heavy]))),
        "median_abs_log_ratio_gap_heavy_decile": float(
            np.median(np.abs(lr[heavy] - lrt[heavy]))),
        "prior_mass_inside_box": dict_.prior.mass_inside_box(seed=seed + 1),
        "note": ("TV(q, q~) <= E_{pi_train}|r - r~| / Z for every observation; "
                 "measured under the training prior only"),
    }
