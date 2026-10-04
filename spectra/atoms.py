"""Spectra: atom extraction and global atom evidence.

Write the exact expansion of :mod:`spectra.prior_shift` as

    r(theta) ∝ sum_k b_k phi_k(theta),      phi_k = N(theta; nu_k, I / kappa_k),

with every ``b_k > 0``.  The target posterior then factorises exactly:

    q(theta | x) ∝ p_0(theta | x) sum_k b_k phi_k(theta)
                 = sum_k b_k Z_k q_k(theta | x),      Z_k = E_{p_0}[phi_k],

so drawing one atom label ``K ~ Categorical(pi)``, ``pi_k ∝ b_k Z_k``, and then
running the exact exponential--quadratic transport of that single atom for the
whole trajectory reproduces ``q``.  Nothing is estimated along the trajectory;
the only estimated quantities are the ``K`` global scalars ``Z_k``.

Two evidence estimators are provided; they are not interchangeable:

``plugin``
    ``Z_k ≈ mean_n phi_k(theta_n)`` on a reusable bank of clean base samples.
    This is the training-free estimator used for Spectra's bank weights.  Its
    effective sample size is reported with it because on the Gaussian Linear
    shifts it drops to an ESS of order 1 at ``N = 5000``: the atom is much
    narrower than the base posterior, so almost no bank point lands in it.
``analytic``
    the closed-form target-posterior component weights, available for Gaussian
    Linear only; this is ``pi*``, the component-transport ceiling.

All evidence arithmetic is in log space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtomSet:
    """``r ∝ sum_k exp(log_b_k) N(theta; nu_k, I / kappa_k)`` with ``kappa_k > 0``."""

    log_b: np.ndarray  # (K,)
    nu: np.ndarray  # (K, d)
    kappa: np.ndarray  # (K,)

    @property
    def num_atoms(self) -> int:
        return int(self.log_b.shape[0])

    @property
    def dim(self) -> int:
        return int(self.nu.shape[-1])

    @property
    def a(self) -> np.ndarray:
        """The exponential--quadratic natural parameter, ``a_k = kappa_k nu_k``.

        ``phi_k(theta) ∝ exp(a_k^T theta - kappa_k/2 ||theta||^2)``, which is what
        :func:`spectra.transport.tq_target_score` consumes; the
        dropped normaliser is a per-atom constant and is already carried by
        ``log_b`` where it matters (the evidence), not by the score.
        """
        return self.kappa[:, None] * self.nu

    def as_dict(self) -> dict:
        return {
            "num_atoms": self.num_atoms,
            "log_b": self.log_b.tolist(),
            "nu": self.nu.tolist(),
            "kappa": self.kappa.tolist(),
        }


def _require_isotropic_cov(cov: np.ndarray, what: str) -> float:
    """Return the single variance of an isotropic covariance, or raise.

    Spectra's per-atom transport is the isotropic exponential--quadratic family, so
    an anisotropic atom has no exact transformed-query form.  Every shipped
    benchmark prior is isotropic; this check enforces that where the Spectra
    parameters are built, instead of using the diagonal of a non-isotropic
    covariance.
    """
    cov = np.asarray(cov, float)
    d = cov.shape[0]
    v = float(cov[0, 0])
    if not np.allclose(cov, v * np.eye(d), rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"{what} is not isotropic; Spectra's exact per-atom transport requires "
            f"cov = v I, got diag={np.diag(cov)} offdiag_max="
            f"{np.abs(cov - np.diag(np.diag(cov))).max():.3g}"
        )
    if v <= 0:
        raise ValueError(f"{what} has non-positive variance {v}")
    return v


def atoms_from_shift(shift) -> AtomSet:
    """Read the canonical atoms straight off a :class:`PriorShift`.

    ``PriorShift`` already stores the exact expansion with positive weights, so
    nothing is fitted here: ``kappa_k = 1 / cov_k[0,0]``, ``nu_k`` is the stored
    component mean and ``log_b_k`` the stored component log weight.  The
    single-component ``expquad`` field is not used: it exists only for
    ``K = 1`` and Spectra needs every atom.
    """
    lw = np.asarray(shift.comp_log_weights, float)
    nu = np.asarray(shift.comp_means, float)
    covs = np.asarray(shift.comp_covs, float)
    kappa = np.empty(lw.shape[0])
    for k in range(lw.shape[0]):
        v = _require_isotropic_cov(covs[k], f"{shift.task} ratio atom {k}")
        kappa[k] = 1.0 / v
    return AtomSet(log_b=lw, nu=nu, kappa=kappa)


def ladder_geometry(atoms, i: int, j: int) -> dict:
    """Everything the ladder rule needs, from the atoms alone.

    Independent of any sample, model or candidate estimator: the ladder length
    is a property of the prior, fixed before the run.
    """
    lam_i, lam_j = 1.0 / atoms.kappa[i], 1.0 / atoms.kappa[j]
    common = bool(np.isclose(lam_i, lam_j, rtol=1e-10, atol=0.0))
    sep = float(np.linalg.norm(atoms.nu[i] - atoms.nu[j]))
    if not common:
        return {"common_bandwidth": False, "atom_sep": sep,
                "lambda": float("nan"), "d2_over_lambda": float("nan"),
                "kl_atom": float("nan"), "ladder_M": -1,
                "per_edge_kl": float("nan")}
    d2 = sep**2 / lam_i
    kl = 0.5 * d2
    m = int(math.ceil(math.sqrt(kl))) if kl > 0 else 1
    return {"common_bandwidth": True, "atom_sep": sep, "lambda": float(lam_i),
            "d2_over_lambda": d2, "kl_atom": kl, "ladder_M": m,
            "per_edge_kl": kl / m**2}


# ------------------------------------------------------------- evidence ----


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def log_phi(bank: np.ndarray, atoms: AtomSet) -> np.ndarray:
    """``log phi_k(theta_n)`` for every bank point and atom, shape ``(N, K)``."""
    bank = np.atleast_2d(np.asarray(bank, float))
    d = bank.shape[-1]
    out = np.empty((bank.shape[0], atoms.num_atoms))
    for k in range(atoms.num_atoms):
        v = 1.0 / atoms.kappa[k]
        r2 = np.sum((bank - atoms.nu[k]) ** 2, axis=-1)
        out[:, k] = -0.5 * r2 / v - 0.5 * d * np.log(2.0 * np.pi * v)
    return out


def evidence_plugin(bank: np.ndarray, atoms: AtomSet, *, num_bootstrap: int = 0,
                    seed: int = 0) -> dict:
    """``log Z_k`` by direct averaging, with the ESS that says whether to trust it.

    ``ESS_k`` is ``(sum_n phi_k)^2 / sum_n phi_k^2`` and
    ``rse_k ≈ sqrt(1/ESS_k - 1/N)`` the implied relative standard error of
    ``Zhat_k``.  Both are always recorded; on Gaussian Linear they give
    ``ESS ≈ 1``, which explains why the direct evidence estimate fails there.

    The optional bootstrap resamples bank indices and returns the spread of the
    two-atom log-odds, which is the quantity the sampler uses.
    """
    bank = np.asarray(bank, float)
    n = bank.shape[0]
    lp = log_phi(bank, atoms)
    log_z = _logsumexp(lp, axis=0) - np.log(n)
    ess = np.exp(2.0 * _logsumexp(lp, axis=0) - _logsumexp(2.0 * lp, axis=0))
    rse = np.sqrt(np.maximum(1.0 / ess - 1.0 / n, 0.0))
    out = {
        "estimator": "plugin",
        "log_z": log_z,
        "ess": ess,
        "rse": rse,
        "n_bank": int(n),
        "max_log_phi": lp.max(axis=0),
        "n_within_10_nats_of_max": (lp > lp.max(axis=0) - 10.0).sum(axis=0),
    }
    if num_bootstrap > 0 and atoms.num_atoms == 2:
        rng = np.random.default_rng(seed)
        odds = np.empty(num_bootstrap)
        for b in range(num_bootstrap):
            idx = rng.integers(0, n, size=n)
            lz = _logsumexp(lp[idx], axis=0) - np.log(n)
            odds[b] = (atoms.log_b[0] + lz[0]) - (atoms.log_b[1] + lz[1])
        out["bootstrap_log_odds_sd"] = float(np.std(odds, ddof=1))
        out["bootstrap_log_odds_q05"] = float(np.quantile(odds, 0.05))
        out["bootstrap_log_odds_q95"] = float(np.quantile(odds, 0.95))
        out["num_bootstrap"] = int(num_bootstrap)
    return out


def atom_log_weights(atoms: AtomSet, log_z: np.ndarray) -> np.ndarray:
    """Normalised ``log pi_k`` from ``log b_k + log Z_k``."""
    l = np.asarray(atoms.log_b, float) + np.asarray(log_z, float)
    return l - _logsumexp(l, axis=0)


def log_odds(atoms: AtomSet, log_z: np.ndarray) -> float:
    """``log(b_1 Z_1) - log(b_2 Z_2)`` for the two-atom case, else ``nan``.

    Reported instead of the weight difference because the failure mode is
    multiplicative: a weight of ``1 - 4e-13`` and one of ``1 - 7e-5`` look
    identical on the simplex and are 19 nats apart here.
    """
    if atoms.num_atoms != 2:
        return float("nan")
    l = np.asarray(atoms.log_b, float) + np.asarray(log_z, float)
    return float(l[0] - l[1])


def analytic_atom_weights(x_o, shift, simulator_scale: float | None = None
                          ) -> np.ndarray:
    """``pi*`` for Gaussian Linear, from the closed-form target posterior.

    The target posterior's component weights and ``pi_k ∝ b_k E_{p_0}[phi_k]``
    are the same quantity (the first is the conjugate update of the test prior,
    the second the atom decomposition of the same product), so this also serves
    as a cross-check of the atom bookkeeping (``tests/test_atoms.py`` compares
    the two).
    """
    from spectra.references import (
        GL_SIMULATOR_SCALE,
        gaussian_linear_posterior,
    )

    scale = GL_SIMULATOR_SCALE if simulator_scale is None else simulator_scale
    return np.asarray(gaussian_linear_posterior(x_o, shift, scale).weights, float)
