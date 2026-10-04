"""Grid-quadrature references for the two-dimensional benchmark tasks.

Two Moons and OUP both have a tractable likelihood and a two-dimensional
parameter, so their posteriors can be computed on a fine tensor grid with the
exact ``log p(x | theta)``.  This gives the normaliser, the moments, the atom
evidences ``Z_k = E_{p_0}[phi_k]`` and the target responsibilities
``pi_k = E_q[gamma_k]`` without Monte-Carlo error; the remaining discretisation
error is measured by refining the grid (:func:`convergence_check`).

The likelihood is the one the simulator code implements.  For OUP, upstream
simulates an Euler-Maruyama chain,

    x_{t+1} = x_t + theta_0 (exp(theta_1) - x_t) dt + 0.5 sqrt(dt) w_t,

so the exact likelihood is the product of those 24 Gaussian transitions at
``dt = 0.2``, not the exact-discretisation OU transition.  Writing
``a = theta_0 dt`` and ``b = exp(theta_1)`` the sum of squared residuals reduces
to five sufficient statistics of the observed trajectory, so the log-likelihood
costs O(1) per grid point regardless of the trajectory length.

Coordinates: everything here is in model coordinates, the same space as the
shipped prior JSONs and the atoms (see :mod:`.coordinates`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from spectra import atoms as atoms_mod
from spectra.benchmark import outside_box_fraction
from spectra.coordinates import get_coordinates

# Upstream OUPTask constants (priorg/sim/tasks/task.py).
OUP_X0 = 10.0
OUP_DT = 0.2
OUP_NUM_POINTS = 25
OUP_DIFFUSION = 0.5  # x_{t+1} = ... + OUP_DIFFUSION * sqrt(dt) * w_t
OUP_X_MEAN = 4.18954
OUP_X_STD = 3.0685966


# ------------------------------------------------------------------ OUP ----


def oup_unstandardise(x_std: np.ndarray) -> np.ndarray:
    """Back to the raw trajectory the transition density is written in."""
    return np.asarray(x_std, float).reshape(-1) * OUP_X_STD + OUP_X_MEAN


def oup_sufficient_stats(x_std: np.ndarray) -> dict:
    """Five scalars that determine the OUP log-likelihood at any ``theta``.

    With ``u_t = x_{t+1} - x_t`` and ``v_t = x_t`` over the 24 transitions,

        sum_t (u_t + a v_t - a b)^2
          = S_uu + 2 a S_uv - 2 a b S_u + a^2 S_vv - 2 a^2 b S_v + T a^2 b^2.
    """
    x = oup_unstandardise(x_std)
    if x.shape[0] != OUP_NUM_POINTS:
        raise ValueError(f"OUP trajectory must have {OUP_NUM_POINTS} points")
    v = x[:-1]
    u = x[1:] - x[:-1]
    return {
        "T": int(v.shape[0]),
        "S_u": float(u.sum()), "S_v": float(v.sum()),
        "S_uu": float((u * u).sum()), "S_uv": float((u * v).sum()),
        "S_vv": float((v * v).sum()),
        "x0_raw": float(x[0]),
    }


def oup_log_likelihood(theta_model: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """Exact ``log p(x | theta)`` of the implemented Euler-Maruyama chain.

    Returned up to the additive constant that is common to every ``theta``
    (the Gaussian normaliser and the standardisation Jacobian), which cancels in
    every posterior, evidence ratio and responsibility computed here.
    """
    theta_model = np.atleast_2d(np.asarray(theta_model, float))
    coords = get_coordinates("oup")
    phys = coords.to_physical(theta_model)
    a = phys[:, 0] * OUP_DT
    b = np.exp(phys[:, 1])
    s = oup_sufficient_stats(x_std)
    var = (OUP_DIFFUSION**2) * OUP_DT
    ss = (s["S_uu"] + 2 * a * s["S_uv"] - 2 * a * b * s["S_u"]
          + a**2 * s["S_vv"] - 2 * a**2 * b * s["S_v"] + s["T"] * a**2 * b**2)
    return -0.5 * ss / var


# ------------------------------------------------------------ Two Moons ----


def two_moons_log_likelihood(theta_model: np.ndarray, x_o: np.ndarray) -> np.ndarray:
    """``log p(x_o | theta)`` reusing the Cartesian density.

    ``-inf`` outside the half-disc, where the parameters cannot have produced
    the observation.
    """
    from spectra.references import _two_moons_likelihood

    dens = _two_moons_likelihood(np.atleast_2d(theta_model),
                                 np.asarray(x_o, float).reshape(-1))
    with np.errstate(divide="ignore"):
        return np.where(dens > 0, np.log(np.maximum(dens, 1e-300)), -np.inf)


LOG_LIKELIHOOD = {
    "oup": oup_log_likelihood,
    "two_moons": two_moons_log_likelihood,
}


def log_likelihood(task: str, theta_model: np.ndarray, x_o: np.ndarray) -> np.ndarray:
    try:
        fn = LOG_LIKELIHOOD[task]
    except KeyError:
        raise ValueError(
            f"no exact likelihood for {task!r}; quadrature references are only "
            f"available for {sorted(LOG_LIKELIHOOD)}") from None
    return fn(theta_model, x_o)


# ------------------------------------------------------------ the grid ----


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


@dataclass
class GridPosterior:
    """A normalised posterior on a tensor grid, in model coordinates."""

    task: str
    kind: str  # "base" | "target"
    box: tuple  # (low, high) arrays
    n: int
    centers: np.ndarray  # (M, 2) cell centres
    log_w: np.ndarray  # (M,) normalised log cell probabilities
    log_norm: float  # log int L * prior d theta, on this grid
    cell_area: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_w)

    def expectation(self, values: np.ndarray) -> float:
        """``E_posterior[f]`` for ``f`` evaluated at the cell centres."""
        values = np.asarray(values, float).reshape(-1)
        return float(np.sum(self.weights * values))

    def log_expectation(self, log_values: np.ndarray) -> float:
        """``log E_posterior[exp(log f)]``, for evidences that underflow."""
        return _logsumexp(self.log_w + np.asarray(log_values, float).reshape(-1))

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.centers

    @property
    def cov(self) -> np.ndarray:
        m = self.mean
        d = self.centers - m
        return (self.weights[:, None] * d).T @ d

    def sample(self, rng, num: int) -> np.ndarray:
        """Draw from the piecewise-uniform density the grid defines.

        Cells are picked by their probability and a point is drawn uniformly
        inside the chosen cell, so the draws are exact for the discretised
        measure and carry the grid's own resolution as their only bias.
        """
        idx = rng.choice(self.centers.shape[0], size=num, p=self.weights)
        h = (self.box[1] - self.box[0]) / self.n
        jitter = (rng.random((num, self.centers.shape[1])) - 0.5) * h
        return self.centers[idx] + jitter

    def edge_mass(self) -> float:
        """Probability sitting in the outermost ring of cells.

        A non-negligible value means the box truncated the posterior and the
        normaliser is wrong.  :func:`grid_posterior` stores it in the
        diagnostics.
        """
        n = self.n
        w = self.weights.reshape(n, n)
        ring = w[0].sum() + w[-1].sum() + w[1:-1, 0].sum() + w[1:-1, -1].sum()
        return float(ring)


def _grid(box, n: int) -> tuple[np.ndarray, float]:
    low, high = np.asarray(box[0], float), np.asarray(box[1], float)
    h = (high - low) / n
    axes = [low[i] + (np.arange(n) + 0.5) * h[i] for i in range(low.shape[0])]
    mesh = np.meshgrid(*axes, indexing="ij")
    centers = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    return centers, float(np.prod(h))


def grid_posterior(task: str, x_o, shift, *, kind: str, box, n: int,
                   log_lik: np.ndarray | None = None,
                   centers: np.ndarray | None = None,
                   cell_area: float | None = None) -> GridPosterior:
    """Normalised posterior of ``kind`` on an ``n x n`` grid over ``box``.

    ``kind="base"`` uses the training prior (what the frozen backbone learned)
    and ``kind="target"`` the shifted test prior.  Both share one likelihood
    evaluation, which the caller may pass in to avoid recomputing it.
    """
    if centers is None:
        centers, cell_area = _grid(box, n)
    if log_lik is None:
        log_lik = log_likelihood(task, centers, x_o)
    log_lik = np.asarray(log_lik, float).reshape(-1)

    if kind == "base":
        if shift.train_box is None:
            raise ValueError("base grid posterior needs a bounded training prior")
        lo, hi = shift.train_box
        inside = np.all((centers >= lo) & (centers <= hi), axis=-1)
        log_prior = np.where(inside, 0.0, -np.inf)
    elif kind == "target":
        # Convention: the target posterior is ``L * pi_new`` with the test
        # prior taken on all of R^d, the same target as the rejection reference
        # and the C2ST scores, so the quadrature can replace it directly.
        log_prior = np.asarray(shift.log_r(centers), float).reshape(-1)
    elif kind == "target_restricted":
        # ``support_matched``: the same posterior cut to supp(pi_train).  A
        # frozen base posterior can place no clean mass outside the training
        # box, so this is the target structured transport is exact for, and the
        # gap between the two is the irreducible coverage floor.
        log_prior = np.asarray(shift.log_r(centers), float).reshape(-1)
        if shift.train_box is not None:
            lo, hi = shift.train_box
            inside = np.all((centers >= lo) & (centers <= hi), axis=-1)
            log_prior = np.where(inside, log_prior, -np.inf)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    log_joint = log_lik + log_prior
    lse = _logsumexp(log_joint)
    if not np.isfinite(lse):
        raise RuntimeError(f"{task} {kind} grid has no finite mass; check the box")
    log_w = log_joint - lse
    post = GridPosterior(
        task=task, kind=kind, box=(np.asarray(box[0], float),
                                   np.asarray(box[1], float)),
        n=n, centers=centers, log_w=log_w,
        log_norm=lse + math.log(cell_area), cell_area=cell_area,
    )
    post.diagnostics["edge_mass"] = post.edge_mass()
    post.diagnostics["finite_cells"] = int(np.isfinite(log_joint).sum())
    post.diagnostics["effective_cells"] = float(
        np.exp(-np.sum(np.exp(log_w) * np.where(np.isfinite(log_w), log_w, 0.0))))
    if shift.train_box is not None and kind != "base":
        post.diagnostics["prior_mass_outside_training_box"] = (
            shift.target_mass_outside_training_box())
    return post


def default_box(task: str, shift, *, pad_sd: float = 8.0) -> tuple:
    """A box that covers the training support and the test prior's bulk."""
    lo = np.full(shift.theta_dim, -np.inf)
    hi = np.full(shift.theta_dim, np.inf)
    if shift.train_box is not None:
        lo, hi = (np.asarray(shift.train_box[0], float).copy(),
                  np.asarray(shift.train_box[1], float).copy())
    plo = (shift.target_mu - pad_sd * shift.target_sigma).min(axis=0)
    phi = (shift.target_mu + pad_sd * shift.target_sigma).max(axis=0)
    return np.minimum(lo, plo), np.maximum(hi, phi)


def convergence_check(task: str, x_o, shift, *, kind: str, box,
                      levels=(1024, 2048, 4096)) -> dict:
    """Refine the grid and report how much the answers move.

    The reported quantities are the ones the references use: the posterior
    mean, the log normaliser and (for a mixture) the atom log-odds.  A reference
    is only called ``analytic``-grade if these stop moving.
    """
    atoms = atoms_mod.atoms_from_shift(shift)
    out = []
    for n in levels:
        centers, area = _grid(box, n)
        ll = log_likelihood(task, centers, x_o)
        post = grid_posterior(task, x_o, shift, kind=kind, box=box, n=n,
                              log_lik=ll, centers=centers, cell_area=area)
        row = {
            "n": n, "log_norm": post.log_norm,
            "edge_mass": post.diagnostics["edge_mass"],
            "mean": post.mean.tolist(),
            "trace_cov": float(np.trace(post.cov)),
        }
        if kind == "base":
            lz = atom_evidence_on_grid(post, atoms)
            row["log_odds"] = atoms_mod.log_odds(atoms, lz["log_z"])
        else:
            resp = responsibility_masses(post, atoms)
            row["log_odds"] = resp["log_odds"]
            row["pi"] = resp["pi"].tolist()
        out.append(row)
    diffs = {}
    if len(out) >= 2:
        a, b = out[-2], out[-1]
        diffs = {
            "d_log_norm": b["log_norm"] - a["log_norm"],
            "d_log_odds": b["log_odds"] - a["log_odds"],
            "d_mean_max": float(np.max(np.abs(np.array(b["mean"])
                                              - np.array(a["mean"])))),
        }
    return {"levels": out, "last_refinement": diffs}


# --------------------------------------------------- atom mass on a grid ----


def atom_evidence_on_grid(base_post: GridPosterior, atoms) -> dict:
    """``log Z_k = log E_{p_0}[phi_k]`` with no Monte-Carlo error.

    This is the same functional the direct bank plug-in estimates, evaluated
    exactly, so it measures the plug-in's error.  On a narrow atom the plug-in
    is a rare-event estimator; this is not.
    """
    lp = atoms_mod.log_phi(base_post.centers, atoms)
    log_z = np.array([base_post.log_expectation(lp[:, k])
                      for k in range(atoms.num_atoms)])
    return {"estimator": "grid_quadrature", "log_z": log_z,
            "log_pi": atoms_mod.atom_log_weights(atoms, log_z),
            "log_odds": atoms_mod.log_odds(atoms, log_z)}


def responsibility_masses(target_post: GridPosterior, atoms) -> dict:
    """``pi_k = E_q[gamma_k]`` with ``gamma_k = b_k phi_k / sum_j b_j phi_j``.

    The bounded reference.  On a grid it is exact, so it also checks that the
    bounded form and the ``b_k Z_k`` form agree: they are the same number
    computed two ways, and disagreement means the atom bookkeeping is wrong.
    """
    lp = atoms_mod.log_phi(target_post.centers, atoms)
    lb = np.asarray(atoms.log_b, float)[None, :] + lp
    log_gamma = lb - atoms_mod._logsumexp(lb, axis=1)[:, None]
    pi = np.array([target_post.expectation(np.exp(log_gamma[:, k]))
                   for k in range(atoms.num_atoms)])
    pi = pi / pi.sum()
    lo = float("nan")
    if atoms.num_atoms == 2:
        lo = float(np.log(pi[0]) - np.log(pi[1]))
    return {"estimator": "grid_responsibility", "pi": pi, "log_pi": np.log(pi),
            "log_odds": lo,
            "gamma_range": [float(np.exp(log_gamma).min()),
                            float(np.exp(log_gamma).max())]}


def responsibility_masses_from_samples(samples: np.ndarray, atoms, *,
                                       num_shards: int = 4, seed: int = 0) -> dict:
    """The same bounded estimator on target-posterior samples.

    Used where no grid is available (any task with ``theta_dim > 2``).  Because
    ``gamma_k`` is bounded in ``[0, 1]`` the shard spread is a meaningful
    uncertainty, unlike the unbounded base-to-narrow-atom plug-in.
    """
    samples = np.atleast_2d(np.asarray(samples, float))
    lp = atoms_mod.log_phi(samples, atoms)
    lb = np.asarray(atoms.log_b, float)[None, :] + lp
    gamma = np.exp(lb - atoms_mod._logsumexp(lb, axis=1)[:, None])
    pi = gamma.mean(axis=0)
    pi = pi / pi.sum()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(samples.shape[0])
    shards = np.array_split(idx, num_shards)
    shard_pi = np.stack([gamma[s].mean(axis=0) for s in shards])
    shard_pi = shard_pi / shard_pi.sum(axis=1, keepdims=True)
    lo = float("nan")
    lo_sd = float("nan")
    if atoms.num_atoms == 2:
        lo = float(np.log(pi[0]) - np.log(pi[1]))
        with np.errstate(divide="ignore"):
            shard_lo = np.log(shard_pi[:, 0]) - np.log(shard_pi[:, 1])
        lo_sd = float(np.std(shard_lo, ddof=1) / math.sqrt(num_shards))
    return {"estimator": "sample_responsibility", "pi": pi,
            "log_odds": lo, "log_odds_se": lo_sd,
            "num_samples": int(samples.shape[0]), "num_shards": num_shards,
            "shard_pi": shard_pi,
            "mc_se_pi": (shard_pi.std(axis=0, ddof=1) / math.sqrt(num_shards)),
            "gamma_mean": gamma.mean(axis=0), "gamma_min": gamma.min(axis=0),
            "gamma_max": gamma.max(axis=0)}


def outside_box(samples: np.ndarray, shift) -> float:
    if shift.train_box is None:
        return 0.0
    return outside_box_fraction(np.asarray(samples, float), shift.train_box)
