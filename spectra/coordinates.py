"""Working-coordinate check for each task.

The exact-atom result holds in the coordinates the Simformer runs its VE
diffusion in, not in the physical parameters of the simulator.  If the
physical-to-model map ``T`` is non-linear, a physical-space Gaussian is not a
model-space exponential-quadratic atom.  For an affine ``T``, isotropy of the
atom covariance is preserved only if ``T`` scales every axis by the same
factor; several upstream tasks rescale unequally (OUP ``[1, 2]``, BAV
``[0.35, 0.35, 0.5, 0.35, 1]``).

All shipped prior JSONs are written in model coordinates (the uniform tasks'
training boxes are ``[-1, 1]^d``), so the isotropic atoms are built in the
space the diffusion uses.  :func:`audit` records this per task; a new task
must pass it before its Spectra numbers enter the common table.

The check has two parts:

``transform``
    the analytic description of ``T`` and whether the exp-quad atom family
    survives it (affine yes, non-linear no; isotropy survives only if ``T``
    scales every axis by the same factor).

``identity check``
    a numerical check on one prior: ``log r_direct - log sum_k b_k phi_k``
    must be constant over support points.  For a native exact mixture it is
    constant to machine precision; for a compiled dictionary it is not, and the
    spread is the approximation error of the dictionary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from spectra import atoms as atoms_mod

# ``theta_phys = theta_model * rescale + shift`` as written in upstream
# ``priorg/sim/tasks/task.py``; the model coordinate is the one the diffusion,
# the shipped prior JSONs and every atom in this repo live in.
#
# ``None`` rescale/shift means the task hands sbibm's own parameters straight to
# the network, i.e. ``T = identity``.


@dataclass(frozen=True)
class TaskCoordinates:
    task: str
    theta_dim: int
    train_kind: str  # "gaussian" | "uniform"
    shift: Optional[np.ndarray]  # physical = model * rescale + shift
    rescale: Optional[np.ndarray]
    physical_prior: str  # human-readable, for the manifest
    source: str  # where the transform was read from
    x_transform: str = "none"  # data-side preprocessing, recorded not used

    @property
    def transform_kind(self) -> str:
        if self.shift is None and self.rescale is None:
            return "identity"
        return "affine_diagonal"

    @property
    def is_affine(self) -> bool:
        return True  # every upstream task; a non-affine one must not be added here

    @property
    def isotropy_preserved(self) -> bool:
        """Does an isotropic covariance stay isotropic across ``T``?

        Only when every axis is scaled equally.  Where this is ``False`` a
        physical-space isotropic Gaussian is an anisotropic model-space atom
        and has no exact single-query transport; the shipped priors avoid this
        by being defined in model coordinates.
        """
        if self.rescale is None:
            return True
        r = np.asarray(self.rescale, float)
        return bool(np.allclose(r, r[0], rtol=1e-12, atol=0.0))

    def to_physical(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)
        if self.rescale is None:
            return y
        return y * np.asarray(self.rescale, float) + np.asarray(self.shift, float)

    def to_model(self, theta_phys: np.ndarray) -> np.ndarray:
        theta_phys = np.asarray(theta_phys, float)
        if self.rescale is None:
            return theta_phys
        return (theta_phys - np.asarray(self.shift, float)) / np.asarray(
            self.rescale, float)

    def log_jacobian(self) -> float:
        """``log |d theta_phys / d y|``; constant, hence cancels in any ratio."""
        if self.rescale is None:
            return 0.0
        return float(np.sum(np.log(np.abs(np.asarray(self.rescale, float)))))

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "theta_dim": self.theta_dim,
            "train_kind": self.train_kind,
            "transform_kind": self.transform_kind,
            "is_affine": self.is_affine,
            "isotropy_preserved_under_T": self.isotropy_preserved,
            "shift": None if self.shift is None else np.asarray(self.shift).tolist(),
            "rescale": (None if self.rescale is None
                        else np.asarray(self.rescale).tolist()),
            "log_jacobian_phys_over_model": self.log_jacobian(),
            "jacobian_cancels_in_ratio": True,
            "physical_prior": self.physical_prior,
            "x_transform": self.x_transform,
            "source": self.source,
        }


COORDINATES = {
    "gaussian_linear": TaskCoordinates(
        task="gaussian_linear", theta_dim=10, train_kind="gaussian",
        shift=None, rescale=None,
        physical_prior="N(0, 0.1 I) (sbibm gaussian_linear, dim 10)",
        source="priorg/sim/tasks/task.py SBIBMTask -> sbibm_task.GaussianLinear "
               "(no theta_shift/theta_rescale)"),
    "gaussian_linear_high": TaskCoordinates(
        task="gaussian_linear_high", theta_dim=20, train_kind="gaussian",
        shift=None, rescale=None,
        physical_prior="N(0, 0.1 I) (sbibm gaussian_linear, dim 20)",
        source="priorg/sim/tasks/task.py SBIBMTask -> sbibm_task.GaussianLinear "
               "(no theta_shift/theta_rescale)"),
    "two_moons": TaskCoordinates(
        task="two_moons", theta_dim=2, train_kind="uniform",
        shift=None, rescale=None,
        physical_prior="U([-1, 1]^2) (sbibm two_moons)",
        source="priorg/sim/tasks/task.py SBIBMTask -> sbibm_task.TwoMoons "
               "(no theta_shift/theta_rescale)"),
    "oup": TaskCoordinates(
        task="oup", theta_dim=2, train_kind="uniform",
        shift=np.array([1.0, 0.0]), rescale=np.array([1.0, 2.0]),
        physical_prior="U([0, 2] x [-2, 2]) after theta*rescale+shift",
        source="priorg/sim/tasks/task.py OUPTask.theta_shift / theta_rescale",
        x_transform="(x - 4.18954) / 3.0685966, constant affine"),
    "bav": TaskCoordinates(
        task="bav", theta_dim=5, train_kind="gaussian",
        shift=np.array([math.log(2), math.log(2), math.log(5), math.log(0.3), 0.0]),
        rescale=np.array([0.35, 0.35, 0.5, 0.35, 1.0]),
        physical_prior="log-scale parameters; model prior is N(0, I)",
        source="priorg/sim/tasks/task.py BAVTask.theta_shift / theta_rescale"),
    "slcp": TaskCoordinates(
        task="slcp", theta_dim=5, train_kind="uniform",
        shift=np.zeros(5), rescale=np.full(5, 3.0),
        physical_prior="U([-3, 3]^5) (standard sbibm SLCP)",
        source="this repo, spectra.slcp: y = theta_phys / 3 so the "
               "model-space training prior is U([-1, 1]^5), matching the "
               "upstream uniform-task convention"),
}


def get_coordinates(task: str) -> TaskCoordinates:
    try:
        return COORDINATES[task]
    except KeyError:
        raise ValueError(
            f"no coordinate entry for {task!r}; add its working coordinates "
            "to COORDINATES first") from None


# ------------------------------------------------------- identity check ----


def support_points(shift, rng, num: int) -> tuple[np.ndarray, str]:
    """Prior-only points at which the ratio identity is checked.

    Drawn from the test prior, which is where the atoms carry mass.  For a
    uniform training prior they are then restricted to the training box,
    because the identity ``r = sum_k b_k phi_k`` only holds on
    ``supp(pi_train)``: outside it ``r`` is zero (or undefined) while the atoms
    remain smooth.  The test-prior mass outside the box is reported separately
    by :func:`ratio_identity_check`.
    """
    # Shipped mixture weights are float32 round-trips and do not always sum to
    # one in float64 (bav/mixture_4 is [0.2537069022655487, 0.7462930679321289],
    # 3e-8 short), which is outside ``rng.choice``'s tolerance.  Renormalising
    # leaves weights that already sum to 1.0 unchanged.
    pi = np.asarray(shift.target_pi, float)
    k = rng.choice(len(pi), size=num, p=pi / pi.sum())
    theta = shift.target_mu[k] + shift.target_sigma[k] * rng.standard_normal(
        (num, shift.theta_dim))
    if shift.train_box is None:
        return theta, "test_prior"
    lo, hi = shift.train_box
    inside = np.all((theta >= lo) & (theta <= hi), axis=-1)
    return theta[inside], "test_prior_restricted_to_training_box"


def ratio_identity_check(shift, atoms=None, *, num: int = 4096, seed: int = 0
                         ) -> dict:
    """Is ``log r_direct - log sum_k b_k phi_k`` constant on the support?

    The constant itself is free (every consumer of ``r`` is invariant to it), so
    what is measured is the spread.  ``max_abs_dev`` at 1e-10 or below means
    the mixture equals the analytic ratio; anything larger indicates a compiled
    or mis-specified dictionary, and the number is its approximation error in
    nats.
    """
    if atoms is None:
        atoms = atoms_mod.atoms_from_shift(shift)
    rng = np.random.default_rng(seed)
    theta, point_source = support_points(shift, rng, num)
    if theta.shape[0] == 0:
        return {"status": "no_support_points", "num_points": 0}
    log_direct = np.asarray(shift.log_r(theta), float).reshape(-1)
    lp = atoms_mod.log_phi(theta, atoms)
    log_atoms = atoms_mod._logsumexp(atoms.log_b[None, :] + lp, axis=1)
    dev = log_direct - log_atoms
    c = float(np.median(dev))
    resid = dev - c
    outside = 0.0
    if shift.train_box is not None:
        lo, hi = shift.train_box
        # same float32 round-trip guard as ``support_points`` above
        pi = np.asarray(shift.target_pi, float)
        tot = rng.choice(len(pi), size=num, p=pi / pi.sum())
        full = shift.target_mu[tot] + shift.target_sigma[tot] * rng.standard_normal(
            (num, shift.theta_dim))
        outside = float(np.mean(~np.all((full >= lo) & (full <= hi), axis=-1)))
    return {
        "status": "ok",
        "num_points": int(theta.shape[0]),
        "point_source": point_source,
        "const_offset": c,
        "max_abs_dev": float(np.max(np.abs(resid))),
        "rms_dev": float(np.sqrt(np.mean(resid**2))),
        "test_prior_mass_outside_training_box": outside,
        "native_exact": bool(np.max(np.abs(resid)) < 1e-8),
    }


def common_bandwidth(atoms) -> dict:
    """Do all atoms share one bandwidth?  The score-line route requires it."""
    lam = 1.0 / np.asarray(atoms.kappa, float)
    return {
        "lambda": lam.tolist(),
        "common_bandwidth": bool(np.allclose(lam, lam[0], rtol=1e-10, atol=0.0)),
        "lambda_common": float(lam[0]) if lam.size else float("nan"),
        "sqrt_lambda": float(math.sqrt(lam[0])) if lam.size else float("nan"),
    }


def atom_geometry(atoms, *, base_mean=None, base_var=None) -> dict:
    """Pairwise atom geometry: the screening quantities."""
    nu = np.asarray(atoms.nu, float)
    lam = 1.0 / np.asarray(atoms.kappa, float)
    k = atoms.num_atoms
    dists, dists_sd = [], []
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.linalg.norm(nu[i] - nu[j]))
            dists.append(d)
            dists_sd.append(d / math.sqrt(0.5 * (lam[i] + lam[j])))
    out = {
        "num_atoms": k,
        "atom_sd": np.sqrt(lam).tolist(),
        "pairwise_distance": dists,
        "pairwise_distance_in_atom_sd": dists_sd,
    }
    if base_mean is not None and base_var is not None:
        base_mean = np.asarray(base_mean, float)
        v_b = float(base_var)
        out["base_scale_proxy_sd"] = math.sqrt(v_b)
        out["v_b_over_lambda"] = v_b / float(lam[0])
        out["atom_center_mahalanobis"] = [
            float(np.linalg.norm(nu[i] - base_mean) / math.sqrt(v_b + lam[i]))
            for i in range(k)
        ]
    return out


def audit(task: str, shift, atoms=None, *, num: int = 4096, seed: int = 0,
          base_mean=None, base_var=None) -> dict:
    """The full per-cell record for one task."""
    if atoms is None:
        atoms = atoms_mod.atoms_from_shift(shift)
    coords = get_coordinates(task)
    ident = ratio_identity_check(shift, atoms, num=num, seed=seed)
    band = common_bandwidth(atoms)
    status = "native_exact"
    if not coords.is_affine:
        status = "unsupported_coordinate"
    elif not ident.get("native_exact", False):
        status = "compiled_or_approximate"
    return {
        "coordinates": coords.as_dict(),
        "prior": {
            "task": shift.task, "prior_type": shift.prior_type,
            "prior_id": int(shift.prior_id), "theta_dim": int(shift.theta_dim),
            "train_kind": shift.train_kind,
            "prior_defined_in": "model_coordinates",
        },
        "atoms": atoms.as_dict(),
        "bandwidth": band,
        "geometry": atom_geometry(atoms, base_mean=base_mean, base_var=base_var),
        "identity_check": ident,
        "score_line_supported": bool(band["common_bandwidth"]),
        "status": status,
    }
