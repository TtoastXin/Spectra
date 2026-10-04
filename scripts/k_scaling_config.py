#!/usr/bin/env python
"""The ``K > 2`` Two Moons mixture family of the component-count study.

One nested dictionary of centres, fixed before any accuracy or timing number is
looked at.  A 64-point circle of radius ``R`` around the shipped ``mild_0``
centre is defined once,

    mu_j = mu_0 + R (cos(2 pi j / 64), sin(2 pi j / 64)),   j = 0 .. 63,

and each ``K`` takes every ``64 / K``-th point, so the families are nested:
``K = 2`` is a subset of ``K = 4`` is a subset of ... is a subset of ``K = 64``.
``K = 1`` is the centre itself.  Components carry equal weights ``1 / K`` and a
common isotropic width ``sigma``.  Neither free number is tuned: ``R = 0.5`` is a
quarter of the ``[-1, 1]^2`` training box, and ``sigma`` is the width of the
shipped ``strong`` benchmark shift.

Because the training prior is uniform, ``r = pi_new / pi_tr`` is the mixture
itself, so PriorGuide consumes the exact ``K``-component ratio with no GMM
fitting and Spectra's atoms are the same components with ``kappa = 1 / sigma^2``.

The family serves three fixed roles:

``timing``     online wall-clock vs ``K`` for Spectra, the benchmarked
               PriorGuide and PriorGuide (VJP) at ``K = 1 .. 64``
               (``scripts/timing_k_scaling.py``).
``accuracy``   composition accuracy with reference component weights,
               ``K <= 16`` (``scripts/sample_k_scaling.py``).
``weights``    path-space weight-estimation cost and error, ``K <= 16``,
               three observations, one model (``scripts/k_scaling_weights.py``).
"""

from __future__ import annotations

import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "results" / "k_scaling")
# the K cells ship with the other benchmark files in the repository's data directory
DATA_ROOT = str(REPO / "data")

TASK = "two_moons"
PRIOR_TYPE = "mixture"
BASE_POINTS = 64
RADIUS = 0.5
# the shipped Two Moons ``strong_0`` width; the prepare script reads it back and
# checks that this constant matches the shipped file
SIGMA = 0.11547005921602249
CENTRE_SOURCE = ("mild", 0)  # centre and observations both come from this cell

K_P1 = (1, 2, 4, 8, 16, 32, 64)
K_P2 = (2, 4, 8, 16)
PRIOR_ID_OFFSET = 100  # upstream ships mixture_0..9, so ids from 101 do not collide

OBS_P1 = (1000000,)
OBS_P2 = (1000000, 1000001, 1000002)
MODELS_P2A = (0, 1, 2)
MODELS_P2B = (0,)
SAMPLER_SEEDS = (0, 1, 2)

CONFIG = (25, 8)
GRID = "power"
NUM_SAMPLES = 1000
TIMING_REPEATS = 5

# Path-space estimator budget, the same as in the K=2 study.
PS_SEEDS = (0, 1, 2)
PS_TRAJECTORIES = 4096
PS_STEPS = 400
PS_GRID = "power"

CHECKPOINT = str(REPO / "checkpoints" / "two_moons" / "model_{m}.pkl")
# a K=2 cell with stored path-space weights, used only as a regression check
# that the general-K runner reproduces the K=2 estimator
REGRESSION_CELL = {
    "prior_type": "mixture", "prior_id": 0, "obs_seed": 1000000, "model_id": 0,
    "weights": str(REPO / "results" / "mixture_controlled" / "weights"
                   / "pathspace_two_moons_p0_o1000000_model0.json"),
}

METHODS_P1 = ("spectra", "pg_legacy", "pg_vjp")
METHODS_P2 = ("pg_exact", "spectra_ref")
METHOD_LABEL = {
    "spectra": "SPECTRA", "pg_legacy": "PriorGuide (legacy derivative)",
    "pg_vjp": "PriorGuide (VJP derivative)",
    "pg_exact": "PriorGuide", "spectra_ref": "SPECTRA-Ref",
}

REFERENCE_KIND = "support_matched"  # same scoring convention as the main sweep


def prior_id(k: int) -> int:
    if k not in set(K_P1) | set(K_P2):
        raise ValueError(f"K={k} is not one of this study's K values")
    return PRIOR_ID_OFFSET + k


def centres(k: int, centre) -> list[list[float]]:
    """The nested centre set for ``K``: every ``64 / K``-th point of the circle."""
    cx, cy = float(centre[0]), float(centre[1])
    if k == 1:
        return [[cx, cy]]
    if BASE_POINTS % k:
        raise ValueError(f"K={k} does not divide the {BASE_POINTS}-point circle")
    stride = BASE_POINTS // k
    out = []
    for j in range(0, BASE_POINTS, stride):
        ang = 2.0 * math.pi * j / BASE_POINTS
        out.append([cx + RADIUS * math.cos(ang), cy + RADIUS * math.sin(ang)])
    return out


def neighbour_separation(k: int) -> float:
    """Centre-to-centre distance between adjacent components, in units of sigma."""
    if k < 2:
        return float("inf")
    return 2.0 * RADIUS * math.sin(math.pi / k) / SIGMA


def tag() -> str:
    n, l = CONFIG
    return f"steps{n}_L{l}_{GRID}"


def k_eff(alpha) -> float:
    """Descriptive effective component count ``1 / sum_k alpha_k^2``."""
    import numpy as np

    a = np.asarray(alpha, float)
    a = a / a.sum()
    return float(1.0 / np.sum(a ** 2))


if __name__ == "__main__":
    import json

    print(json.dumps({
        "root": ROOT, "data_root": DATA_ROOT, "task": TASK,
        "radius": RADIUS, "sigma": SIGMA, "base_points": BASE_POINTS,
        "K_p1": K_P1, "K_p2": K_P2,
        "prior_ids": {k: prior_id(k) for k in sorted(set(K_P1) | set(K_P2))},
        "neighbour_separation_in_sigma": {
            k: round(neighbour_separation(k), 3) for k in K_P1 if k > 1},
        "p1_rows": len(K_P1) * len(METHODS_P1),
        "p2a_rows": len(K_P2) * len(OBS_P2) * len(MODELS_P2A) * len(SAMPLER_SEEDS)
        * len(METHODS_P2),
        "p2b_atom_seed_runs": sum(K_P2) * len(PS_SEEDS) * len(OBS_P2) * len(MODELS_P2B),
    }, indent=2))
