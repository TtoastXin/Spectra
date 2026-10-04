#!/usr/bin/env python
"""Configuration of the correlated-prior ``K = 64`` study.

The compiled dictionary (prior ``mixture_21``) is reused unchanged; Path-Space
estimates the component weights at the practical budget and Spectra samples with
those weights (``spectra_ps``) next to the reference weights (``spectra_ref``).

Every value follows the protocol of the main-table studies; none is tuned on a
result.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
K64_ROOT = str(REPO / "results" / "correlated_prior")
# the K = 64 references and compilation records are written next to the weights
REFERENCE_ROOT = K64_ROOT


def checkpoint(task: str, model: int) -> str:
    return str(REPO / "checkpoints" / task / f"model_{model}.pkl")


K64_TASK = "two_moons"
K64_PRIOR_TYPE, K64_PRIOR_ID, K64_NUM_ATOMS = "mixture", 21, 64
K64_OBS = (1000003, 1000005, 1000010)
K64_MODELS = (0, 1, 2)
K64_SAMPLER_SEEDS = (0, 1, 2)
K64_SAMPLER = (100, 0)                            # Protocol A, as in the main table
K64_TAG = "steps100_L0_power"
PS_SEEDS, PS_TRAJECTORIES, PS_STEPS, PS_GRID = (0, 1, 2), 4096, 400, "power"

K64_WEIGHT_SOURCES = ("reference", "pathspace")
K64_ROW_LABEL = {"reference": "spectra_ref", "pathspace": "spectra_ps"}
K64_MASS_THRESHOLD = 0.05   # "components with posterior mass >= 0.05"


def k64_checkpoint(model: int) -> str:
    return checkpoint(K64_TASK, model)


def k_eff(alpha) -> float:
    s = sum(float(a) ** 2 for a in alpha)
    return 1.0 / s if s > 0 else float("nan")
