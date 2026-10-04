#!/usr/bin/env python
"""The controlled anisotropy boundary on Two Moons: cell definition and run plan.

Every cell keeps the shift-severity sweep's ``mild_20`` target (the ``mild_0``
prior centre and its ten observations, marginal scale ``s = 0.20 * s_tr``), the
shipped Two Moons checkpoints and the practical
``(N, L) = (25, 8)`` sampler; only the covariance anisotropy changes at fixed
determinant ``s^4``:

    Sigma_gamma^(+) = diag(s^2 / gamma, s^2 gamma)
    Sigma_gamma^(-) = diag(s^2 gamma, s^2 / gamma)

so the per-axis sd ratio is ``gamma`` and the condition number ``gamma^2``.

``gamma = 1`` is the exact single-factor control and runs through the K=1 exact
factor on ``mild_20`` (``scripts/sample_single_factor.py``).  ``gamma > 1`` is
outside the isotropic exact class: Spectra sees a compiled positive isotropic
mixture (``scripts/anisotropy_compiler.py``), PriorGuide the exact anisotropic
ratio.  A compiled ``gamma = 1`` dictionary exists only as a compiler sanity
diagnostic and is not used in any method curve.

Cell ids are shared by three objects: the exact target (``aniso_<id>``, a cell
record in ``data/anisotropy/cells.json`` rather than a prior JSON, because the
prior JSON schema cannot express an anisotropic target and ``build_prior_shift``
rejects one), the compiled dictionary (``mixture_<id>``, an ordinary isotropic mixture
prior JSON) and the reference directory.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "results" / "anisotropy")
# the compiled cells ship with the other benchmark files in the repository's data directory
DATA_ROOT = str(REPO / "data")
CELLS_JSON = str(REPO / "data" / "anisotropy" / "cells.json")
# the (100, 0) Base bank of the SIR rows: the mild_0 single-factor controlled run
SIR_BANK_ROOT = str(REPO / "results" / "single_factor_controlled_mild")

TASK = "two_moons"
SOURCE_PRIOR_TYPE, SOURCE_PRIOR_ID = "mild", 20  # the severity sweep's width 0.20 at the mild_0 centre
SIR_SOURCE_PRIOR = ("mild", 0)                   # the bank's cell: same centre + observations

OBS_SEEDS = (1000000, 1000001, 1000002, 1000003, 1000004,
             1000005, 1000008, 1000009, 1000010, 1000012)
MODELS = (0, 1, 2)
SAMPLER_SEEDS = (0, 1, 2)
NUM_SAMPLES = 1000
CONFIG = (25, 8)
GRID = "power"
CHECKPOINT = str(REPO / "checkpoints" / "two_moons" / "model_{m}.pkl")

# Path-space budget: the practical protocol of the mixture and component-count studies.
PS_SEEDS = (0, 1, 2)
PS_TRAJECTORIES = 4096
PS_STEPS = 400
PS_GRID = "power"

CAPACITY = "medium"  # the correlated-prior study's capacity (spread factor 6.0); not swept
MASS_THRESHOLD = 0.05

# id -> (gamma, orientation).  60 is the K=1 exact control, 61 its compiled
# sanity twin; 62..65 are the anisotropic cells.
CELLS = {
    60: {"gamma": 1.0, "orientation": "iso", "role": "exact_control"},
    61: {"gamma": 1.0, "orientation": "iso", "role": "compiler_sanity"},
    62: {"gamma": 2.0, "orientation": "plus", "role": "anisotropic"},
    63: {"gamma": 2.0, "orientation": "minus", "role": "anisotropic"},
    64: {"gamma": 4.0, "orientation": "plus", "role": "anisotropic"},
    65: {"gamma": 4.0, "orientation": "minus", "role": "anisotropic"},
}
ANISO_CELLS = tuple(c for c, v in CELLS.items() if v["role"] == "anisotropic")
COMPILED_CELLS = tuple(c for c, v in CELLS.items() if v["role"] != "exact_control")

# Grid for the exact / compiled target posteriors: the training box padded as in
# the correlated-prior compilation, refined 2048 -> 4096 so the discretisation
# drift can be measured.
GRID_PAD = 0.3
GRID_LEVELS = (2048, 4096)
REF_NUM_SAMPLES = 10000


def cov_diag(cell: int, s: float) -> tuple:
    """Per-axis variances of the exact target in model coordinates."""
    g = CELLS[cell]["gamma"]
    o = CELLS[cell]["orientation"]
    if o == "minus":
        return (s * s * g, s * s / g)
    return (s * s / g, s * s * g)


def tag() -> str:
    n, l = CONFIG
    return f"steps{n}_L{l}_{GRID}"


def checkpoint(model: int) -> str:
    return CHECKPOINT.format(m=model)


def k_eff(alpha) -> float:
    import numpy as np
    a = np.asarray(alpha, float)
    a = a / a.sum()
    return float(1.0 / np.sum(a * a))


def load_cells(path=None) -> dict:
    """``cell id -> record`` from the cell file written by ``prepare_anisotropy_priors.py``."""
    import json
    rec = json.loads(Path(path or CELLS_JSON).read_text())
    return {int(k): v for k, v in rec["cells"].items()}


if __name__ == "__main__":
    import json
    print(json.dumps({"root": ROOT, "cells": CELLS, "config": CONFIG,
                      "obs": OBS_SEEDS, "models": MODELS}, indent=2))
