#!/usr/bin/env python
"""The fixed-centre shift-severity sweep: cell definition and run plan.

Every cell keeps the ``mild_0`` prior centre, the ``mild_0`` observations, the
trained checkpoints and the practical ``(N, L) = (25, 8)`` sampler; only the
target-prior marginal width changes:

    sigma(lambda) = (lambda / 0.5) * sigma_mild_shipped,

so ``lambda = 0.50`` reproduces the shipped ``mild_0`` width exactly and
``lambda = 0.20`` lands on the shipped ``strong`` width (at the mild centre, so
this cell is not the Table 1 ``strong_0`` row, whose centre and observations
are different).  ``lambda`` is the width in units of the training
prior's marginal standard deviation; every benchmark task here trains on
``U([-1, 1]^d)``, whose marginal sd is ``2 / sqrt(12) = 0.57735``.

The cells are ordinary ``mild_<id>`` priors with ``id = round(100 * lambda)`` in
``data/``; ``scripts/prepare_severity_priors.py`` writes and checks them.
"""

from __future__ import annotations

# lambda -> prior id.  Ordered from the mildest to the sharpest shift.
LAMBDAS = (0.50, 0.40, 0.30, 0.25, 0.20)
PRIOR_TYPE = "mild"
SOURCE_PRIOR_ID = 0  # the shipped cell that fixes centre + observations
ANCHOR_LAMBDA = 0.50


def prior_id(lam: float) -> int:
    """``lambda -> mild_<id>``; ids 20/25/30/40/50 are free upstream (0..9)."""
    pid = int(round(lam * 100))
    if not 10 <= pid <= 99 or abs(pid / 100.0 - lam) > 1e-12:
        raise ValueError(f"lambda {lam!r} does not map to a two-digit prior id")
    return pid


PRIOR_IDS = tuple(prior_id(l) for l in LAMBDAS)

# P0 uses the full ten-observation benchmark grid; P1 (SLCP) is the qualitative
# replication and uses the first three held-out observations in the split
# order (1000003, 1000004, 1000005), chosen before any result was seen.
OBS_SEEDS_P0 = (1000000, 1000001, 1000002, 1000003, 1000004,
                1000005, 1000008, 1000009, 1000010, 1000012)
OBS_SEEDS_SLCP = (1000003, 1000004, 1000005)

MODELS = (0, 1, 2)
SAMPLER_SEEDS = (0, 1, 2)
METHODS = ("pg_exact", "spectra")
NUM_SAMPLES = 1000
GRID = "power"
CONFIG = (25, 8)  # the practical budget of the PriorGuide paper, not tuned here

TASKS = ("two_moons", "oup", "slcp")
TASK_LABEL = {"two_moons": "Two Moons", "oup": "OUP", "slcp": "SLCP"}

# Reference protocol, the same as for the mild/strong cells these runs are
# compared against.
REFERENCE = {
    "two_moons": {"method": "quadrature", "levels": "1024,2048,4096",
                  "num_samples": 10000, "compare_rejection": True},
    "oup": {"method": "quadrature", "levels": "1024,2048,4096",
            "num_samples": 10000, "compare_rejection": False},
    "slcp": {"method": "smc", "particles": 40000, "smc_seeds": "0,1,2,3",
             "smc_mcmc": 48, "num_samples": 10000},
}

# Scoring convention for the whole sweep: the target and the reference are both
# taken inside the training support, i.e. the ``support_matched`` reference
# kind.  compute_metrics.py also writes ``full`` rows; they stay in the raw
# metrics but are not used for the curves.
REFERENCE_KIND = "support_matched"


def obs_seeds(task: str) -> tuple:
    return OBS_SEEDS_SLCP if task == "slcp" else OBS_SEEDS_P0


def tag() -> str:
    n, l = CONFIG
    return f"steps{n}_L{l}_{GRID}"


def cells():
    """Every ``(task, lambda, model, obs, seed, method)`` row the sweep needs."""
    for task in TASKS:
        for lam in LAMBDAS:
            for model in MODELS:
                for obs in obs_seeds(task):
                    for seed in SAMPLER_SEEDS:
                        for method in METHODS:
                            yield task, lam, model, obs, seed, method
