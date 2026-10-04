#!/usr/bin/env python
"""Prepare SLCP: fix the x standardisation, then write priors, observations, data.

Four steps, all deterministic given the seeds, all writing outside the upstream
checkout:

1. ``moments``: one pilot simulation from the training prior fixes the constant
   affine map applied to ``x``.  It runs first because every later artefact is
   expressed in the standardised coordinate.
2. ``priors``: the 30 test priors, drawn in model coordinates by the same recipe
   upstream uses for its uniform tasks.
3. ``observations``: ``theta_true`` from the test prior truncated to the
   training box, then simulated, per observation seed, as upstream does.
4. ``data``: Simformer training sets, one per model seed.

    python scripts/prepare_slcp.py --step all --out $PRIORGUIDE_LOCAL_DATA
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import slcp  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PRIOR_TYPES = ("mild", "strong", "mixture")


def step_moments(num: int, seed: int) -> dict:
    """Fix ``X_MEAN`` / ``X_STD`` from a pilot over the training prior."""
    rng = np.random.default_rng(seed)
    means = np.zeros(slcp.X_DIM)
    sq = np.zeros(slcp.X_DIM)
    total = 0
    chunk = 200_000
    t0 = time.perf_counter()
    while total < num:
        m = min(chunk, num - total)
        y = rng.uniform(-1.0, 1.0, size=(m, slcp.THETA_DIM))
        x = slcp.simulate(y, rng)
        means += x.sum(axis=0)
        sq += (x**2).sum(axis=0)
        total += m
    mean = means / total
    var = sq / total - mean**2
    blob = {
        "x_mean": mean.tolist(), "x_std": np.sqrt(var).tolist(),
        "num_pilot": int(total), "seed": int(seed),
        "note": "constant affine standardisation of the conditioned variable; "
                "does not touch the theta coordinate",
        "wall_clock_s": time.perf_counter() - t0,
    }
    (REPO / "spectra" / "slcp_x_moments.json").write_text(
        json.dumps(blob, indent=2))
    slcp._load_moments()
    return blob


def step_priors(out: Path, seed: int) -> int:
    d = out / "priors" / "slcp"
    d.mkdir(parents=True, exist_ok=True)
    (d / "training.json").write_text(json.dumps(slcp.training_prior_json(), indent=4))
    priors = slcp.generate_priors(seed=seed)
    for name, blob in priors.items():
        (d / f"{name}.json").write_text(json.dumps(blob, indent=4))
    return len(priors) + 1


def step_observations(out: Path) -> int:
    n = 0
    for ptype in PRIOR_TYPES:
        for pid in range(10):
            prior = json.loads(
                (out / "priors" / "slcp" / f"{ptype}_{pid}.json").read_text())
            d = out / "observations" / "slcp" / f"{ptype}_{pid}"
            d.mkdir(parents=True, exist_ok=True)
            for obs_seed in slcp.OBSERVATION_SEEDS:
                blob = slcp.generate_observation(prior, obs_seed)
                blob.update({"task": "slcp", "prior_type": ptype, "prior_id": pid})
                (d / f"obs_{obs_seed}.json").write_text(json.dumps(blob, indent=4))
                n += 1
    return n


def step_data(out: Path, seeds, num_sims: int) -> list:
    d = out / "training_data" / "slcp"
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        rng = np.random.default_rng(10_000 + seed)
        theta = rng.uniform(-1.0, 1.0, size=(num_sims, slcp.THETA_DIM))
        x = slcp.standardise(slcp.simulate(theta, rng))
        if not (np.isfinite(theta).all() and np.isfinite(x).all()):
            raise RuntimeError("SLCP simulator produced non-finite values")
        path = d / f"sims{num_sims}_seed{seed}.npz"
        np.savez_compressed(path, theta=theta.astype(np.float32),
                            x=x.astype(np.float32))
        path.with_suffix(".json").write_text(json.dumps({
            "task": "slcp", "seed": seed, "num_sims": num_sims,
            "theta_dim": slcp.THETA_DIM, "x_dim": slcp.X_DIM,
            "theta_mean": theta.mean(0).tolist(),
            "theta_std": theta.std(0).tolist(),
            "x_mean": x.mean(0).tolist(), "x_std": x.std(0).tolist(),
            "simulator": "spectra.slcp (native, sbibm-equivalent)",
            "coordinates": "theta in model space y = theta_phys / 3; x standardised",
            "repo_git": git_state(REPO),
        }, indent=2))
        written.append(str(path))
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", default="all",
                   choices=("all", "moments", "priors", "observations", "data"))
    p.add_argument("--out", required=True)
    p.add_argument("--pilot-sims", type=int, default=1_000_000)
    p.add_argument("--pilot-seed", type=int, default=0)
    p.add_argument("--prior-seed", type=int, default=0)
    p.add_argument("--num-sims", type=int, default=10000)
    p.add_argument("--model-seeds", default="0,1,2")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = (("moments", "priors", "observations", "data") if args.step == "all"
             else (args.step,))
    report = {}

    if "moments" in steps:
        blob = step_moments(args.pilot_sims, args.pilot_seed)
        report["moments"] = blob
        print(f"x_mean {np.round(blob['x_mean'], 4)}", flush=True)
        print(f"x_std  {np.round(blob['x_std'], 4)}", flush=True)
    if "priors" in steps:
        n = step_priors(out, args.prior_seed)
        report["priors_written"] = n
        print(f"wrote {n} prior JSONs", flush=True)
    if "observations" in steps:
        n = step_observations(out)
        report["observations_written"] = n
        print(f"wrote {n} observation JSONs", flush=True)
    if "data" in steps:
        seeds = [int(v) for v in args.model_seeds.split(",")]
        w = step_data(out, seeds, args.num_sims)
        report["training_data"] = w
        for path in w:
            print(f"wrote {path}", flush=True)

    # a cheap end-to-end sanity check on the pieces that exist
    if (out / "priors" / "slcp" / "mixture_0.json").is_file():
        from spectra.prior_shift import build_prior_shift
        shift = build_prior_shift("slcp", "mixture", 0, root=out)
        print(f"prior_shift built: K={len(shift.target_pi)} "
              f"sigma={shift.target_sigma[0][0]:.6f} "
              f"outside_box={shift.target_mass_outside_training_box():.5f}",
              flush=True)
    (out / "slcp_prepare_report.json").write_text(json.dumps(report, indent=2,
                                                            default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
