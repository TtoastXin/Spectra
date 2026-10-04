#!/usr/bin/env python
"""Clean-space SIR for the anisotropy cells, on the existing (100, 0) Base bank.

SIR reuses the ``sir_n20000`` Base bank of the ``mild_0`` single-factor
controlled run (``slurm/single_factor_controlled.sh``) rather than drawing a
(25, 8) one: the same centre, the same ten observations and the same three
checkpoints as the anisotropy cells, and a Base bank depends only on the
observation and the checkpoint, not on the target prior.  Each cell reweights it
with the exact target density ``w = N(theta; mu, Sigma_gamma)`` (uniform
training prior; bank points outside the training box get zero weight, as
``spectra.baselines.snis_weights`` does) and resamples 1000 draws with
replacement under the single-factor SIR RNG convention
``default_rng(4_000_000 + obs_seed + n_bank)``.

``--self-test`` first reweights every bank with the ``mild_0`` target and requires
the stored resample indices back exactly, which checks that this implementation
reproduces the single-factor SIR.

    python scripts/sample_anisotropy_sir.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark, compile_atoms  # noqa: E402
from spectra.utils import git_state  # noqa: E402
from scripts import anisotropy_config as G  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
N_BANK = 20000
BOX = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))


def bank_path(bank_root, model: int, obs: int) -> Path:
    t, i = G.SIR_SOURCE_PRIOR
    return (Path(bank_root) / "samples" / G.TASK / f"sir_n{N_BANK}" / f"model_{model}"
            / f"{t}_{i}" / f"obs_{obs}" / "steps100_L0_power")


def weights(log_w: np.ndarray, bank: np.ndarray) -> dict:
    outside = ~np.all((bank >= BOX[0]) & (bank <= BOX[1]), axis=1)
    log_w = np.where(outside, -np.inf, log_w)
    m = np.max(log_w)
    w = np.exp(log_w - m)
    s1, s2 = w.sum(), (w ** 2).sum()
    wn = w / s1
    order = np.sort(wn)[::-1]
    return {"log_w": log_w, "w": wn, "ess": float(s1 ** 2 / s2),
            "ess_frac": float(s1 ** 2 / s2) / bank.shape[0],
            "max_weight": float(order[0]),
            "top1pct_weight_mass": float(order[: max(1, bank.shape[0] // 100)].sum()),
            "outside_box": int(outside.sum())}


def resample(obs: int, w: np.ndarray, bank: np.ndarray, num: int):
    rng = np.random.default_rng(4_000_000 + obs + N_BANK)
    idx = rng.choice(bank.shape[0], size=num, replace=True, p=w)
    return bank[idx], idx


def gaussian_log_prob(mu, cov, pts):
    return compile_atoms.CorrelatedGaussianPrior(
        mu=np.asarray(mu, float), cov=np.asarray(cov, float), box=None, rule="").log_prob(pts)


def self_test(bank_root) -> dict:
    mild0 = benchmark.load_prior_json(G.TASK, *G.SIR_SOURCE_PRIOR)
    mu, sd = np.asarray(mild0["mu"], float), np.asarray(mild0["sigma"], float)
    out = {"exact_index_match": 0, "cells": 0, "max_abs_w_diff": 0.0}
    for m in G.MODELS:
        for o in G.OBS_SEEDS:
            z = np.load(bank_path(bank_root, m, o).with_suffix(".npz"))
            w = weights(gaussian_log_prob(mu, np.diag(sd ** 2), z["bank"]), z["bank"])
            _, idx = resample(o, w["w"], z["bank"], z["samples"].shape[0])
            out["cells"] += 1
            out["exact_index_match"] += int(np.array_equal(idx, z["resample_idx"]))
            out["max_abs_w_diff"] = max(out["max_abs_w_diff"],
                                        float(np.max(np.abs(w["w"] - z["weights"]))))
    out["pass"] = out["exact_index_match"] == out["cells"]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=G.ROOT)
    p.add_argument("--bank-root", default=G.SIR_BANK_ROOT,
                   help="results root of the mild_0 single-factor controlled run")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--models", default=",".join(str(m) for m in G.MODELS))
    args = p.parse_args()
    root = Path(args.root)

    if args.self_test:
        st = self_test(args.bank_root)
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        (root / "manifests" / "sir_self_test.json").write_text(json.dumps(st, indent=2))
        print(f"[self-test] {st}", flush=True)
        if not st["pass"]:
            print("SIR SELF-TEST FAILED", flush=True)
            return 1

    cells = G.load_cells()
    targets = {60: ("mild", 20)} | {c: ("aniso", c) for c in G.ANISO_CELLS}
    git = git_state(REPO)
    for m in [int(v) for v in args.models.split(",") if v.strip()]:
        for o in G.OBS_SEEDS:
            src = bank_path(args.bank_root, m, o)
            z = np.load(src.with_suffix(".npz"))
            src_man = json.loads(src.with_suffix(".json").read_text())
            bank = z["bank"]
            for c, (ptype, pid) in targets.items():
                w = weights(gaussian_log_prob(cells[c]["mu"], cells[c]["cov"], bank), bank)
                samples, idx = resample(o, w["w"], bank, G.NUM_SAMPLES)
                out = (root / "samples" / G.TASK / f"sir_n{N_BANK}" / f"model_{m}"
                       / f"{ptype}_{pid}" / f"obs_{o}")
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / "steps100_L0_power.npz", samples=samples,
                                    x_o=z["x_o"], theta_true=z["theta_true"],
                                    log_w=w["log_w"], weights=w["w"], resample_idx=idx)
                man = {k: src_man[k] for k in ("task", "theta_dim", "x_dim", "model_id",
                                               "checkpoint", "checkpoint_sha256_16",
                                               "conditioning", "sampler", "nfe", "cost",
                                               "wall_clock_s", "wall_clock_per_1000_samples_s")
                       if k in src_man}
                man.update({
                    "round": "round16_anisotropy_sir", "cell": c,
                    "gamma": cells[c]["gamma"], "orientation": cells[c]["orientation"],
                    "prior_type": ptype, "prior_id": pid, "obs_seed": o,
                    "method": f"sir_n{N_BANK}", "method_family": "sir",
                    "method_meta": {"method": "sir", "bank_size": N_BANK,
                                    "resampling": "with replacement",
                                    "weights": "w = exact target density on the training box, self-normalised",
                                    "bank_protocol": "(100, 0) mild_0 controlled Base bank, reused unchanged"},
                    "bank_source": str(src.with_suffix(".npz")),
                    "sir": {k: w[k] for k in ("ess", "ess_frac", "max_weight",
                                              "top1pct_weight_mass", "outside_box")}
                           | {"unique_resampled": int(np.unique(idx).size)},
                    "nan_or_inf": int((~np.isfinite(samples)).sum()),
                    "repo_git": git,
                })
                (out / "steps100_L0_power.json").write_text(json.dumps(man, indent=2, default=str))
            print(f"[model {m} obs {o}] SIR rows written for {len(targets)} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
