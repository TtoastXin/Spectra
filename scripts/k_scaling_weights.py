#!/usr/bin/env python
"""Path-space component weights for ``K >= 2``, and their cost.

``pathspace.pathspace_evidence`` estimates ``log Zhat_k`` from one proposal
chain per atom, and the ``K = 2`` rule adds only the pairwise combination

    log pi = softmax(log b + log Zhat)   with K = 2 written as a single Delta.

Averaging ``Delta`` over estimator seeds, as that rule does, gives the same
number as averaging each atom's ``log Zhat_k`` over seeds and taking the
difference, because averaging is linear.  So the general-``K`` rule used here
reduces exactly to the ``K = 2`` rule; ``--regression`` checks this on a stored
cell.

Budget, seeds and chain construction are the benchmark ones
(``1000 * seed + 17 * atom`` chain seeds, 4096 trajectories, 400 steps, power
grid).

    python scripts/k_scaling_weights.py --regression      # K=2 sanity first
    python scripts/k_scaling_weights.py --ks 2,4,8,16 --obs-seeds 1000000,...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.utils import file_hash, git_state  # noqa: E402
from spectra.weights import pathspace_weights  # noqa: E402
from scripts import k_scaling_config as KG  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
def regression_check(backbone_factory, args) -> dict:
    """Reproduce a stored K=2 path-space cell before any K>2 run is allowed."""
    cell = KG.REGRESSION_CELL
    # The stored cell lives in the mixture study, not this one, so it cannot be
    # derived from --out.  Default to the configured path; let a caller running
    # under SPECTRA_RESULTS point at their own copy.
    stored_path = Path(args.regression_weights or cell["weights"])
    if not stored_path.is_file():
        raise SystemExit(f"missing stored K=2 weights for the regression check: "
                         f"{stored_path}")
    stored = json.loads(stored_path.read_text())
    shift = build_prior_shift(KG.TASK, cell["prior_type"], cell["prior_id"])
    atoms = atoms_mod.atoms_from_shift(shift)
    _, x_o = benchmark.load_observation(KG.TASK, cell["prior_type"], cell["prior_id"],
                                        cell["obs_seed"])
    # the caller's --checkpoint wins, so the regression runs on the model the rest
    # of the invocation uses rather than always on the shipped one
    reg_ckpt = args.checkpoint or KG.CHECKPOINT.format(m=cell["model_id"])
    backbone = backbone_factory(reg_ckpt)
    print(f"[regression] K=2 cell {cell['prior_type']}_{cell['prior_id']} "
          f"obs {cell['obs_seed']} model {cell['model_id']}", flush=True)
    fresh = pathspace_weights(backbone, x_o, atoms, list(KG.PS_SEEDS),
                              KG.PS_TRAJECTORIES, KG.PS_STEPS, KG.PS_GRID)

    stored_log_z = np.array([[a["log_z"] for a in s["atoms"]]
                             for s in stored["per_seed"]], float)
    fresh_log_z = np.asarray(fresh["log_z_per_seed"], float)
    stored_pi = np.asarray(stored["pi"], float)
    fresh_pi = np.asarray(fresh["pi"], float)
    # the K=2 rule written out, to show the general-K rule reduces to it
    stored_delta = float(np.mean([p["delta"] for p in stored["per_seed"]]))
    fresh_delta = float(np.mean(fresh_log_z[:, 0] - fresh_log_z[:, 1]))
    out = {
        "cell": cell, "stored_weights": str(stored_path),
        "checkpoint": str(reg_ckpt),
        "stored_log_z": stored_log_z.tolist(), "fresh_log_z": fresh_log_z.tolist(),
        "max_abs_log_z_delta": float(np.max(np.abs(stored_log_z - fresh_log_z))),
        "stored_pi": stored_pi.tolist(), "fresh_pi": fresh_pi.tolist(),
        "max_abs_pi_delta": float(np.max(np.abs(stored_pi - fresh_pi))),
        "stored_delta_mean": stored_delta, "fresh_delta_mean": fresh_delta,
        "delta_rule_gap": abs(stored_delta - fresh_delta),
        "fresh_wall_clock_s": fresh["wall_clock_s"],
        "stored_budget": stored.get("budget"), "fresh_budget": fresh["budget"],
        "note": ("regression check that the general-K runner reproduces the "
                 "stored K=2 path-space weights.  Not a statistical test: the "
                 "estimator is deterministic given its seeds, so the expected "
                 "deviation is zero.  The tolerances are element-wise and are an "
                 "engineering choice for cross-device float noise, not a "
                 "calibrated bound"),
    }
    out["budget_matches"] = (
        stored["budget"]["trajectories"] == fresh["budget"]["trajectories"]
        and stored["budget"]["steps"] == fresh["budget"]["steps"]
        and stored["budget"]["grid"] == fresh["budget"]["grid"]
        and list(stored["budget"]["seeds"]) == list(fresh["budget"]["seeds"]))

    # Given its seed set the estimator is deterministic: re-running it twice in
    # one process, and across processes with different thread counts, reproduces
    # every log Z exactly (measured max difference 0.0).  So the expected
    # deviation is zero and a tolerance only has to absorb float differences
    # between devices.  Those have not been measured, so the value below is an
    # engineering choice rather than a calibrated bound.
    #
    # The comparison is element-wise.  Scaling one tolerance by the largest
    # |log Z| in the array would let a component whose own log Z is near zero
    # drift by that whole amount: with log Z of order 1e5 in the array, a 0.5
    # nat error on another component would pass.
    #
    # ``pi`` is the binding check for anything that changes sampling: it is an
    # absolute tolerance on probabilities, so it constrains the mixture weights
    # directly, while the log Z check constrains the evidence that produced them.
    rtol = atol = float(args.regression_rtol)
    log_z_tol = atol + rtol * np.abs(stored_log_z)
    out["regression_rtol"] = rtol
    out["regression_atol"] = atol
    out["log_z_tolerance_max"] = float(np.max(log_z_tol))
    out["log_z_tolerance_min"] = float(np.min(log_z_tol))
    out["pi_tolerance"] = atol
    out["log_z_matches"] = bool(
        np.all(np.abs(stored_log_z - fresh_log_z) <= log_z_tol))
    out["pi_matches"] = out["max_abs_pi_delta"] <= atol
    out["delta_rule_matches"] = (out["delta_rule_gap"]
                                 <= atol + rtol * abs(stored_delta))
    out["passed"] = bool(out["budget_matches"] and out["log_z_matches"]
                         and out["pi_matches"] and out["delta_rule_matches"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ks", default=",".join(str(k) for k in KG.K_P2))
    p.add_argument("--obs-seeds", default=",".join(str(o) for o in KG.OBS_P2))
    p.add_argument("--model-id", default=str(KG.MODELS_P2B[0]))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seeds", default=",".join(str(s) for s in KG.PS_SEEDS))
    p.add_argument("--trajectories", type=int, default=KG.PS_TRAJECTORIES)
    p.add_argument("--steps", type=int, default=KG.PS_STEPS)
    p.add_argument("--grid", default=KG.PS_GRID)
    p.add_argument("--out", default=KG.ROOT)
    p.add_argument("--regression", action="store_true",
                   help="only run the stored-K=2 regression check and exit")
    p.add_argument("--regression-rtol", type=float, default=1e-6,
                   help="element-wise tolerance of the --regression gate, used as both "
                        "the relative and the absolute term (|a-b| <= atol + rtol*|a|).  "
                        "The estimator is deterministic given its seeds, so the expected "
                        "deviation is zero; this is an engineering margin for cross-device "
                        "float noise and has not been calibrated against one")
    p.add_argument("--regression-weights", default=None,
                   help="stored K=2 path-space weights the --regression check reproduces; "
                        "defaults to the configured path under the repository's results/. "
                        "Point this at your own results root when running with "
                        "SPECTRA_RESULTS set")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    nn = upstream_nn()

    def backbone_factory(path):
        ckpt = load_checkpoint(path)
        return Backbone(ckpt, mode="conditional", nn=nn)

    out_root = Path(args.out)
    (out_root / "weights").mkdir(parents=True, exist_ok=True)

    if args.regression:
        res = regression_check(backbone_factory, args)
        path = out_root / "manifests" / "regression.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res[k] for k in
                          ("max_abs_log_z_delta", "log_z_tolerance_min",
                           "log_z_tolerance_max", "max_abs_pi_delta",
                           "pi_tolerance", "delta_rule_gap",
                           "budget_matches", "passed")}, indent=2))
        print(f"-> {path}")
        if not res["budget_matches"]:
            raise SystemExit("regression check ran a different budget than the "
                             "stored cell")
        if not res["passed"]:
            failed = [k for k in ("log_z_matches", "pi_matches", "delta_rule_matches")
                      if not res[k]]
            raise SystemExit("regression check did not reproduce the stored cell: "
                             + ", ".join(failed)
                             + f" (max |log Z| delta {res['max_abs_log_z_delta']:.6g}, "
                               f"element-wise tolerance {res['log_z_tolerance_min']:.6g}"
                               f"..{res['log_z_tolerance_max']:.6g}; max |pi| delta "
                               f"{res['max_abs_pi_delta']:.6g} vs {res['pi_tolerance']:.6g})")
        return 0

    ks = [int(v) for v in args.ks.split(",") if v.strip()]
    obs_seeds = [int(v) for v in args.obs_seeds.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    ckpt_path = args.checkpoint or KG.CHECKPOINT.format(m=args.model_id)
    backbone = backbone_factory(ckpt_path)
    common = {
        "round": "round14_p2b_pathspace", "task": KG.TASK,
        "model_id": args.model_id, "checkpoint": str(Path(ckpt_path).resolve()),
        "checkpoint_sha256_16": file_hash(ckpt_path),
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "devices": [str(d) for d in jax.devices()],
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
    }

    for k in ks:
        pid = KG.prior_id(k)
        shift = build_prior_shift(KG.TASK, KG.PRIOR_TYPE, pid)
        atoms = atoms_mod.atoms_from_shift(shift)
        for obs_seed in obs_seeds:
            path = (out_root / "weights"
                    / f"pathspace_{KG.TASK}_K{k}_p{pid}_o{obs_seed}"
                      f"_model{args.model_id}.json")
            if args.skip_existing and path.is_file():
                print(f"[K={k} obs {obs_seed}] skip (exists)", flush=True)
                continue
            _, x_o = benchmark.load_observation(KG.TASK, KG.PRIOR_TYPE, pid,
                                                obs_seed)
            print(f"[K={k} obs {obs_seed}] path-space over {k} atoms x "
                  f"{len(seeds)} seeds", flush=True)
            t0 = time.perf_counter()
            res = pathspace_weights(backbone, x_o, atoms, seeds, args.trajectories,
                                    args.steps, args.grid)
            res.update({**common, "K": k, "prior_type": KG.PRIOR_TYPE,
                        "prior_id": pid, "obs_seed": obs_seed,
                        "prior_json_sha256_16": benchmark.prior_json_hash(
                            KG.TASK, KG.PRIOR_TYPE, pid),
                        "total_wall_clock_s": time.perf_counter() - t0})
            path.write_text(json.dumps(res, indent=2, default=str))
            print(f"[K={k} obs {obs_seed}] pi={np.round(res['pi'], 4).tolist()} "
                  f"K_eff={KG.k_eff(res['pi']):.2f} "
                  f"estimator wall {res['wall_clock_s']:.1f}s -> {path.name}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
