#!/usr/bin/env python
"""Re-sample the two-component posterior trajectories under the practical corrector.

The K=2 pipeline has two independent stages:

1. Component-weight estimation.  ``pihat`` comes from the Direct (plug-in base
   bank) or Path-Space (path-space importance sampling) estimator, each with a
   fixed protocol: ``3 x 4000 @ 500`` uniform steps for Direct, ``4096
   trajectories x 400 steps x 3 estimator seeds x 2 atoms`` for Path-Space,
   both at ``L = 0``.  This runner does not re-run them; it reads the stored
   ``weights/*.json`` of the controlled study and its per-pair
   ``pg_total_variance``.  The path-space likelihood ratio is only defined for
   the corrector-free chain and is not used here.
2. Posterior component sampling.  Once a walker has been assigned an atom, the
   trajectory is an ordinary reverse-SDE run whose score is that atom's exactly
   transported score.  The single-factor study already uses that score inside
   every Langevin corrector step, so the shared Langevin-corrected reverse
   sampler applies unchanged.

So component weights are estimated once with the fixed Direct or Path-Space
protocol, and posterior component trajectories are then sampled with the shared
Langevin-corrected reverse sampler.  The Path-Space estimator itself does not
use Langevin.

Everything else is paired element-wise with the controlled ``(100, 0)`` rows:
same terminal-draw key, same atom-label key and therefore the same labels, same
``log_pi``, same PriorGuide ``Sigma_post``.  Only the sampler configuration
changes.

    python scripts/sample_mixture.py --task two_moons --model-id 0 \\
        --checkpoint checkpoints/two_moons/model_0.pkl \\
        --weights-root results/mixture_controlled --config 25:8 \\
        --prior-ids 0-9 --obs-seeds 1000000,1000001,1000002 \\
        --out results/mixture_practical
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, e2e, hooks  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra import weights as weights_mod  # noqa: E402
from spectra.utils import file_hash, git_state, parse_configs, parse_ids  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
K = 2  # this runner is the two-component lane; main() re-asserts it on the atoms
ROWS = ("pg", "eamt_ref_direct", "eamt_ref_pathspace", "eamt_ref_reference")
DEFAULT_ROWS = ("pg", "eamt_ref_direct", "eamt_ref_pathspace")
# row name -> the stored weight file prefix it reads (None: not weight-driven)
WEIGHT_FILE = {"eamt_ref_direct": "direct", "eamt_ref_pathspace": "pathspace"}


def load_frozen(weights_root: Path, task: str, pid: int, obs: int, model_id: str) -> dict:
    """The controlled study's stored weights and Sigma_post for one (task, model, pair)."""
    out = {"weights": {}, "problems": []}
    for row, prefix in WEIGHT_FILE.items():
        f = weights_root / "weights" / f"{prefix}_{task}_p{pid}_o{obs}_model{model_id}.json"
        if not f.is_file():
            out["problems"].append(f"{row}: missing {f}")
            continue
        blob = json.loads(f.read_text())
        # Check the status and the numbers: a failed estimate writes finite equal
        # logits, which read as an ordinary uniform component assignment.
        bad = weights_mod.weight_record_problems(blob, expected_k=K)
        if bad:
            out["problems"].append(f"{row}: " + "; ".join(bad))
            continue
        out["weights"][row] = {
            "source": prefix, "path": str(f), "sha256_16": file_hash(f),
            "log_pi": [float(v) for v in blob["log_pi"]],
            "pi": [float(v) for v in blob["pi"]],
            "log_odds": blob.get("log_odds"), "delta_estimate": blob.get("delta_estimate"),
            "budget": blob.get("budget"),
            "weight_error_vs_reference": blob.get("weight_error_vs_reference"),
        }
    status_f = (weights_root / "status" / task / f"model_{model_id}"
                / f"mixture_{pid}_obs_{obs}.json")
    if not status_f.is_file():
        out["problems"].append(f"missing controlled-run pair status {status_f}")
        return out
    status = json.loads(status_f.read_text())
    out["pg_total_variance"] = status.get("pg_total_variance")
    out["round10_status_path"] = str(status_f)
    out["round10_status_sha256_16"] = file_hash(status_f)
    out["round10_complete"] = bool(status.get("complete"))
    ref = status.get("reference_weights")
    if ref is not None:
        bad = weights_mod.weight_record_problems(ref, expected_k=K)
        if bad:
            out["problems"].append("eamt_ref_reference: " + "; ".join(bad))
            ref = None
    if ref is not None:
        out["reference_weights"] = ref
        out["weights"]["eamt_ref_reference"] = {
            "source": "reference", "path": ref.get("path"), "sha256_16": ref.get("sha256_16"),
            "log_pi": [float(v) for v in ref["log_pi"]], "pi": ref["pi"],
        }
    if out.get("pg_total_variance") is None:
        out["problems"].append("controlled-run status carries no pg_total_variance")
    return out


def expected_outputs(out_root: Path, task: str, model_id: str, pid: int, obs: int,
                     rows_want, sampler_seeds, tag: str):
    """Every sample file a finished pair must have, one per (row, sampler seed).

    A pair is complete only when all of these exist.  Recording ``complete``
    without checking them lets a run that skipped a method for a missing stored
    input mark itself done, after which ``--skip-existing`` would not revisit it.
    """
    def files(name, seed):
        d = (out_root / "samples" / task / f"{name}_ms{seed}" / f"model_{model_id}"
             / f"mixture_{pid}" / f"obs_{obs}")
        # Both files are required: a tree with JSONs but no samples has nothing
        # to score and does not count as finished.
        return (d / f"{tag}.npz", d / f"{tag}.json")

    return {(name, seed): files(name, seed)
            for name in rows_want for seed in sampler_seeds}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--weights-root", required=True,
                   help="controlled-study root holding weights/ and status/")
    p.add_argument("--prior-ids", default="0-9")
    p.add_argument("--obs-seeds", default="1000000,1000001,1000002")
    p.add_argument("--rows", default=",".join(DEFAULT_ROWS))
    p.add_argument("--sampler-seeds", default="0,1,2")
    p.add_argument("--config", default="25:8", help="single num_steps:langevin_steps")
    p.add_argument("--langevin-ratio", type=float, default=0.5)
    p.add_argument("--grid", default="power", choices=e2e.GRIDS)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows_want = [r for r in args.rows.split(",") if r]
    bad = set(rows_want) - set(ROWS)
    if bad:
        p.error(f"unknown rows {sorted(bad)}; known {ROWS}")
    configs = parse_configs(args.config)
    if len(configs) != 1:
        p.error("--config takes exactly one num_steps:langevin_steps")
    num_steps, lang_steps = configs[0]
    prior_ids = parse_ids(args.prior_ids)
    obs_seeds = parse_ids(args.obs_seeds)
    sampler_seeds = parse_ids(args.sampler_seeds)
    task = benchmark.get_task(args.task)
    out_root = Path(args.out)
    weights_root = Path(args.weights_root)

    nn = upstream_nn()
    ckpt_hash = file_hash(args.checkpoint)
    ckpt = load_checkpoint(args.checkpoint)
    if ckpt.theta_dim != task.theta_dim or ckpt.x_dim != task.x_dim:
        p.error(f"checkpoint is ({ckpt.theta_dim}, {ckpt.x_dim}) but {args.task} is "
                f"({task.theta_dim}, {task.x_dim})")
    backbone = Backbone(ckpt, mode="conditional", nn=nn)
    cfg = e2e.SamplerConfig(num_steps=num_steps, grid=args.grid, langevin_steps=lang_steps,
                            langevin_ratio=args.langevin_ratio)
    tag = f"steps{num_steps}_L{lang_steps}_{args.grid}"
    mean, std = backbone.terminal_moments()
    (out_root / "status" / args.task / f"model_{args.model_id}").mkdir(parents=True, exist_ok=True)

    common = {
        "round": "round11_k2_corrector", "task": args.task, "theta_dim": task.theta_dim,
        "x_dim": task.x_dim, "model_id": args.model_id,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256_16": ckpt_hash,
        "prior_type": "mixture", "upstream_commit": UPSTREAM_COMMIT,
        "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
        "schedule": {"sigma_min": ckpt.schedule.sigma_min, "sigma_max": ckpt.schedule.sigma_max,
                     "t_min": ckpt.schedule.t_min, "t_max": ckpt.schedule.t_max},
        "num_samples": args.num_samples, "sampler": cfg.as_dict(),
        "langevin_semantics": "delta = eta g(t)^2 dt / 2 (released PriorGuide corrector); "
                              "see e2e.langevin_step_size",
        "weight_stage": "component weights are read from the Direct / Path-Space estimates "
                        "under weights_root and not re-estimated; the corrector acts only on "
                        "the posterior component trajectories.",
        "weights_root": str(weights_root.resolve()),
        "python": sys.version.split()[0], "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
    }

    incomplete = []
    for pid in prior_ids:
        for obs in obs_seeds:
            status_path = (out_root / "status" / args.task / f"model_{args.model_id}"
                           / f"mixture_{pid}_obs_{obs}.json")
            want = expected_outputs(out_root, args.task, args.model_id, pid, obs,
                                    rows_want, sampler_seeds, tag)
            if args.skip_existing and status_path.is_file():
                recorded = json.loads(status_path.read_text()).get("complete")
                missing = [k for k, fs in want.items()
                           if not all(f.is_file() for f in fs)]
                if recorded and not missing:
                    print(f"[p{pid} o{obs}] skip (complete)", flush=True)
                    continue
                if recorded and missing:
                    # The status claims completion but outputs are missing;
                    # redo the pair.
                    print(f"[p{pid} o{obs}] recorded complete but {len(missing)} "
                          f"output(s) missing -- rerunning", flush=True)
            t_pair = time.perf_counter()
            status = {"kind": "round11_k2_corrector_pair", **common, "prior_id": pid,
                      "obs_seed": obs, "rows": {}, "complete": False, "errors": []}
            try:
                frozen = load_frozen(weights_root, args.task, pid, obs, args.model_id)
                status["frozen"] = {k: v for k, v in frozen.items() if k != "weights"}
                status["frozen_weights"] = frozen["weights"]
                shift = build_prior_shift(args.task, "mixture", pid)
                atoms = atoms_mod.atoms_from_shift(shift)
                if atoms.num_atoms != 2:
                    raise RuntimeError(f"K={atoms.num_atoms}; this lane is K=2")
                theta_true, x_o = benchmark.load_observation(
                    args.task, "mixture", pid, obs)

                built = {}
                if "pg" in rows_want:
                    tv = frozen.get("pg_total_variance")
                    if tv is None:
                        status["rows"]["pg"] = {"status": "skipped_no_frozen_sigma_post"}
                    else:
                        built["pg"] = hooks.make_pg_hook(
                            backbone, x_o, shift.components,
                            reverse_cov=hooks.upstream_reverse_cov(float(tv), shift.theta_dim))
                for name in ("eamt_ref_direct", "eamt_ref_pathspace", "eamt_ref_reference"):
                    if name in rows_want and name not in frozen["weights"]:
                        status["rows"][name] = {"status": "skipped_no_frozen_weights"}

                for seed in sampler_seeds:
                    k_init, k_noise = jax.random.split(
                        jax.random.PRNGKey((seed * 1_000_003 + obs) % (2**31)))
                    x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
                    diff, lang = e2e.make_noise(k_noise, num_steps, args.num_samples,
                                                task.theta_dim, lang_steps)
                    per_seed = dict(built)
                    labels_by = {}
                    for name in ("eamt_ref_direct", "eamt_ref_pathspace", "eamt_ref_reference"):
                        if name not in rows_want or name not in frozen["weights"]:
                            continue
                        log_pi = frozen["weights"][name]["log_pi"]
                        # same key and same log_pi as the controlled run -> same labels
                        labels = hooks.draw_atom_labels(jax.random.PRNGKey(7 + seed),
                                                        jnp.asarray(log_pi), args.num_samples)
                        labels_by[name] = np.asarray(labels)
                        per_seed[name] = hooks.make_spectra_hook(backbone, x_o, atoms, labels)
                    for name, (hook, ops, hook_meta) in per_seed.items():
                        row_name = f"{name}_ms{seed}"
                        out = (out_root / "samples" / args.task / row_name
                               / f"model_{args.model_id}" / f"mixture_{pid}" / f"obs_{obs}")
                        if args.skip_existing and (out / f"{tag}.json").is_file() \
                                and (out / f"{tag}.npz").is_file():
                            status["rows"][row_name] = {"status": "existing"}
                            continue
                        t1 = time.perf_counter()
                        res = e2e.reverse_sample(
                            e2e.instrument_hook(hook, ops), schedule=backbone.schedule, cfg=cfg,
                            x_init=x_init, diffusion_noise=diff, langevin_noise=lang,
                            snapshot_fracs=())
                        wall = time.perf_counter() - t1
                        nfe_static = e2e.nfe_accounting(cfg, ops)
                        nfe_runtime = e2e.nfe_from_aux(res.step_aux)
                        out.mkdir(parents=True, exist_ok=True)
                        payload = {"samples": res.samples, "x_o": np.asarray(x_o),
                                   "theta_true": np.asarray(theta_true), "times": res.times,
                                   **{f"aux_{k}": v for k, v in res.step_aux.items()
                                      if not k.startswith(e2e.NFE_PREFIX)}}
                        if name in labels_by:
                            payload["atom_labels"] = labels_by[name]
                        np.savez_compressed(out / f"{tag}.npz", **payload)
                        man = {**common, "prior_id": pid, "obs_seed": obs, "method": row_name,
                               "method_family": name, "method_meta": hook_meta,
                               "conditioning": "conditional", "sampler_seed": seed,
                               "nfe": nfe_static, "nfe_runtime": nfe_runtime,
                               "nfe_runtime_equals_static": nfe_runtime == nfe_static,
                               "wall_clock_s": wall, "wall_clock_includes_compile": True,
                               "wall_clock_per_1000_samples_s": wall * 1000.0 / args.num_samples,
                               "nan_or_inf": int((~np.isfinite(res.samples)).sum()),
                               "pg_total_variance": (frozen.get("pg_total_variance")
                                                     if name == "pg" else None)}
                        if name in frozen["weights"]:
                            man["eamt_weights"] = {**frozen["weights"][name],
                                                   "atom_label_seed": 7 + seed,
                                                   "reused_from_round10": True}
                            man["atoms"] = atoms.as_dict()
                        (out / f"{tag}.json").write_text(json.dumps(man, indent=2, default=str))
                        status["rows"][row_name] = {"status": "ok", "wall_clock_s": wall,
                                                    "nan_or_inf": man["nan_or_inf"]}
                        print(f"[p{pid} o{obs}][{row_name}] {wall:.1f}s "
                              f"nfe={nfe_runtime.get('hook_calls')} nan={man['nan_or_inf']}",
                              flush=True)
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                status["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
                print(f"[p{pid} o{obs}] FAILED: {exc!r}", flush=True)
            absent = sorted(f"{name}_ms{seed}" for (name, seed), fs in want.items()
                            if not all(f.is_file() for f in fs))
            status["missing_outputs"] = absent
            status["complete"] = not absent and not status["errors"]
            if not status["complete"]:
                incomplete.append(f"mixture_{pid}/obs_{obs}: "
                                  + (", ".join(absent) if absent else "errors"))
            status["pair_wall_clock_s"] = time.perf_counter() - t_pair
            status_path.write_text(json.dumps(status, indent=2, default=str))
            print(f"[p{pid} o{obs}] pair done in {status['pair_wall_clock_s']:.1f}s"
                  + (f" INCOMPLETE ({len(absent)} missing)" if absent else ""), flush=True)
    if incomplete:
        print("incomplete pairs:\n  " + "\n  ".join(incomplete), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
