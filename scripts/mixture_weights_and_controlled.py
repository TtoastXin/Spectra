#!/usr/bin/env python
"""The systematic two-component mixture benchmark, batched per (task, model).

One process loads one checkpoint once and walks a list of fixed
prior--observation pairs; for every pair it runs the whole practical pipeline:

1. Spectra-Direct normaliser: 3 base banks x 4000 draws @ 500 uniform steps
   (the fixed budget), plug-in ``log Z_k`` per bank, Delta averaged over banks
   (``weights.direct_evidence``, ``plugin_base``).
2. Spectra-PS normaliser: path-space chains, 4096 trajectories x 400 steps x
   3 estimator seeds x 2 atoms, ``weights.pathspace_evidence`` on the learned
   score.
3. PriorGuide's scalar ``Sigma_post`` bank (5000 x 500 uniform, upstream protocol).
4. SIR/SNIS on a 20k base bank at the sampler's own step budget.
5. The paired reverse-SDE rows, three sampler seeds each: ``base``, ``pg``
   (exact analytic ratio), ``a_full``, ``eamt_ref_direct``, ``eamt_ref_pathspace``
   and, when certified reference weights exist, ``eamt_ref_reference``
   (appendix diagnostic only).  Same terminal draw, grid, Brownian increments;
   ``L = 0`` because the path-space weight is only defined for the
   corrector-free chain.

Outputs land in the standard layout (``samples/<task>/<row>_ms<seed>/model_<m>/
mixture_<pid>/obs_<seed>/steps100_L0_power.{npz,json}``) so
``scripts/compute_metrics.py`` consumes them unchanged, plus ``weights/*.json`` in the reference schema and one ``status/`` JSON per
pair with stage timings, every estimator diagnostic (ESS, log-odds, |alpha_hat -
alpha*|, raw and unclipped) and the reference geometry covariates.  A pair that
fails is recorded with its traceback and the loop goes on.

Hooks whose inputs do not change across sampler seeds are built once per pair;
per-row wall clocks therefore include compilation only where a hook is new.
The paper's timing comes from ``scripts/timing_mixture.py``.

    python scripts/mixture_weights_and_controlled.py --task two_moons --model-id 0 \\
        --checkpoint checkpoints/two_moons/model_0.pkl \\
        --prior-ids 0-9 --obs-seeds 1000000,1000001,1000002 \\
        --ref-roots results/mixture_controlled --out results/mixture_controlled
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
from spectra.baselines import base_bank, sir_resample, snis_weights  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.utils import file_hash, git_state, parse_ids  # noqa: E402
from spectra.weights import direct_evidence, pathspace_evidence, weight_error  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROWS = ("base", "pg", "a_full", "eamt_ref_reference", "eamt_ref_direct",
        "eamt_ref_pathspace")
def find_reference_weights(ref_roots, task, pid, obs):
    for root in ref_roots:
        f = Path(root) / "references" / task / f"mixture_{pid}" / f"obs_{obs}_weights.json"
        if f.is_file():
            return f
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--prior-ids", default="0-9")
    p.add_argument("--obs-seeds", default="1000000,1000001,1000002")
    p.add_argument("--rows", default=",".join(ROWS))
    p.add_argument("--sampler-seeds", default="0,1,2")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--grid", default="power", choices=e2e.GRIDS)
    p.add_argument("--direct-n-per-seed", type=int, default=4000)
    p.add_argument("--direct-seeds", default="0,1,2")
    p.add_argument("--direct-steps", type=int, default=500)
    p.add_argument("--ps-steps", type=int, default=400)
    p.add_argument("--ps-trajectories", type=int, default=4096)
    p.add_argument("--ps-seeds", default="0,1,2")
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--sir-bank", type=int, default=20000, help="0 disables the SIR row")
    p.add_argument("--ref-roots", default="",
                   help="comma list of roots searched (in order) for reference weights")
    p.add_argument("--timing-repeat", type=int, default=1)
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows_want = [r for r in args.rows.split(",") if r]
    bad = set(rows_want) - set(ROWS)
    if bad:
        p.error(f"unknown rows {sorted(bad)}; known {ROWS}")
    prior_ids = parse_ids(args.prior_ids)
    obs_seeds = parse_ids(args.obs_seeds)
    sampler_seeds = parse_ids(args.sampler_seeds)
    direct_seeds = parse_ids(args.direct_seeds)
    ps_seeds = parse_ids(args.ps_seeds)
    ref_roots = [r for r in args.ref_roots.split(",") if r]
    task = benchmark.get_task(args.task)
    out_root = Path(args.out)
    (out_root / "weights").mkdir(parents=True, exist_ok=True)
    (out_root / "status" / args.task / f"model_{args.model_id}").mkdir(parents=True, exist_ok=True)

    nn = upstream_nn()
    ckpt_hash = file_hash(args.checkpoint)
    ckpt = load_checkpoint(args.checkpoint)
    if ckpt.theta_dim != task.theta_dim or ckpt.x_dim != task.x_dim:
        p.error(f"checkpoint is ({ckpt.theta_dim}, {ckpt.x_dim}) but {args.task} is "
                f"({task.theta_dim}, {task.x_dim})")
    backbone = Backbone(ckpt, mode="conditional", nn=nn)
    box = benchmark.training_support(args.task)
    cfg = e2e.SamplerConfig(num_steps=args.num_steps, grid=args.grid, langevin_steps=0)
    snaps = e2e.DEFAULT_SNAPSHOT_FRACS if args.snapshots else ()
    tag = f"steps{args.num_steps}_L0_{args.grid}"

    common = {
        "round": "round10_k2", "task": args.task, "theta_dim": task.theta_dim,
        "x_dim": task.x_dim, "model_id": args.model_id,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256_16": ckpt_hash,
        "prior_type": "mixture", "upstream_commit": UPSTREAM_COMMIT,
        "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
        "schedule": {"sigma_min": ckpt.schedule.sigma_min, "sigma_max": ckpt.schedule.sigma_max,
                     "t_min": ckpt.schedule.t_min, "t_max": ckpt.schedule.t_max},
        "num_samples": args.num_samples, "sampler": cfg.as_dict(),
        "python": sys.version.split()[0], "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
    }
    process_summary = {"kind": "round10_k2_process", **common, "pairs": [],
                       "started": time.time()}

    def write_row(row_name, pid, obs, samples, manifest, extra=None):
        out = (out_root / "samples" / args.task / row_name / f"model_{args.model_id}"
               / f"mixture_{pid}" / f"obs_{obs}")
        out.mkdir(parents=True, exist_ok=True)
        payload = {"samples": samples, "x_o": np.asarray(manifest.pop("_x_o")),
                   "theta_true": np.asarray(manifest.pop("_theta_true"))}
        payload.update(extra or {})
        np.savez_compressed(out / f"{tag}.npz", **payload)
        (out / f"{tag}.json").write_text(json.dumps(manifest, indent=2, default=str))
        return out / f"{tag}.npz"

    def expected_outputs(pid, obs):
        """Every sample file a finished pair must have.

        ``complete`` is derived from these rather than asserted at the end of
        the loop: a pair that skipped a method (no weights, degenerate SIR) or
        raised would otherwise mark itself done, after which ``--skip-existing``
        would not revisit it.  The SIR row is written once per pair; the sampler
        rows once per (row, sampler seed).  Each entry is the pair of files a
        finished row writes (the samples and their manifest), because a
        manifest on its own leaves nothing to score.

        Recovery here is per pair, not per row: an incomplete pair re-runs its
        weight estimation as well.  ``scripts/sample_mixture.py``, whose rows are
        independent of one another, resumes row by row instead.
        """
        base = (out_root / "samples" / args.task)

        def files(row):
            d = (base / row / f"model_{args.model_id}" / f"mixture_{pid}" / f"obs_{obs}")
            # Both: a manifest without its samples is not a finished row.
            return (d / f"{tag}.npz", d / f"{tag}.json")

        want = {f"{name}_ms{seed}": files(f"{name}_ms{seed}")
                for name in rows_want for seed in sampler_seeds}
        if args.sir_bank > 0:
            want[f"sir_n{args.sir_bank}"] = files(f"sir_n{args.sir_bank}")
        return want

    incomplete = []
    for pid in prior_ids:
        for obs in obs_seeds:
            status_path = (out_root / "status" / args.task / f"model_{args.model_id}"
                           / f"mixture_{pid}_obs_{obs}.json")
            want = expected_outputs(pid, obs)
            if args.skip_existing and status_path.is_file():
                prev = json.loads(status_path.read_text())
                missing = [k for k, fs in want.items()
                           if not all(f.is_file() for f in fs)]
                if prev.get("complete") and not missing:
                    print(f"[p{pid} o{obs}] skip (complete)", flush=True)
                    continue
                if prev.get("complete") and missing:
                    # The whole pair is redone, weight estimation included: the
                    # stages are not individually resumable here.  The estimators
                    # are deterministic given their seeds, so the rewritten files
                    # match what was there; the cost is the wasted compute.
                    print(f"[p{pid} o{obs}] recorded complete but {len(missing)} "
                          f"output(s) missing -- rerunning the whole pair",
                          flush=True)
            t_pair = time.perf_counter()
            status = {"kind": "round10_k2_pair", **common, "prior_id": pid, "obs_seed": obs,
                      "stages": {}, "rows": {}, "complete": False, "errors": []}
            try:
                shift = build_prior_shift(args.task, "mixture", pid)
                atoms = atoms_mod.atoms_from_shift(shift)
                if atoms.num_atoms != 2:
                    raise RuntimeError(f"K={atoms.num_atoms}; this lane is K=2")
                theta_true, x_o = benchmark.load_observation(
                    args.task, "mixture", pid, obs)
                geom = atoms_mod.ladder_geometry(atoms, 0, 1)
                status["geometry"] = {
                    "lambda": geom["lambda"], "atom_sep": geom["atom_sep"],
                    "kl_atom": geom["kl_atom"], "theta_dim": task.theta_dim,
                    "prior_pi": np.asarray(shift.target_pi, float).tolist(),
                    "log_b": np.asarray(atoms.log_b, float).tolist(),
                    "prior_mass_outside_training_box": (
                        shift.target_mass_outside_training_box()
                        if shift.train_box is not None else None),
                }
                status["prior_json_sha256_16"] = benchmark.prior_json_hash(
                    args.task, "mixture", pid)

                # ---- reference weights (diagnostic only) --------------------
                ref_file = find_reference_weights(ref_roots, args.task, pid, obs)
                pi_star = None
                if ref_file is not None:
                    blob = json.loads(ref_file.read_text())
                    pi_star = np.asarray(blob["pi"], float)
                    status["reference_weights"] = {
                        "path": str(ref_file), "sha256_16": file_hash(ref_file),
                        "source": blob.get("source"), "pi": pi_star.tolist(),
                        "log_pi": [float(v) for v in blob["log_pi"]],
                        "alpha_min": float(pi_star.min()),
                        "entropy_nats": float(-np.sum(pi_star[pi_star > 0]
                                                      * np.log(pi_star[pi_star > 0]))),
                        "stratum": ("substantive_mixture" if pi_star.min() >= 0.05
                                    else "dominant_component"),
                    }
                else:
                    status["reference_weights"] = None

                # ---- Direct ------------------------------------------------
                t0 = time.perf_counter()
                direct, pooled = direct_evidence(backbone, x_o, atoms, direct_seeds,
                                                 args.direct_n_per_seed, args.direct_steps)
                status["stages"]["direct_s"] = time.perf_counter() - t0
                direct["weight_error_vs_reference"] = weight_error(direct["pi"], pi_star)
                direct.update({"task": args.task, "prior_type": "mixture", "prior_id": pid,
                               "obs_seed": obs, "model_id": args.model_id,
                               "delta_star": (float(np.log(pi_star[0]) - np.log(pi_star[1])
                                                    - direct["log_b_diff"])
                                              if pi_star is not None and (pi_star > 0).all()
                                              else float("nan")),
                               "repo_git": common["repo_git"]})
                wpath = out_root / "weights" / f"direct_{args.task}_p{pid}_o{obs}_model{args.model_id}.json"
                wpath.write_text(json.dumps(direct, indent=2, default=str))
                status["direct"] = {k: v for k, v in direct.items() if k != "per_seed"}
                status["direct"]["per_seed"] = direct["per_seed"]
                bank_dir = (out_root / "benchmark_evidence" / args.task / f"model_{args.model_id}")
                bank_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(bank_dir / f"direct_bank_p{pid}_o{obs}.npz",
                                    bank=pooled.astype(np.float32), x_o=np.asarray(x_o))
                print(f"[p{pid} o{obs}] direct: Delta={direct['delta_estimate']:+.4f} "
                      f"pi={np.round(direct['pi'], 4).tolist()} ess_min={direct['ess_min_over_seeds']:.1f} "
                      f"({status['stages']['direct_s']:.1f}s)", flush=True)

                # ---- Path-space --------------------------------------------
                t0 = time.perf_counter()
                ps = pathspace_evidence(backbone, x_o, atoms, ps_seeds, args.ps_trajectories,
                                        args.ps_steps, args.grid)
                status["stages"]["pathspace_s"] = time.perf_counter() - t0
                ps["weight_error_vs_reference"] = weight_error(ps["pi"], pi_star)
                ps.update({"task": args.task, "prior_type": "mixture", "prior_id": pid,
                           "obs_seed": obs, "model_id": args.model_id,
                           "delta_star": direct["delta_star"],
                           "repo_git": common["repo_git"]})
                wpath = out_root / "weights" / f"pathspace_{args.task}_p{pid}_o{obs}_model{args.model_id}.json"
                wpath.write_text(json.dumps(ps, indent=2, default=str))
                status["pathspace"] = ps
                print(f"[p{pid} o{obs}] pathspace: Delta={ps['delta_estimate']:+.4f} "
                      f"pi={np.round(ps['pi'], 4).tolist()} ess_norm_min={ps['ess_norm_min_over_seeds']:.2e} "
                      f"({status['stages']['pathspace_s']:.1f}s)", flush=True)

                # ---- PriorGuide's scalar Sigma_post bank ---------------------
                t0 = time.perf_counter()
                pg_bank, pg_bank_wall = base_bank(
                    backbone, x_o, jax.random.PRNGKey(90000), args.pg_var_samples,
                    e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform", langevin_steps=0))
                total_variance = float(np.var(pg_bank))
                status["stages"]["pg_bank_s"] = time.perf_counter() - t0
                status["pg_total_variance"] = total_variance

                # ---- SIR / SNIS ------------------------------------------------
                if args.sir_bank > 0:
                    t0 = time.perf_counter()
                    n_bank = args.sir_bank
                    bank, bank_wall = base_bank(
                        backbone, x_o,
                        jax.random.PRNGKey((7_000_000 + obs * 97 + n_bank) % (2**31)),
                        n_bank, cfg)
                    w = snis_weights(shift, bank, box)
                    row_name = f"sir_n{n_bank}"
                    if w["degenerate"]:
                        status["rows"][row_name] = {"status": "degenerate_zero_weights"}
                    else:
                        rng = np.random.default_rng(4_000_000 + obs + n_bank)
                        sir_samples, idx = sir_resample(rng, bank, w["w"], args.num_samples)
                        man = {**common, "_x_o": x_o, "_theta_true": theta_true,
                               "prior_id": pid, "obs_seed": obs, "method": row_name,
                               "method_family": "sir", "conditioning": "conditional",
                               "method_meta": {"method": "sir", "bank_size": n_bank,
                                               "resampling": "with replacement",
                                               "weights": "w = r(theta) self-normalised"},
                               "sir": {"bank_size": n_bank, "ess": w["ess"], "ess_frac": w["ess_frac"],
                                       "max_normalised_weight": w["max_weight"],
                                       "top1pct_weight_mass": w["top10_weight_mass"],
                                       "log_weight_range": w["log_w_range"],
                                       "bank_points_outside_training_box": w["outside_box"],
                                       "unique_resampled": int(np.unique(idx).size)},
                               "nfe": {"base_score_calls": n_bank * (cfg.num_steps - 1),
                                       "denoiser_jacobian_calls": 0,
                                       "hook_calls": cfg.num_steps - 1},
                               "cost": {"bank_wall_clock_s": bank_wall, "first_use_total_s": bank_wall,
                                        "online_per_1000_samples_s": 0.0},
                               "wall_clock_s": bank_wall,
                               "wall_clock_per_1000_samples_s": bank_wall * 1000.0 / n_bank,
                               "nan_or_inf": int((~np.isfinite(sir_samples)).sum())}
                        write_row(row_name, pid, obs, sir_samples, man,
                                  {"log_w": w["log_w"], "resample_idx": idx})
                        status["rows"][row_name] = {"status": "ok", "ess": w["ess"],
                                                    "ess_frac": w["ess_frac"],
                                                    "unique_resampled": int(np.unique(idx).size),
                                                    "wall_clock_s": bank_wall}
                    status["stages"]["sir_s"] = time.perf_counter() - t0

                # ---- paired reverse-SDE rows --------------------------------
                t0 = time.perf_counter()
                mean, std = backbone.terminal_moments()
                static_hooks = {}
                if "base" in rows_want:
                    static_hooks["base"] = hooks.make_base_hook(backbone, x_o)
                if "pg" in rows_want:
                    static_hooks["pg"] = hooks.make_pg_hook(
                        backbone, x_o, shift.components,
                        reverse_cov=hooks.upstream_reverse_cov(total_variance, shift.theta_dim))
                if "a_full" in rows_want:
                    static_hooks["a_full"] = hooks.make_a_full_hook(backbone, x_o, shift.components)
                weight_sources = {}
                if "eamt_ref_direct" in rows_want and direct["status"] == "ok":
                    weight_sources["eamt_ref_direct"] = ("direct", direct["log_pi"])
                if "eamt_ref_pathspace" in rows_want and ps["status"] == "ok":
                    weight_sources["eamt_ref_pathspace"] = ("pathspace", ps["log_pi"])
                if "eamt_ref_reference" in rows_want and pi_star is not None:
                    weight_sources["eamt_ref_reference"] = ("reference",
                                                            status["reference_weights"]["log_pi"])
                for name in ("eamt_ref_direct", "eamt_ref_pathspace", "eamt_ref_reference"):
                    if name in rows_want and name not in weight_sources:
                        status["rows"][name] = {"status": "skipped_no_weights"}

                for seed in sampler_seeds:
                    k_init, k_noise = jax.random.split(
                        jax.random.PRNGKey((seed * 1_000_003 + obs) % (2**31)))
                    x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
                    diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples,
                                                task.theta_dim, 0)
                    built = dict(static_hooks)
                    labels_by = {}
                    for name, (src, log_pi) in weight_sources.items():
                        labels = hooks.draw_atom_labels(jax.random.PRNGKey(7 + seed),
                                                        jnp.asarray(log_pi), args.num_samples)
                        labels_by[name] = np.asarray(labels)
                        built[name] = hooks.make_spectra_hook(backbone, x_o, atoms, labels)
                    for name, (hook, ops, hook_meta) in built.items():
                        walls = []
                        for _ in range(max(args.timing_repeat, 1)):
                            t1 = time.perf_counter()
                            res = e2e.reverse_sample(
                                hook, schedule=backbone.schedule, cfg=cfg, x_init=x_init,
                                diffusion_noise=diff, langevin_noise=lang, snapshot_fracs=snaps)
                            walls.append(time.perf_counter() - t1)
                        wall = walls[-1]
                        row_name = f"{name}_ms{seed}"
                        man = {**common, "_x_o": x_o, "_theta_true": theta_true,
                               "prior_id": pid, "obs_seed": obs, "method": row_name,
                               "method_family": name, "method_meta": hook_meta,
                               "conditioning": "conditional", "sampler_seed": seed,
                               "nfe": e2e.nfe_accounting(cfg, ops),
                               "wall_clock_s": wall, "wall_clock_all_passes_s": walls,
                               "wall_clock_includes_compile": len(walls) == 1,
                               "wall_clock_per_1000_samples_s": wall * 1000.0 / args.num_samples,
                               "nan_or_inf": int((~np.isfinite(res.samples)).sum()),
                               "pg_total_variance": total_variance if name == "pg" else None}
                        extra = {"times": res.times,
                                 **{f"aux_{k}": v for k, v in res.step_aux.items()}}
                        if name in weight_sources:
                            src, log_pi = weight_sources[name]
                            man["eamt_weights"] = {"source": src, "log_pi": list(map(float, log_pi)),
                                                   "atom_label_seed": 7 + seed}
                            man["atoms"] = atoms.as_dict()
                            extra["atom_labels"] = labels_by[name]
                        path = write_row(row_name, pid, obs, res.samples, man, extra)
                        status["rows"][row_name] = {"status": "ok", "wall_clock_s": wall,
                                                    "nan_or_inf": man["nan_or_inf"]}
                        print(f"[p{pid} o{obs}][{row_name}] {wall:.1f}s nan={man['nan_or_inf']}",
                              flush=True)
                status["stages"]["rows_s"] = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                status["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
                print(f"[p{pid} o{obs}] FAILED: {exc!r}", flush=True)
            absent = sorted(k for k, fs in want.items()
                            if not all(f.is_file() for f in fs))
            status["missing_outputs"] = absent
            status["complete"] = not absent and not status["errors"]
            if not status["complete"]:
                incomplete.append(f"mixture_{pid}/obs_{obs}: "
                                  + (", ".join(absent) if absent else "errors"))
            status["pair_wall_clock_s"] = time.perf_counter() - t_pair
            status_path.write_text(json.dumps(status, indent=2, default=str))
            process_summary["pairs"].append({
                "prior_id": pid, "obs_seed": obs, "complete": status["complete"],
                "pair_wall_clock_s": status["pair_wall_clock_s"], "stages": status["stages"],
                "n_errors": len(status["errors"])})
            print(f"[p{pid} o{obs}] pair done in {status['pair_wall_clock_s']:.1f}s "
                  f"stages={ {k: round(v, 1) for k, v in status['stages'].items()} }", flush=True)
            process_summary["elapsed_s"] = time.time() - process_summary["started"]
            (out_root / "status" / args.task / f"model_{args.model_id}" / "process_summary.json"
             ).write_text(json.dumps(process_summary, indent=2, default=str))
    if incomplete:
        print("incomplete pairs:\n  " + "\n  ".join(incomplete), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
