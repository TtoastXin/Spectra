#!/usr/bin/env python
"""Single-factor exact transport, one cell, five methods.

With ``K = 1``, a single exponential-quadratic prior-ratio factor needs no
component-weight estimation, so Spectra here is only the transformed base-score
query: no atom labels, no evidence, no normaliser.  The error sources of the
mixture branch (bank plug-in ESS, moment-matched evidence, path-space weights)
do not arise.

Rows
----
``base``       unguided Simformer (no adaptation).
``sir_n<N>``   clean-space SNIS/SIR on a fixed base bank, ``w = r(theta)``,
               reported at several bank budgets to show how the weight
               distribution degrades.
``pg_exact``   PriorGuide's reverse-Gaussian closure fed the exact analytic
               ratio, which removes the ratio-fit error from the comparison.
``a_full``     PG-FullCov: full-Jacobian Gaussian guidance, which does not need
               a global normaliser.
``spectra``    exact transport, one transformed score query, no Jacobians.

The four sampler rows are paired element-wise: same terminal draw, same time
grid, same Brownian increments, so only the target score differs (Protocol A).

SIR is not paired with them because it is not a reverse-SDE method.  It uses its
own bank and its own measured cost, and the bank generation is counted in its
first-use total.

Samples land in the standard layout so ``scripts/compute_metrics.py`` scores them
against ``<out>/references/<task>/<prior>/obs_<seed>.npz`` with no special case.

    python scripts/sample_single_factor_baselines.py --task slcp \\
        --checkpoint checkpoints/slcp/model_0.pkl --model-id 0 \\
        --prior-type strong --prior-id 0 \\
        --obs-seeds 1000000,1000001,1000002 --sampler-seeds 0,1,2 \\
        --sir-budgets 1000,5000,20000 --out results/single_factor_controlled_strong
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark, e2e, hooks  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.baselines import draw_base_bank, sir_resample, snis_weights  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.transport import effective_noise_report  # noqa: E402
from spectra.utils import file_hash, git_state, peak_gpu_memory_mb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Sampler rows, in the order they are reported.  ``sir`` is handled separately
# because it is a reweighting of a bank, not a hook in the reverse driver.
SAMPLER_ROWS = ("base", "pg_exact", "a_full", "spectra")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--prior-type", default="strong",
                   choices=benchmark.PRIOR_TYPES)
    p.add_argument("--prior-id", type=int, default=0)
    p.add_argument("--obs-seeds", default="1000000,1000001,1000002")
    p.add_argument("--sampler-seeds", default="0,1,2")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--grid", default="power", choices=e2e.GRIDS)
    p.add_argument("--sir-budgets", default="1000,5000,20000")
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--timing-repeat", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    obs_seeds = [int(v) for v in args.obs_seeds.split(",")]
    seeds = [int(v) for v in args.sampler_seeds.split(",")]
    budgets = [int(v) for v in args.sir_budgets.split(",") if v.strip()]
    task = benchmark.get_task(args.task)
    out_root = Path(args.out)

    shift = build_prior_shift(args.task, args.prior_type, args.prior_id)
    if shift.expquad is None:
        p.error(
            f"{args.task}/{args.prior_type}_{args.prior_id} is not a single "
            "exponential-quadratic factor; this runner is K=1 only")
    eq = shift.expquad
    box = benchmark.training_support(args.task)

    nn = upstream_nn()
    ckpt_hash = file_hash(args.checkpoint)
    ckpt = load_checkpoint(args.checkpoint)
    if ckpt.theta_dim != task.theta_dim or ckpt.x_dim != task.x_dim:
        p.error(f"checkpoint is ({ckpt.theta_dim}, {ckpt.x_dim}) but "
                f"{args.task} is ({task.theta_dim}, {task.x_dim})")
    backbone = Backbone(ckpt, mode="conditional", nn=nn)

    cfg = e2e.SamplerConfig(num_steps=args.num_steps, grid=args.grid,
                            langevin_steps=0)
    common = {
        "round": "round8_k1",
        "task": args.task,
        "theta_dim": task.theta_dim,
        "x_dim": task.x_dim,
        "model_id": args.model_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256_16": ckpt_hash,
        "prior_type": args.prior_type,
        "prior_id": args.prior_id,
        "prior_json_sha256_16": benchmark.prior_json_hash(
            args.task, args.prior_type, args.prior_id),
        "training_prior_json_sha256_16": benchmark.prior_json_hash(
            args.task, "training", None),
        "num_atoms": int(shift.comp_log_weights.shape[0]),
        "transport_a": eq.a.tolist(),
        "transport_kappa": eq.kappa,
        "prior_mass_outside_training_box": (
            shift.target_mass_outside_training_box()
            if shift.train_box is not None else None),
        "upstream_commit": UPSTREAM_COMMIT,
        "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
        "schedule": {
            "sigma_min": ckpt.schedule.sigma_min,
            "sigma_max": ckpt.schedule.sigma_max,
            "t_min": ckpt.schedule.t_min, "t_max": ckpt.schedule.t_max,
            "scale_min": ckpt.schedule.scale_min,
        },
        "num_samples": args.num_samples,
        "sampler": cfg.as_dict(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "host": platform.node(),
    }

    def write_row(row_name, obs_seed, samples, manifest, extra_payload=None):
        out = (out_root / "samples" / args.task / row_name
               / f"model_{args.model_id}"
               / f"{args.prior_type}_{args.prior_id}" / f"obs_{obs_seed}")
        out.mkdir(parents=True, exist_ok=True)
        tag = f"steps{args.num_steps}_L0_{args.grid}"
        payload = {"samples": samples, "x_o": np.asarray(manifest["x_o_ref"]),
                   "theta_true": np.asarray(manifest["theta_true_ref"])}
        payload.update(extra_payload or {})
        np.savez_compressed(out / f"{tag}.npz", **payload)
        man = {k: v for k, v in manifest.items()
               if k not in ("x_o_ref", "theta_true_ref")}
        (out / f"{tag}.json").write_text(json.dumps(man, indent=2, default=str))
        return out / f"{tag}.npz"

    for obs_seed in obs_seeds:
        theta_true, x_o = benchmark.load_observation(
            args.task, args.prior_type, args.prior_id, obs_seed)
        ref = {"x_o_ref": np.asarray(x_o).tolist(),
               "theta_true_ref": np.asarray(theta_true).tolist()}

        # --- PriorGuide's own scalar Sigma_post, upstream's protocol ----------
        pg_bank, pg_bank_wall = draw_base_bank(
            backbone, jax.random.PRNGKey((90000 + obs_seed) % (2**31)),
            args.pg_var_samples, x_o,
            e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform",
                              langevin_steps=0))
        total_variance = float(np.var(pg_bank))
        pg_var_score_calls = args.pg_var_samples * (args.pg_var_steps - 1)
        print(f"[obs {obs_seed}] PG Sigma_post bank: {args.pg_var_samples} x "
              f"{args.pg_var_steps} steps in {pg_bank_wall:.1f}s "
              f"(total_var={total_variance:.5g})", flush=True)

        # --- clean-space SNIS / SIR at each bank budget -----------------------
        for n_bank in budgets:
            bank, bank_wall = draw_base_bank(
                backbone,
                jax.random.PRNGKey((7_000_000 + obs_seed * 97 + n_bank) % (2**31)),
                n_bank, x_o, cfg)
            w = snis_weights(shift, bank, box)
            row_name = f"sir_n{n_bank}"
            if w["degenerate"]:
                print(f"[obs {obs_seed}][{row_name}] DEGENERATE: every bank "
                      f"point has zero weight", flush=True)
                continue
            rng = np.random.default_rng(4_000_000 + obs_seed + n_bank)
            sir_samples, idx = sir_resample(rng, bank, w["w"], args.num_samples)
            snis_mean = (w["w"][:, None] * bank).sum(0)
            d = bank - snis_mean
            snis_cov = (w["w"][:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
            bank_score_calls = n_bank * (cfg.num_steps - 1)
            man = {
                **common, **ref, "obs_seed": obs_seed, "method": row_name,
                "method_family": "sir",
                "method_meta": {
                    "method": "sir", "bank_size": n_bank,
                    "resampling": "with replacement",
                    "weights": "w = r(theta) = pi_new/pi_train, self-normalised",
                },
                "sir": {
                    "bank_size": n_bank,
                    "ess": w["ess"], "ess_frac": w["ess_frac"],
                    "max_normalised_weight": w["max_weight"],
                    "top1pct_weight_mass": w["top10_weight_mass"],
                    "log_weight_range": w["log_w_range"],
                    "bank_points_outside_training_box": w["outside_box"],
                    "unique_resampled": int(np.unique(idx).size),
                    "unique_fraction": float(np.unique(idx).size / args.num_samples),
                    "snis_mean": snis_mean.tolist(),
                    "snis_cov_diag": np.diag(snis_cov).tolist(),
                },
                "nfe": {"base_score_calls": bank_score_calls,
                        "denoiser_jacobian_calls": 0,
                        "hook_calls": bank_score_calls},
                "cost": {
                    "bank_wall_clock_s": bank_wall,
                    "first_use_total_s": bank_wall,
                    "online_per_1000_samples_s": 0.0,
                    "note": "resampling is O(N) and negligible next to the bank "
                            "draw, so the cost is that of the bank draw",
                },
                "wall_clock_s": bank_wall,
                "wall_clock_per_1000_samples_s": bank_wall * 1000.0 / n_bank,
                "nan_or_inf": int((~np.isfinite(sir_samples)).sum()),
                "peak_gpu_memory_mb": peak_gpu_memory_mb(),
            }
            path = write_row(row_name, obs_seed, sir_samples, man,
                             {"bank": bank, "log_w": w["log_w"],
                              "weights": w["w"], "resample_idx": idx})
            print(f"[obs {obs_seed}][{row_name}] ESS={w['ess']:.1f} "
                  f"({100*w['ess_frac']:.3f}% of {n_bank})  "
                  f"maxw={w['max_weight']:.4f}  "
                  f"unique={np.unique(idx).size}/{args.num_samples}  "
                  f"bank {bank_wall:.1f}s -> {path}", flush=True)

        # --- the four paired reverse-SDE rows ---------------------------------
        for seed in seeds:
            k_init, k_noise = jax.random.split(
                jax.random.PRNGKey((seed * 1_000_003 + obs_seed) % (2**31)))
            mean, std = backbone.terminal_moments()
            x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
            diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples,
                                        task.theta_dim, 0)
            taus = np.asarray(
                backbone.schedule.sigma(e2e.time_grid(backbone.schedule, cfg))**2)

            built = {
                "base": hooks.make_base_hook(backbone, x_o),
                "pg_exact": hooks.make_pg_hook(
                    backbone, x_o, shift.components,
                    reverse_cov=hooks.upstream_reverse_cov(total_variance,
                                                           shift.theta_dim)),
                "a_full": hooks.make_a_full_hook(backbone, x_o, shift.components),
                "spectra": hooks.make_tq_hook(backbone, x_o, eq.a_jnp, eq.kappa),
            }
            for name in SAMPLER_ROWS:
                hook, ops, hook_meta = built[name]
                walls = []
                for _ in range(max(args.timing_repeat, 1)):
                    t0 = time.perf_counter()
                    res = e2e.reverse_sample(
                        hook, schedule=backbone.schedule, cfg=cfg,
                        x_init=x_init, diffusion_noise=diff, langevin_noise=lang,
                        snapshot_fracs=e2e.DEFAULT_SNAPSHOT_FRACS)
                    walls.append(time.perf_counter() - t0)
                wall = walls[-1]
                nfe = e2e.nfe_accounting(cfg, ops)
                row_name = f"{name}_ms{seed}"
                first_use = wall
                if name == "pg_exact":
                    first_use = wall + pg_bank_wall
                man = {
                    **common, **ref, "obs_seed": obs_seed, "method": row_name,
                    "method_family": name, "method_meta": hook_meta,
                    "conditioning": "conditional", "sampler_seed": seed,
                    "nfe": nfe,
                    "wall_clock_s": wall,
                    "wall_clock_first_pass_s": walls[0],
                    "wall_clock_all_passes_s": walls,
                    "compile_overhead_s": (walls[0] - wall) if len(walls) > 1 else None,
                    "wall_clock_includes_compile": len(walls) == 1,
                    "wall_clock_per_1000_samples_s": wall * 1000.0 / args.num_samples,
                    "cost": {
                        "online_per_1000_samples_s": wall * 1000.0 / args.num_samples,
                        "first_use_total_s": first_use,
                        "precompute_s": pg_bank_wall if name == "pg_exact" else 0.0,
                        "precompute_score_calls": (pg_var_score_calls
                                                   if name == "pg_exact" else 0),
                        "precompute_what": ("upstream scalar Sigma_post base draw"
                                            if name == "pg_exact" else None),
                    },
                    "nan_or_inf": int((~np.isfinite(res.samples)).sum()),
                    "peak_gpu_memory_mb": peak_gpu_memory_mb(),
                    "pg_total_variance": (total_variance if name == "pg_exact"
                                          else None),
                }
                if name == "spectra":
                    man["effective_noise"] = effective_noise_report(
                        taus, eq.kappa, backbone.schedule)
                extra = {"times": res.times,
                         **{f"aux_{k}": v for k, v in res.step_aux.items()},
                         **{f"snapshot_{str(f).replace('.', 'p')}": s["state"]
                            for f, s in res.snapshots.items()}}
                path = write_row(row_name, obs_seed, res.samples, man, extra)
                print(f"[obs {obs_seed}][{row_name}] {wall:.2f}s "
                      f"nan={man['nan_or_inf']} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
