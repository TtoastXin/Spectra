#!/usr/bin/env python
"""Single-factor (K=1) sampling under a list of ``(N, L)`` sampler configurations.

The controlled K=1 rows are Protocol A at ``L = 0``.  This runner keeps every
other element of Protocol A (same terminal draw, same power grid, same
Brownian increments, same Langevin increments) and runs the three guided
methods under each requested ``(num_steps, langevin_steps)``:

``pg_exact``  PriorGuide's reverse-Gaussian closure, exact analytic ratio,
              upstream's scalar ``Sigma_post`` (5000 x 500 uniform base draw);
``a_full``    PG-FullCov (ours);
``spectra``   exact transport, one transformed base-score query per hook call.
``base`` is optional (``--with-base``) and only meaningful at ``L = 0``.

Every run uses the runtime-instrumented hooks (``e2e.instrument_hook``), so the
manifest carries the NFE the traced program executed next to the static
accounting; ``tests/test_langevin.py`` checks that the two are equal and that
the samples match the uninstrumented driver.  Wall clocks here are single
passes and may include compilation; the paper's timing comes from
``scripts/timing_single_factor.py``.  Outputs use the standard layout with the tag
``steps<N>_L<L>_<grid>`` so ``compute_metrics.py`` scores them unchanged.

    python scripts/sample_single_factor.py --task two_moons \\
        --checkpoint checkpoints/two_moons/model_0.pkl --model-id 0 \\
        --prior-type strong --prior-id 0 --obs-seeds 1000000,1000001,1000002 \\
        --sampler-seeds 0,1,2 --configs 25:8,100:0 --out results/single_factor_practical
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark, e2e, hooks  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.baselines import draw_base_bank  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.transport import effective_noise_report  # noqa: E402
from spectra.utils import file_hash, git_state, parse_configs, peak_gpu_memory_mb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GUIDED = ("pg_exact", "a_full", "spectra")
KNOWN = ("pg_exact", "pg_vjp", "a_full", "spectra")
PG_FAMILIES = ("pg_exact", "pg_vjp")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--prior-type", default="strong", choices=benchmark.PRIOR_TYPES)
    p.add_argument("--prior-id", type=int, default=0)
    p.add_argument("--obs-seeds", default="1000000,1000001,1000002")
    p.add_argument("--sampler-seeds", default="0,1,2")
    p.add_argument("--configs", default="25:0,25:8,50:2,100:0,100:2,250:0",
                   help="comma list of num_steps:langevin_steps")
    p.add_argument("--langevin-ratio", type=float, default=0.5)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--grid", default="power", choices=e2e.GRIDS)
    p.add_argument("--methods", default=",".join(GUIDED))
    p.add_argument("--with-base", action="store_true")
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    bad = set(methods) - set(KNOWN)
    if bad:
        p.error(f"unknown methods {sorted(bad)}; known {KNOWN}")
    if args.with_base:
        methods = ["base", *methods]
    configs = parse_configs(args.configs)
    obs_seeds = [int(v) for v in args.obs_seeds.split(",")]
    seeds = [int(v) for v in args.sampler_seeds.split(",")]
    task = benchmark.get_task(args.task)
    out_root = Path(args.out)

    shift = build_prior_shift(args.task, args.prior_type, args.prior_id)
    if shift.expquad is None:
        p.error("K=1 runner: the prior ratio must be a single exponential-quadratic factor")
    eq = shift.expquad
    nn = upstream_nn()
    ckpt_hash = file_hash(args.checkpoint)
    ckpt = load_checkpoint(args.checkpoint)
    if ckpt.theta_dim != task.theta_dim or ckpt.x_dim != task.x_dim:
        p.error(f"checkpoint is ({ckpt.theta_dim}, {ckpt.x_dim}) but {args.task} is "
                f"({task.theta_dim}, {task.x_dim})")
    backbone = Backbone(ckpt, mode="conditional", nn=nn)
    snaps = e2e.DEFAULT_SNAPSHOT_FRACS if args.snapshots else ()

    common = {
        "round": "round10_k1_langevin", "task": args.task, "theta_dim": task.theta_dim,
        "x_dim": task.x_dim, "model_id": args.model_id,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256_16": ckpt_hash,
        "prior_type": args.prior_type, "prior_id": args.prior_id,
        "prior_json_sha256_16": benchmark.prior_json_hash(
            args.task, args.prior_type, args.prior_id),
        "transport_a": eq.a.tolist(), "transport_kappa": eq.kappa,
        "langevin_semantics": "delta = eta g(t)^2 dt / 2 (released PriorGuide corrector); "
                              "see e2e.langevin_step_size",
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
        "schedule": {"sigma_min": ckpt.schedule.sigma_min, "sigma_max": ckpt.schedule.sigma_max,
                     "t_min": ckpt.schedule.t_min, "t_max": ckpt.schedule.t_max},
        "num_samples": args.num_samples, "python": sys.version.split()[0],
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "host": platform.node(),
    }

    for obs_seed in obs_seeds:
        theta_true, x_o = benchmark.load_observation(
            args.task, args.prior_type, args.prior_id, obs_seed)
        t0 = time.perf_counter()
        pg_bank, pg_bank_wall = draw_base_bank(
            backbone, jax.random.PRNGKey((90000 + obs_seed) % (2**31)),
            args.pg_var_samples, x_o,
            e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform", langevin_steps=0))
        total_variance = float(np.var(pg_bank))
        print(f"[obs {obs_seed}] PG Sigma_post bank in {time.perf_counter() - t0:.1f}s "
              f"(total_var={total_variance:.5g})", flush=True)
        built = {}
        if "base" in methods:
            built["base"] = hooks.make_base_hook(backbone, x_o)
        if "pg_exact" in methods:
            built["pg_exact"] = hooks.make_pg_hook(
                backbone, x_o, shift.components,
                reverse_cov=hooks.upstream_reverse_cov(total_variance, shift.theta_dim))
        if "pg_vjp" in methods:
            built["pg_vjp"] = hooks.make_pg_vjp_hook(
                backbone, x_o, shift.components,
                reverse_cov=hooks.upstream_reverse_cov(total_variance, shift.theta_dim))
        if "a_full" in methods:
            built["a_full"] = hooks.make_a_full_hook(backbone, x_o, shift.components)
        if "spectra" in methods:
            built["spectra"] = hooks.make_tq_hook(backbone, x_o, eq.a_jnp, eq.kappa)
        mean, std = backbone.terminal_moments()

        for num_steps, lang_steps in configs:
            cfg = e2e.SamplerConfig(num_steps=num_steps, grid=args.grid,
                                    langevin_steps=lang_steps,
                                    langevin_ratio=args.langevin_ratio)
            tag = f"steps{num_steps}_L{lang_steps}_{args.grid}"
            taus = np.asarray(backbone.schedule.sigma(
                e2e.time_grid(backbone.schedule, cfg)) ** 2)
            for seed in seeds:
                k_init, k_noise = jax.random.split(
                    jax.random.PRNGKey((seed * 1_000_003 + obs_seed) % (2**31)))
                x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
                diff, lang = e2e.make_noise(k_noise, num_steps, args.num_samples,
                                            task.theta_dim, lang_steps)
                for name, (hook, ops, hook_meta) in built.items():
                    if name == "base" and lang_steps > 0:
                        continue
                    row_name = f"{name}_ms{seed}"
                    out = (out_root / "samples" / args.task / row_name / f"model_{args.model_id}"
                           / f"{args.prior_type}_{args.prior_id}" / f"obs_{obs_seed}")
                    if args.skip_existing and (out / f"{tag}.json").is_file():
                        continue
                    t1 = time.perf_counter()
                    res = e2e.reverse_sample(
                        e2e.instrument_hook(hook, ops), schedule=backbone.schedule, cfg=cfg,
                        x_init=x_init, diffusion_noise=diff, langevin_noise=lang,
                        snapshot_fracs=snaps)
                    wall = time.perf_counter() - t1
                    nfe_static = e2e.nfe_accounting(cfg, ops)
                    nfe_runtime = e2e.nfe_from_aux(res.step_aux)
                    out.mkdir(parents=True, exist_ok=True)
                    payload = {"samples": res.samples, "x_o": np.asarray(x_o),
                               "theta_true": np.asarray(theta_true), "times": res.times,
                               **{f"aux_{k}": v for k, v in res.step_aux.items()
                                  if not k.startswith(e2e.NFE_PREFIX)}}
                    np.savez_compressed(out / f"{tag}.npz", **payload)
                    is_pg = name in PG_FAMILIES
                    man = {**common, "obs_seed": obs_seed, "method": row_name,
                           "method_family": name, "method_meta": hook_meta,
                           "derivative_path": ({"pg_exact": "full denoiser Jacobian "
                                                            "(vmap(jacrev))",
                                                "pg_vjp": "one reverse-mode VJP"}.get(name)),
                           "conditioning": "conditional", "sampler_seed": seed,
                           "sampler": cfg.as_dict(), "nfe": nfe_static,
                           "nfe_runtime": nfe_runtime,
                           "nfe_runtime_equals_static": nfe_runtime == nfe_static,
                           "wall_clock_s": wall, "wall_clock_includes_compile": True,
                           "wall_clock_per_1000_samples_s": wall * 1000.0 / args.num_samples,
                           "cost": {"precompute_s": pg_bank_wall if is_pg else 0.0,
                                    "precompute_what": ("upstream scalar Sigma_post base draw"
                                                        if is_pg else None)},
                           "nan_or_inf": int((~np.isfinite(res.samples)).sum()),
                           "peak_gpu_memory_mb": peak_gpu_memory_mb(),
                           "pg_total_variance": total_variance if is_pg else None}
                    if "clip_hit_frac" in res.step_aux:
                        # e2e keeps the hook diagnostics of the predictor call of
                        # each reverse step; corrector calls only add NFE counters
                        clip = np.asarray(res.step_aux["clip_hit_frac"], float)
                        man["clip_hit_frac_mean_over_reverse_steps"] = float(np.mean(clip))
                        man["clip_hit_frac_max_over_reverse_steps"] = float(np.max(clip))
                        man["reverse_steps_with_any_clip"] = int((clip > 0).sum())
                    if name == "spectra":
                        man["effective_noise"] = effective_noise_report(
                            taus, eq.kappa, backbone.schedule)
                    (out / f"{tag}.json").write_text(json.dumps(man, indent=2, default=str))
                    print(f"[obs {obs_seed}][{tag}][{row_name}] {wall:.1f}s "
                          f"nfe={nfe_runtime.get('hook_calls')} nan={man['nan_or_inf']}",
                          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
