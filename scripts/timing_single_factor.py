#!/usr/bin/env python
"""Steady-state online cost of every single-factor method, PriorGuide-VJP included.

The measurement is the steady-state one: one process per task on one
A100, inputs (terminal draw, Brownian and Langevin increments, fixed key
``20260906``) built outside the timed region, one compiled kernel per
configuration from ``e2e.make_sampler``, one untimed compile + warm-up call,
``block_until_ready``, ``--repeats`` timed passes with the raw repeats kept, and
the gate of ``spectra.timing.timed_kernel``: a row whose timed repeats
trigger any XLA compilation is invalid and the script exits non-zero.  The hook
table is built inline, with one row beyond the benchmarked set:

``PriorGuide-VJP``  ``hooks.make_pg_vjp_hook``: PriorGuide's guidance with
                    ``J_m^T u`` from one reverse-mode VJP.

``PriorGuide`` (the benchmarked full-Jacobian implementation), Spectra,
PriorGuide-FullCov, Base and the SIR/SNIS banks are re-measured in the same
process, so every ratio compares two rows of one run.  PriorGuide's
``Sigma_post`` precompute is timed as its own first-use row.

Two checks travel with every run:

* the compile-gate self-test: a callable that builds a fresh ``jit`` on every
  call must be rejected by the same ``timed_kernel``, and the real kernels'
  first calls must register compile events (so a counter that never counts
  cannot pass);
* the call chain: jaxpr statistics of the ``pg_exact`` / ``pg_vjp`` / ``spectra``
  hooks (``spectra.timing.jaxpr_stats``).  ``pg_exact`` must show the
  ``theta_dim``-batched backward activations of ``vmap(jacrev)``, ``pg_vjp`` must
  show strictly fewer; the counts are recorded as they are.

    python scripts/timing_single_factor.py --task two_moons \\
        --checkpoint checkpoints/two_moons/model_0.pkl \\
        --configs 25:8,100:0 --out results/timing/single_factor
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.timing import CompileCounter, jaxpr_stats, timed_kernel  # noqa: E402
from spectra.utils import git_state, parse_configs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INPUT_KEY = 20260906  # the timing harness's fixed input key


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", default="0")
    p.add_argument("--obs-seed", type=int, default=1000000)
    p.add_argument("--prior-type", default="strong")
    p.add_argument("--prior-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--configs", default="25:0,25:8,50:2,100:0,100:2,250:0")
    p.add_argument("--langevin-ratio", type=float, default=0.5)
    p.add_argument("--grid", default="power")
    p.add_argument("--sir-budgets", default="1000,5000,20000")
    p.add_argument("--sir-steps", type=int, default=100)
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    configs = parse_configs(args.configs)
    budgets = [int(v) for v in args.sir_budgets.split(",") if v.strip()]
    spec = benchmark.get_task(args.task)
    ckpt = load_checkpoint(args.checkpoint)
    backbone = Backbone(ckpt, mode="conditional", nn=upstream_nn())
    shift = build_prior_shift(args.task, args.prior_type, args.prior_id)
    eq = shift.expquad
    _, x_o = benchmark.load_observation(args.task, args.prior_type, args.prior_id,
                                        args.obs_seed)
    mean, std = backbone.terminal_moments()
    rows = []

    def make_kernel(num, hook, cfg):
        """Compiled kernel plus its inputs; inputs are built outside the timed region."""
        k_init, k_noise = jax.random.split(jax.random.PRNGKey(INPUT_KEY))
        x_init = e2e.draw_terminal(k_init, num, mean, std)
        diff, lang = e2e.make_noise(k_noise, cfg.num_steps, num, spec.theta_dim,
                                    cfg.langevin_steps)
        jax.block_until_ready((x_init, diff, lang))
        kernel = e2e.make_sampler(hook, schedule=backbone.schedule, cfg=cfg,
                                  snapshot_fracs=())
        return kernel, (x_init, diff, lang)

    def record(label, family, n_draw, n_ret, cfg, t, ops, role, extra=None):
        nfe = e2e.nfe_accounting(cfg, ops) if ops is not None else {}
        rows.append({
            "timing_valid": int(t["timing_valid"]),
            "compile_s": round(t["compile_s"], 4),
            "compile_events_first_call": t["compile_events_first_call"],
            "compile_events_during_timing": t["compile_events_during_timing"],
            "task": args.task, "row": label, "family": family, "model_id": args.model_id,
            "obs_seed": args.obs_seed, "prior": f"{args.prior_type}_{args.prior_id}",
            "role": role, "trajectories_drawn": n_draw, "posterior_samples_returned": n_ret,
            "num_steps": cfg.num_steps, "langevin_steps": cfg.langevin_steps,
            "langevin_ratio": cfg.langevin_ratio, "grid": cfg.grid,
            "hook_calls_per_trajectory": nfe.get("hook_calls"),
            "base_score_calls_per_trajectory": nfe.get("base_score_calls"),
            "denoiser_jacobian_calls_per_trajectory": nfe.get("denoiser_jacobian_calls", 0),
            "score_vjp_calls_per_trajectory": nfe.get("score_vjp_calls", 0),
            "median_s": round(t["median_s"], 4), "q25_s": round(t["q25_s"], 4),
            "q75_s": round(t["q75_s"], 4), "iqr_s": round(t["iqr_s"], 4),
            "min_s": round(t["min_s"], 4), "max_s": round(t["max_s"], 4),
            "repeats": t["repeats"], "all_s": ";".join(f"{v:.4f}" for v in t["all_s"]),
            **(extra or {}),
        })
        print(f"[{args.task}] N={cfg.num_steps:<4} L={cfg.langevin_steps:<2} {label:<22} "
              f"median {t['median_s']:8.4f}s IQR {t['iqr_s']:.4f} "
              f"valid={int(t['timing_valid'])}", flush=True)

    counter = CompileCounter().install()
    base_hook, base_ops, _ = hooks.make_base_hook(backbone, x_o)
    pg_cfg = e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform", langevin_steps=0)
    k_pg, a_pg = make_kernel(args.pg_var_samples, base_hook, pg_cfg)
    t_pg = timed_kernel(k_pg, a_pg, args.repeats, counter)
    record("PG Sigma_post precompute", "pg_exact", args.pg_var_samples, 0, pg_cfg, t_pg,
           base_ops, "first_use_precompute", {"shared_by": "pg_exact,pg_vjp"})
    total_variance = float(np.var(np.asarray(k_pg(*a_pg)[0])))
    reverse_cov = hooks.upstream_reverse_cov(total_variance, shift.theta_dim)

    built = {
        "Base": ("base", hooks.make_base_hook(backbone, x_o)),
        "SPECTRA": ("spectra", hooks.make_tq_hook(backbone, x_o, eq.a_jnp, eq.kappa)),
        "PriorGuide": ("pg_exact", hooks.make_pg_hook(
            backbone, x_o, shift.components, reverse_cov=reverse_cov)),
        "PriorGuide-VJP": ("pg_vjp", hooks.make_pg_vjp_hook(
            backbone, x_o, shift.components, reverse_cov=reverse_cov)),
        "PriorGuide-FullCov": ("a_full", hooks.make_a_full_hook(backbone, x_o, shift.components)),
    }

    # --- call chain: what each hook's traced graph contains (not timed) -------
    t_mid = jnp.asarray(0.5 * (backbone.schedule.t_min + backbone.schedule.t_max))
    z_probe = e2e.draw_terminal(jax.random.PRNGKey(INPUT_KEY), args.num_samples, mean, std)
    callchain = {}
    for family in ("pg_exact", "pg_vjp", "spectra"):
        hook = next(v[1][0] for v in built.values() if v[0] == family)
        st_ = jaxpr_stats(jax.make_jaxpr(hook)(z_probe, t_mid), args.num_samples,
                          spec.theta_dim, backbone.num_nodes)
        callchain[family] = {k: v for k, v in st_.items() if k != "largest_outputs"}
    exact_d = callchain["pg_exact"]["d_batched_output_elements"]
    vjp_d = callchain["pg_vjp"]["d_batched_output_elements"]
    callchain["check"] = {
        "rule": "pg_exact carries theta_dim-batched backward activations (> 0) and "
                "pg_vjp strictly fewer; counts from spectra.timing.jaxpr_stats, recorded as is",
        "pg_exact_d_batched_output_elements": exact_d,
        "pg_vjp_d_batched_output_elements": vjp_d,
        "pass": bool(exact_d > 0 and vjp_d < exact_d),
    }
    print(f"[{args.task}] call chain: d-batched output elements pg_exact={exact_d} "
          f"vjp={vjp_d} spectra={callchain['spectra']['d_batched_output_elements']} "
          f"pass={callchain['check']['pass']}", flush=True)

    for num_steps, lang_steps in configs:
        cfg = e2e.SamplerConfig(num_steps=num_steps, grid=args.grid, langevin_steps=lang_steps,
                                langevin_ratio=args.langevin_ratio)
        for label, (family, (hook, ops, _)) in built.items():
            if family == "base" and lang_steps > 0:
                continue
            k, a = make_kernel(args.num_samples, hook, cfg)
            t = timed_kernel(k, a, args.repeats, counter)
            record(label, family, args.num_samples, args.num_samples, cfg, t, ops, "online")
    cfg_sir = e2e.SamplerConfig(num_steps=args.sir_steps, grid=args.grid, langevin_steps=0)
    for n_bank in budgets:
        k, a = make_kernel(n_bank, base_hook, cfg_sir)
        t = timed_kernel(k, a, args.repeats, counter)
        record(f"SIR/SNIS bank n={n_bank}", "sir", n_bank, args.num_samples, cfg_sir, t,
               base_ops, "online", {"sir_bank": n_bank})

    # --- compile-gate self-test: the gate must reject a recompiling callable ---
    def recompiling(x):
        return jax.jit(lambda y: y * 2.0 + 1.0)(x)

    probe = timed_kernel(recompiling, (jnp.ones((8,)),), 3, counter)
    firsts = [r["compile_events_first_call"] for r in rows if r["compile_s"] > 1.0]
    selftest = {"recompiling_callable_rejected": not probe["timing_valid"],
                "recompiling_events_during_timing": probe["compile_events_during_timing"],
                "min_first_call_events_over_rows": min(firsts) if firsts else 0}
    selftest["pass"] = bool(selftest["recompiling_callable_rejected"]
                            and selftest["min_first_call_events_over_rows"] > 0)
    print(f"[{args.task}] compile-gate self-test {selftest}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.task}.csv"
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps({
        "what": "steady-state online timing (spectra.timing.timed_kernel), including PriorGuide-VJP",
        "protocol": {"warmup": "one untimed pass per configuration", "snapshots": "disabled",
                     "sync": "block_until_ready", "repeats": args.repeats,
                     "statistic": "median / IQR, raw repeats kept in all_s",
                     "initial_draw_and_noise": f"PRNGKey({INPUT_KEY}), generated once per "
                                               "configuration outside the timed region",
                     "kernel": "e2e.make_sampler compiled kernel; compile_s is the first call, "
                               "the timed repeats execute the compiled program and are "
                               "asserted to add no compilation",
                     "hooks": "uninstrumented (production graph)", "grid": args.grid,
                     "langevin_ratio": args.langevin_ratio, "num_samples": args.num_samples,
                     "configs": configs, "sir_steps": args.sir_steps,
                     "pg_precompute": f"{args.pg_var_samples} x {args.pg_var_steps} uniform "
                                      "steps; shared by pg_exact and pg_vjp; its own row"},
        "task": args.task, "model_id": args.model_id, "obs_seed": args.obs_seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256_16": benchmark.source_hashes([args.checkpoint])[Path(args.checkpoint).name],
        "pg_total_variance": total_variance,
        "callchain_jaxpr": callchain, "compile_gate_selftest": selftest,
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
        "env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "NVIDIA_TF32_OVERRIDE",
                                                "JAX_DEFAULT_MATMUL_PRECISION",
                                                "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID")},
        "jax": jax.__version__, "python": sys.version.split()[0],
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2, default=str))
    invalid = [r for r in rows if not r["timing_valid"]]
    print(f"wrote {len(rows)} rows -> {path}", flush=True)
    failures = []
    if invalid:
        failures.append(f"{len(invalid)} of {len(rows)} rows recompiled during their timed "
                        "repeats")
        for r in invalid:
            print(f"  INVALID {r['task']} {r['row']} N={r['num_steps']} L={r['langevin_steps']} "
                  f"compile_events_during_timing={r['compile_events_during_timing']}", flush=True)
    if not selftest["pass"]:
        failures.append("compile-gate self-test failed")
    if not callchain["check"]["pass"]:
        failures.append("call-chain check failed")
    if failures:
        print("TIMING GATE FAILED: " + "; ".join(failures), flush=True)
        return 1
    print("all rows valid; compile-gate self-test and call-chain check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
