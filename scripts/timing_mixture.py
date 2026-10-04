#!/usr/bin/env python
"""Paired first-use cost stack for the two-component practical methods.

For a few pre-registered pairs, every cost component of every method is timed
in one process on one GPU with the benchmark budgets: one untimed warm-up,
``block_until_ready`` synchronisation, ``--repeats`` timed passes, raw repeats
kept.  Inputs (terminal draws, noise arrays, atom labels) are generated outside
the timed region.  The cost stack is

    T_first_use = T_ratio + T_precompute + T_weights + T_online

with T_ratio = 0 for the exact analytic ratio used throughout;
T_precompute = PriorGuide's scalar Sigma_post bank (5000 x 500 uniform);
T_weights = Spectra-Direct's 3 base banks x 4000 @ 500 + plug-in evaluation, or
Spectra-PS's 2 atoms x 3 estimator seeds x 4096 trajectories x 400 steps;
T_online = 1000 posterior samples at N=100, L=0 (Base / PriorGuide / FullCov /
Spectra transported sampler).  SIR/SNIS is a 20k base bank at N=100 plus
weights and resampling.  Reuse across priors is not measured here; the
algebraic reuse conditions are only stated in the manifest.

    python scripts/timing_mixture.py --task two_moons \\
        --checkpoint checkpoints/two_moons/model_0.pkl \\
        --pairs 0:1000000,1:1000001 --repeats 5 --out results/timing/mixture
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, e2e, hooks, pathspace  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.baselines import sir_resample, snis_weights  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.timing import CompileCounter  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def timed_cpu(fn, repeats):
    t0 = time.perf_counter()
    fn()
    compile_s = time.perf_counter() - t0
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - t0)
    walls.sort()
    return {"median_s": float(st.median(walls)), "min_s": walls[0], "max_s": walls[-1],
            "repeats": repeats, "all_s": walls, "compile_s": compile_s,
            "compile_events_first_call": 0, "compile_events_during_timing": 0,
            "timing_valid": True}


COUNTER = CompileCounter()


def timed_gpu(fn, repeats):
    """First call = tracing + XLA compilation + one execution; then ``repeats``
    executions of the compiled program, which must add no compilation."""
    n0 = COUNTER.n
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    n_first = COUNTER.n - n0
    n1 = COUNTER.n
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        jax.block_until_ready(res)
        walls.append(time.perf_counter() - t0)
    n_timed = COUNTER.n - n1
    walls.sort()
    return {"median_s": float(st.median(walls)), "min_s": walls[0], "max_s": walls[-1],
            "repeats": repeats, "all_s": walls, "compile_s": compile_s,
            "compile_events_first_call": n_first, "compile_events_during_timing": n_timed,
            # a repeat that recompiles is not a steady-state measurement
            "timing_valid": n_timed == 0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", default="0")
    p.add_argument("--pairs", default="0:1000000")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--direct-n-per-seed", type=int, default=4000)
    p.add_argument("--direct-seeds", type=int, default=3)
    p.add_argument("--direct-steps", type=int, default=500)
    p.add_argument("--ps-steps", type=int, default=400)
    p.add_argument("--ps-trajectories", type=int, default=4096)
    p.add_argument("--ps-seeds", type=int, default=3)
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--sir-bank", type=int, default=20000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--ps-repeats", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    COUNTER.install()
    spec = benchmark.get_task(args.task)
    ckpt = load_checkpoint(args.checkpoint)
    backbone = Backbone(ckpt, mode="conditional", nn=upstream_nn())
    sch = backbone.schedule
    mean, std = backbone.terminal_moments()
    box = benchmark.training_support(args.task)
    dim = spec.theta_dim
    rows = []

    def rec(pair, method, component, t, **extra):
        rows.append({"task": args.task, "model_id": args.model_id, "prior_id": pair[0], "obs_seed": pair[1],
                     "method": method, "component": component,
                     "median_s": round(t["median_s"], 4), "min_s": round(t["min_s"], 4), "max_s": round(t["max_s"], 4),
                     "timing_valid": int(t["timing_valid"]),
                     "compile_s": round(t["compile_s"], 4),
                     "compile_events_first_call": t["compile_events_first_call"],
                     "compile_events_during_timing": t["compile_events_during_timing"],
                     "repeats": t["repeats"], "all_s": ";".join(f"{v:.4f}" for v in t["all_s"]), **extra})
        print(f"[{args.task} p{pair[0]} o{pair[1]}] {method:<16} {component:<28} median {t['median_s']:8.3f}s", flush=True)

    for spec_pair in [s for s in args.pairs.split(",") if s]:
        pid, obs = (int(v) for v in spec_pair.split(":"))
        pair = (pid, obs)
        shift = build_prior_shift(args.task, "mixture", pid)
        atoms = atoms_mod.atoms_from_shift(shift)
        _, x_o = benchmark.load_observation(args.task, "mixture", pid, obs)

        # generic base draw builder (inputs outside the timed region)
        def base_run(num, steps, grid, key):
            """A compiled base-sampling kernel bound to fixed inputs (built outside the timed region)."""
            cfg = e2e.SamplerConfig(num_steps=steps, grid=grid, langevin_steps=0)
            k_init, k_noise = jax.random.split(jax.random.PRNGKey(key))
            x_init = e2e.draw_terminal(k_init, num, mean, std)
            diff, lang = e2e.make_noise(k_noise, steps, num, dim, 0)
            hook, _, _ = hooks.make_base_hook(backbone, x_o)
            jax.block_until_ready((x_init, diff))
            kernel = e2e.make_sampler(hook, schedule=sch, cfg=cfg, snapshot_fracs=())

            def run():
                return kernel(x_init, diff, lang)[0]
            return run

        # --- PriorGuide precompute: scalar Sigma_post bank
        pg_run = base_run(args.pg_var_samples, args.pg_var_steps, "uniform", 90000)
        rec(pair, "PriorGuide", "precompute_sigma_post_bank", timed_gpu(pg_run, args.repeats),
            draws=args.pg_var_samples, steps=args.pg_var_steps)
        total_var = float(np.var(np.asarray(pg_run())))

        # --- Spectra-Direct weights: banks + plug-in
        direct_runs = [base_run(args.direct_n_per_seed, args.direct_steps, "uniform", 20260826 + s)
                       for s in range(args.direct_seeds)]

        def direct_banks():
            return [r() for r in direct_runs]
        t_banks = timed_gpu(direct_banks, args.repeats)
        rec(pair, "SPECTRA-Direct", "weights_base_banks", t_banks, draws=args.direct_n_per_seed * args.direct_seeds,
            steps=args.direct_steps, seeds=args.direct_seeds)
        banks = [np.asarray(b, float) for b in direct_banks()]

        def plugin():
            return [atoms_mod.evidence_plugin(b, atoms)["log_z"] for b in banks]
        rec(pair, "SPECTRA-Direct", "weights_plugin_eval_cpu", timed_cpu(plugin, args.repeats))
        lz = np.mean([atoms_mod.evidence_plugin(b, atoms)["log_z"] for b in banks], axis=0)
        log_pi = atoms_mod.atom_log_weights(atoms, lz)

        # --- Spectra-PS weights: all atoms x all estimator seeds
        #
        # Everything that is not the measured computation is hoisted out of the
        # timed region: the per-atom transported score (memoised by
        # ``learned_scores``), the compiled chain kernel for each atom, the
        # terminal draws, the Brownian increments and the zero initial log-weight
        # vector.  The timed function calls only compiled kernels on
        # materialised inputs, so no repeat rebuilds the scan body or recompiles.
        score_p, make_q = pathspace.learned_scores(backbone, x_o, atoms)
        cfg_ps = e2e.SamplerConfig(num_steps=args.ps_steps, grid="power", langevin_steps=0)
        ps_calls = []
        for s in range(args.ps_seeds):
            for k in range(atoms.num_atoms):
                key = jax.random.PRNGKey(1000 * s + 17 * k)
                k_init, k_noise = jax.random.split(key)
                x_init = e2e.draw_terminal(k_init, args.ps_trajectories, mean, std)
                diff, _ = e2e.make_noise(k_noise, args.ps_steps, args.ps_trajectories, dim, 0)
                lw0 = jnp.zeros(args.ps_trajectories)
                kern = pathspace.make_atom_chain(score_p, make_q(k), schedule=sch, cfg=cfg_ps)
                jax.block_until_ready((x_init, diff, lw0))
                ps_calls.append((kern, x_init, diff, lw0))

        def ps_all():
            return [kern(x, d, lw) for kern, x, d, lw in ps_calls]
        rec(pair, "SPECTRA-PS", "weights_path_chains_all_seeds_atoms", timed_gpu(ps_all, args.ps_repeats),
            chains=len(ps_calls), trajectories=args.ps_trajectories, steps=args.ps_steps, seeds=args.ps_seeds,
            atoms=atoms.num_atoms)

        # --- SIR/SNIS: bank + weights + resample
        sir_run = base_run(args.sir_bank, args.num_steps, "power", (7_000_000 + obs * 97 + args.sir_bank) % (2**31))
        rec(pair, "SIR/SNIS", "bank_draw", timed_gpu(sir_run, args.repeats), draws=args.sir_bank, steps=args.num_steps)
        bank = np.asarray(sir_run(), float)

        def sir_post():
            w = snis_weights(shift, bank, box)
            if not w["degenerate"]:
                sir_resample(np.random.default_rng(0), bank, w["w"], args.num_samples)
        rec(pair, "SIR/SNIS", "weights_and_resample_cpu", timed_cpu(sir_post, args.repeats))

        # --- online rows (N=100, L=0, 1000 samples), paired inputs
        cfg = e2e.SamplerConfig(num_steps=args.num_steps, grid="power", langevin_steps=0)
        k_init, k_noise = jax.random.split(jax.random.PRNGKey((0 * 1_000_003 + obs) % (2**31)))
        x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
        diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples, dim, 0)
        labels = hooks.draw_atom_labels(jax.random.PRNGKey(7), jnp.asarray(log_pi), args.num_samples)
        jax.block_until_ready((x_init, diff, labels))
        built = {
            "Base": hooks.make_base_hook(backbone, x_o),
            "PriorGuide": hooks.make_pg_hook(backbone, x_o, shift.components,
                                             reverse_cov=hooks.upstream_reverse_cov(total_var, dim)),
            "PriorGuide-FullCov": hooks.make_a_full_hook(backbone, x_o, shift.components),
            "SPECTRA (Direct/PS online)": hooks.make_spectra_hook(backbone, x_o, atoms, labels),
        }
        for name, (hook, ops, _) in built.items():
            kern = e2e.make_sampler(hook, schedule=sch, cfg=cfg, snapshot_fracs=())

            def run(kern=kern):
                return kern(x_init, diff, lang)[0]
            nfe = e2e.nfe_accounting(cfg, ops)
            rec(pair, name, "online_1000_samples", timed_gpu(run, args.repeats), samples=args.num_samples,
                steps=args.num_steps, hook_calls=nfe["hook_calls"],
                jacobian_calls=nfe.get("denoiser_jacobian_calls", 0))

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
        w.writeheader(); w.writerows(rows)

    # cost stack per pair
    stack = []
    for spec_pair in [s for s in args.pairs.split(",") if s]:
        pid, obs = (int(v) for v in spec_pair.split(":"))
        get = {(r["method"], r["component"]): r["median_s"] for r in rows if r["prior_id"] == pid and r["obs_seed"] == obs}
        on = {m: get.get((m, "online_1000_samples"), float("nan")) for m in
              ("Base", "PriorGuide", "PriorGuide-FullCov", "SPECTRA (Direct/PS online)")}
        stack += [
            {"task": args.task, "prior_id": pid, "obs_seed": obs, "method": "Base", "ratio_s": 0.0, "precompute_s": 0.0,
             "weights_s": 0.0, "online_s": on["Base"], "first_use_s": on["Base"]},
            {"task": args.task, "prior_id": pid, "obs_seed": obs, "method": "SIR/SNIS (20k)", "ratio_s": 0.0, "precompute_s": 0.0,
             "weights_s": get.get(("SIR/SNIS", "bank_draw"), float("nan")) + get.get(("SIR/SNIS", "weights_and_resample_cpu"), 0.0),
             "online_s": 0.0,
             "first_use_s": get.get(("SIR/SNIS", "bank_draw"), float("nan")) + get.get(("SIR/SNIS", "weights_and_resample_cpu"), 0.0)},
            {"task": args.task, "prior_id": pid, "obs_seed": obs, "method": "PriorGuide", "ratio_s": 0.0,
             "precompute_s": get.get(("PriorGuide", "precompute_sigma_post_bank"), float("nan")), "weights_s": 0.0,
             "online_s": on["PriorGuide"],
             "first_use_s": get.get(("PriorGuide", "precompute_sigma_post_bank"), float("nan")) + on["PriorGuide"]},
            {"task": args.task, "prior_id": pid, "obs_seed": obs, "method": "PriorGuide-FullCov (ours)", "ratio_s": 0.0,
             "precompute_s": 0.0, "weights_s": 0.0, "online_s": on["PriorGuide-FullCov"], "first_use_s": on["PriorGuide-FullCov"]},
            {"task": args.task, "prior_id": pid, "obs_seed": obs, "method": "SPECTRA-Direct", "ratio_s": 0.0, "precompute_s": 0.0,
             "weights_s": get.get(("SPECTRA-Direct", "weights_base_banks"), float("nan")) + get.get(("SPECTRA-Direct", "weights_plugin_eval_cpu"), 0.0),
             "online_s": on["SPECTRA (Direct/PS online)"],
             "first_use_s": get.get(("SPECTRA-Direct", "weights_base_banks"), float("nan")) + get.get(("SPECTRA-Direct", "weights_plugin_eval_cpu"), 0.0) + on["SPECTRA (Direct/PS online)"]},
            {"task": args.task, "prior_id": pid, "obs_seed": obs, "method": "SPECTRA-PS", "ratio_s": 0.0, "precompute_s": 0.0,
             "weights_s": get.get(("SPECTRA-PS", "weights_path_chains_all_seeds_atoms"), float("nan")),
             "online_s": on["SPECTRA (Direct/PS online)"],
             "first_use_s": get.get(("SPECTRA-PS", "weights_path_chains_all_seeds_atoms"), float("nan")) + on["SPECTRA (Direct/PS online)"]},
        ]
    with open(out_dir / f"{args.task}_cost_stack.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stack[0].keys()))
        w.writeheader(); w.writerows(stack)
    (out_dir / f"{args.task}.json").write_text(json.dumps({
        "what": "paired K=2 first-use cost stack, one process on one GPU",
        "protocol": {"warmup": "one untimed pass per component", "sync": "block_until_ready", "snapshots": "disabled",
                     "repeats": args.repeats, "ps_repeats": args.ps_repeats, "statistic": "median; raw repeats kept",
                     "inputs": "terminal draws, noise arrays and atom labels generated outside the timed region",
                     "kernel": "e2e.make_sampler compiled kernels; compile_s is the first call, the timed repeats "
                               "execute the compiled program and are asserted to add no compilation",
                     "hooks": "uninstrumented (production graph)",
                     "budgets": {"direct": f"{args.direct_seeds} x {args.direct_n_per_seed} @ {args.direct_steps} uniform",
                                 "ps": f"{args.ps_seeds} seeds x atoms x {args.ps_trajectories} @ {args.ps_steps} power",
                                 "pg_bank": f"{args.pg_var_samples} @ {args.pg_var_steps} uniform",
                                 "sir": f"{args.sir_bank} @ {args.num_steps} power", "online": f"{args.num_samples} @ {args.num_steps}, L=0"},
                     "ratio": "exact analytic components (T_ratio = 0)",
                     "reuse": "not measured; algebraic reuse conditions only (Direct bank: same x across b_k and new components; PS chains: same x, same components, new b_k; PG bank: same x)"},
        "task": args.task, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256_16": benchmark.source_hashes([args.checkpoint])[Path(args.checkpoint).name],
        "devices": [str(d) for d in jax.devices()], "host": platform.node(), "jax": jax.__version__,
        "python": sys.version.split()[0], "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2, default=str))
    invalid = [r for r in rows if not r["timing_valid"]]
    print(f"wrote {path} and cost stack", flush=True)
    if invalid:
        print(f"INVALID: {len(invalid)} of {len(rows)} rows recompiled during their timed repeats "
              "and are not steady-state measurements:", flush=True)
        for r in invalid:
            print(f"  {r['task']} p{r['prior_id']} {r['method']} {r['component']} "
                  f"compile_events_during_timing={r['compile_events_during_timing']}", flush=True)
        return 1
    print("all rows valid: no compilation during any timed repeat", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
