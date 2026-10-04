#!/usr/bin/env python
"""Online sampling cost against mixture complexity ``K``.

One observation, one trained model, one sampler configuration, one GPU.  Only
the number of components in the exact prior ratio changes.  Three online paths
are timed at every ``K``:

``spectra``     one atom is drawn per trajectory before the timed region and
                fixed for the whole run, so each step is one batched backbone
                call plus that trajectory's analytic transform, with no
                ``K``-loop;
``pg_legacy``   PriorGuide's closure as the accuracy rows use it: one backbone
                score, the full denoiser Jacobian, and the ``K``-component
                Gaussian product recombined at every step;
``pg_vjp``      the same algebra with the Jacobian replaced by one
                vector--Jacobian product (timed only in this study: it
                agrees with ``pg_legacy`` in float64 but not within the
                strict float32/TF32 tolerance, so no accuracy row here uses
                it).

Timing uses the steady-state harness ``spectra.timing.timed_kernel``: inputs
built outside the timed region, one untimed compile+warm-up call,
``block_until_ready``, ``--repeats`` timed passes, and a gate that invalidates
any row whose timed repeats triggered a recompilation.  PriorGuide's
``Sigma_post`` precompute and the atom draw are excluded from the timed region
and reported separately.

Alongside the wall clock the script records the per-step operation count of
each path (``PER_STEP_OPS``), which shows the structural ``K``-dependence
independently of the hardware.

    python scripts/timing_k_scaling.py --checkpoint checkpoints/two_moons/model_0.pkl \\
        --model-id 0 --out results/k_scaling
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, e2e, hooks  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.baselines import draw_base_bank  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.timing import CompileCounter, timed_kernel  # noqa: E402
from spectra.utils import file_hash, git_state  # noqa: E402
from scripts import k_scaling_config as KG  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# What one hook call costs, read off the implementations in
# spectra/hooks.py.  This describes the computation graph rather than a
# measurement, so it does not depend on the hardware, the batch size or the
# dimension.
PER_STEP_OPS = {
    "spectra": {
        "backbone_calls": "1 (batched; per-trajectory transformed query and time shift)",
        "derivative_path": "none",
        "k_dependent_work": "none -- each trajectory carries one fixed component",
        "source": "hooks.make_spectra_hook",
    },
    "pg_legacy": {
        "backbone_calls": "1 score + 1 full denoiser Jacobian (theta_dim passes)",
        "derivative_path": "jacobian",
        "k_dependent_work": ("K covariance solves, K log-determinants, K-way "
                             "softmax responsibilities and a K-way residual sum "
                             "(methods.gaussian_product_moments)"),
        "source": "hooks.make_pg_hook",
    },
    "pg_vjp": {
        "backbone_calls": "1 score + 1 vector-Jacobian product (one backward pass)",
        "derivative_path": "vjp",
        "k_dependent_work": ("K covariance solves, K log-determinants, K-way "
                             "softmax responsibilities and a K-way residual sum "
                             "(methods.gaussian_product_moments)"),
        "source": "hooks.make_pg_vjp_hook",
    },
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=KG.CHECKPOINT.format(m=0))
    p.add_argument("--model-id", default="0")
    p.add_argument("--obs-seed", type=int, default=KG.OBS_P1[0])
    p.add_argument("--ks", default=",".join(str(k) for k in KG.K_P1))
    p.add_argument("--num-samples", type=int, default=KG.NUM_SAMPLES)
    p.add_argument("--repeats", type=int, default=KG.TIMING_REPEATS)
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--out", default=KG.ROOT)
    args = p.parse_args()

    ks = [int(v) for v in args.ks.split(",") if v.strip()]
    task = benchmark.get_task(KG.TASK)
    ckpt = load_checkpoint(args.checkpoint)
    backbone = Backbone(ckpt, mode="conditional", nn=upstream_nn())
    mean, std = backbone.terminal_moments()
    cfg = e2e.SamplerConfig(num_steps=KG.CONFIG[0], grid=KG.GRID,
                            langevin_steps=KG.CONFIG[1])
    counter = CompileCounter().install()

    # Everything below is built once and outside every timed region.
    _, x_o = benchmark.load_observation(KG.TASK, KG.PRIOR_TYPE, KG.prior_id(ks[0]),
                                        args.obs_seed)
    k_init, k_noise = jax.random.split(jax.random.PRNGKey(20260917))
    x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
    diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples,
                                task.theta_dim, cfg.langevin_steps)
    jax.block_until_ready((x_init, diff, lang))

    t0 = time.perf_counter()
    pg_bank, pg_bank_wall = draw_base_bank(
        backbone, jax.random.PRNGKey((90000 + args.obs_seed) % (2 ** 31)),
        args.pg_var_samples, x_o,
        e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform",
                          langevin_steps=0))
    total_variance = float(np.var(pg_bank))
    precompute_s = time.perf_counter() - t0
    print(f"[precompute] PG Sigma_post bank {precompute_s:.1f}s "
          f"(total_var={total_variance:.5g}); excluded from every timed row",
          flush=True)

    rows, label_draw = [], {}
    for k in ks:
        pid = KG.prior_id(k)
        shift = build_prior_shift(KG.TASK, KG.PRIOR_TYPE, pid)
        atoms = atoms_mod.atoms_from_shift(shift)
        if atoms.num_atoms != k:
            raise SystemExit(f"K={k} cell expanded into {atoms.num_atoms} atoms")
        _, x_k = benchmark.load_observation(KG.TASK, KG.PRIOR_TYPE, pid,
                                            args.obs_seed)
        if not np.array_equal(np.asarray(x_k), np.asarray(x_o)):
            raise SystemExit(f"K={k} cell carries a different observation")

        # the online timer treats the weights as already available; the label
        # distribution does not affect timing, so the equal prior weights are
        # used and the draw happens outside the timed region
        t_lab = time.perf_counter()
        labels = hooks.draw_atom_labels(
            jax.random.PRNGKey(7), jnp.asarray(np.log(np.full(k, 1.0 / k))),
            args.num_samples)
        labels = jax.block_until_ready(labels)
        label_draw[k] = time.perf_counter() - t_lab

        built = {
            "spectra": hooks.make_spectra_hook(backbone, x_o, atoms, labels),
            "pg_legacy": hooks.make_pg_hook(
                backbone, x_o, shift.components,
                reverse_cov=hooks.upstream_reverse_cov(total_variance, task.theta_dim)),
            "pg_vjp": hooks.make_pg_vjp_hook(
                backbone, x_o, shift.components,
                reverse_cov=hooks.upstream_reverse_cov(total_variance, task.theta_dim)),
        }
        for name in KG.METHODS_P1:
            hook, ops, hook_meta = built[name]
            kernel = e2e.make_sampler(hook, schedule=backbone.schedule, cfg=cfg,
                                      snapshot_fracs=())
            t = timed_kernel(kernel, (x_init, diff, lang), args.repeats, counter)
            nfe = e2e.nfe_accounting(cfg, ops)
            rows.append({
                "task": KG.TASK, "K": k, "prior_id": pid, "method": name,
                "method_label": KG.METHOD_LABEL[name], "model_id": args.model_id,
                "obs_seed": args.obs_seed, "num_samples": args.num_samples,
                "num_steps": cfg.num_steps, "langevin_steps": cfg.langevin_steps,
                "grid": cfg.grid,
                "median_s": t["median_s"], "min_s": t["min_s"], "max_s": t["max_s"],
                "q25_s": t["q25_s"], "q75_s": t["q75_s"], "iqr_s": t["iqr_s"],
                "repeats": t["repeats"],
                "all_s": ";".join(f"{v:.5f}" for v in t["all_s"]),
                "timing_valid": int(t["timing_valid"]),
                "compile_s": t["compile_s"],
                "compile_events_first_call": t["compile_events_first_call"],
                "compile_events_during_timing": t["compile_events_during_timing"],
                "hook_calls_per_trajectory": nfe.get("hook_calls"),
                "base_score_calls_per_trajectory": nfe.get("base_score_calls"),
                "denoiser_jacobian_calls_per_trajectory": nfe.get(
                    "denoiser_jacobian_calls", 0),
                "score_vjp_calls_per_trajectory": nfe.get("score_vjp_calls", 0),
                "per_step_backbone_calls": PER_STEP_OPS[name]["backbone_calls"],
                "per_step_derivative_path": PER_STEP_OPS[name]["derivative_path"],
                "per_step_k_dependent_work": PER_STEP_OPS[name]["k_dependent_work"],
                "excluded_precompute_s": (precompute_s if name.startswith("pg")
                                          else 0.0),
                "excluded_label_draw_s": (label_draw[k] if name == "spectra" else 0.0),
                "hook_meta": json.dumps(hook_meta, default=str),
            })
            print(f"[K={k:<3} {name:<10}] median {t['median_s']:8.4f}s  "
                  f"IQR {t['iqr_s']:.4f}  valid={t['timing_valid']}", flush=True)

    out = Path(args.out)
    (out / "timing").mkdir(parents=True, exist_ok=True)
    csv_path = out / "timing" / "online_timing.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by = {(r["K"], r["method"]): r["median_s"] for r in rows}
    ratios = [{"K": k,
               "spectra_s": by[(k, "spectra")],
               "pg_legacy_s": by[(k, "pg_legacy")],
               "pg_vjp_s": by[(k, "pg_vjp")],
               "pg_legacy_over_spectra": by[(k, "pg_legacy")] / by[(k, "spectra")],
               "pg_vjp_over_spectra": by[(k, "pg_vjp")] / by[(k, "spectra")]}
              for k in ks]
    with open(out / "timing" / "ratios.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ratios[0].keys()))
        w.writeheader()
        for r in ratios:
            w.writerow(r)

    (out / "timing" / "timing.json").write_text(json.dumps({
        "task": KG.TASK, "ks": ks, "config": list(KG.CONFIG),
        "num_samples": args.num_samples, "repeats": args.repeats,
        "obs_seed": args.obs_seed, "model_id": args.model_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256_16": file_hash(args.checkpoint),
        "pg_sigma_post_precompute_s": precompute_s,
        "pg_total_variance": total_variance,
        "atom_label_draw_s": label_draw,
        "per_step_accounting": PER_STEP_OPS,
        "harness": ("spectra/timing.timed_kernel: inputs outside the "
                    "timed region, one untimed compile/warm-up call, "
                    "block_until_ready, timed repeats invalidated by any "
                    "recompilation"),
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
        "jax": jax.__version__, "python": sys.version.split()[0],
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
    }, indent=2, default=str))

    invalid = [r for r in rows if not r["timing_valid"]]
    print(f"-> {csv_path}")
    if invalid:
        for r in invalid:
            print(f"INVALID K={r['K']} {r['method']}: "
                  f"{r['compile_events_during_timing']} compilations during timing",
                  flush=True)
        raise SystemExit(f"{len(invalid)} of {len(rows)} timing rows are not "
                         "steady-state measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
