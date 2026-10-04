#!/usr/bin/env python
"""Appendix sensitivity: two-component online PriorGuide cost with the VJP derivative.

The first-use cost stack of ``scripts/timing_mixture.py`` is unchanged and
remains the benchmarked implementation.  Of its components only PriorGuide's
online row depends on how ``J_m^T u`` is computed; the ``Sigma_post``
precompute, Spectra's weight stages and Spectra's online sampler do not.  So
no re-sampling and no new weights are needed: for each of the same pairs, this
script rebuilds the online inputs of ``timing_mixture.py`` (N = 100, L = 0,
1000 samples, key ``PRNGKey(obs)``, ``Sigma_post`` from its ``PRNGKey(90000)``
bank) and times, in one process,

* ``PriorGuide``        ``hooks.make_pg_hook`` (the stack's row, re-measured here),
* ``PriorGuide-VJP``    ``hooks.make_pg_vjp_hook``,
* ``SPECTRA (online)``  ``hooks.make_spectra_hook``, online cost only; its atom
                        labels are drawn from the prior component weights because
                        the online kernel's cost does not depend on which atom a
                        trajectory carries (the Direct weights of the stack are
                        not rebuilt).

with ``spectra.timing.timed_kernel`` (compile excluded, rows marked invalid on
recompilation).  The first-use total for PG-VJP is the stack's precompute plus
this online row.

    python scripts/timing_mixture_vjp.py --task two_moons \\
        --checkpoint checkpoints/two_moons/model_0.pkl \\
        --pairs 0:1000000,1:1000001 --out results/timing/mixture_vjp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
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
from spectra.timing import CompileCounter, timed_kernel  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", default="0")
    p.add_argument("--pairs", default="0:1000000,1:1000001")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    counter = CompileCounter().install()
    spec = benchmark.get_task(args.task)
    backbone = Backbone(load_checkpoint(args.checkpoint), mode="conditional",
                        nn=upstream_nn())
    sch, dim = backbone.schedule, spec.theta_dim
    mean, std = backbone.terminal_moments()
    rows = []
    for spec_pair in [s for s in args.pairs.split(",") if s]:
        pid, obs = (int(v) for v in spec_pair.split(":"))
        shift = build_prior_shift(args.task, "mixture", pid)
        atoms = atoms_mod.atoms_from_shift(shift)
        _, x_o = benchmark.load_observation(args.task, "mixture", pid, obs)

        # Sigma_post exactly as timing_mixture.base_run(..., 90000) builds it
        cfg_b = e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform", langevin_steps=0)
        k_init, k_noise = jax.random.split(jax.random.PRNGKey(90000))
        xb = e2e.draw_terminal(k_init, args.pg_var_samples, mean, std)
        db, lb = e2e.make_noise(k_noise, args.pg_var_steps, args.pg_var_samples, dim, 0)
        bh, _, _ = hooks.make_base_hook(backbone, x_o)
        bank = e2e.make_sampler(bh, schedule=sch, cfg=cfg_b, snapshot_fracs=())(xb, db, lb)[0]
        total_var = float(np.var(np.asarray(bank)))

        # online inputs as in timing_mixture.py's online rows
        cfg = e2e.SamplerConfig(num_steps=args.num_steps, grid="power", langevin_steps=0)
        k_init, k_noise = jax.random.split(jax.random.PRNGKey((0 * 1_000_003 + obs) % (2**31)))
        x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
        diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples, dim, 0)
        labels = hooks.draw_atom_labels(jax.random.PRNGKey(7), jnp.asarray(atoms.log_b),
                                        args.num_samples)
        jax.block_until_ready((x_init, diff, labels))
        rc = hooks.upstream_reverse_cov(total_var, dim)
        built = {
            "PriorGuide": hooks.make_pg_hook(backbone, x_o, shift.components, reverse_cov=rc),
            "PriorGuide-VJP": hooks.make_pg_vjp_hook(backbone, x_o, shift.components,
                                                     reverse_cov=rc),
            "SPECTRA (online)": hooks.make_spectra_hook(backbone, x_o, atoms, labels),
        }
        for name, (hook, ops, _) in built.items():
            kern = e2e.make_sampler(hook, schedule=sch, cfg=cfg, snapshot_fracs=())
            t = timed_kernel(kern, (x_init, diff, lang), args.repeats, counter)
            nfe = e2e.nfe_accounting(cfg, ops)
            rows.append({"task": args.task, "model_id": args.model_id, "prior_id": pid,
                         "obs_seed": obs, "method": name, "component": "online_1000_samples",
                         "median_s": round(t["median_s"], 4), "iqr_s": round(t["iqr_s"], 4),
                         "timing_valid": int(t["timing_valid"]),
                         "compile_s": round(t["compile_s"], 4),
                         "compile_events_during_timing": t["compile_events_during_timing"],
                         "repeats": t["repeats"],
                         "all_s": ";".join(f"{v:.4f}" for v in t["all_s"]),
                         "hook_calls": nfe["hook_calls"],
                         "jacobian_calls": nfe.get("denoiser_jacobian_calls", 0),
                         "score_vjp_calls": nfe.get("score_vjp_calls", 0),
                         "pg_total_variance": total_var})
            print(f"[{args.task} p{pid} o{obs}] {name:<18} median {t['median_s']:.4f}s "
                  f"valid={int(t['timing_valid'])}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.task}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps({
        "what": "K=2 online PriorGuide cost with the VJP derivative (appendix sensitivity)",
        "inputs_mirror": "scripts/timing_mixture.py online rows and Sigma_post bank",
        "spectra_labels": "drawn from prior component weights (online cost only)",
        "task": args.task, "pairs": args.pairs, "checkpoint": str(Path(args.checkpoint).resolve()),
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
        "env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "NVIDIA_TF32_OVERRIDE", "SLURM_JOB_ID")},
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
    }, indent=2, default=str))
    bad = [r for r in rows if not r["timing_valid"]]
    print(f"-> {path}", flush=True)
    if bad:
        print(f"INVALID: {len(bad)} rows recompiled during timing", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
