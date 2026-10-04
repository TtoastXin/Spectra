#!/usr/bin/env python
"""Sample the ``K``-component cells with PriorGuide and Spectra.

A thin runner over the K-general library; no estimator, hook or sampler
logic is defined here.  For each ``(K, observation, model, sampler
seed)`` it runs

``pg_exact``      PriorGuide's closure on the exact ``K``-component ratio (the
                  same hook every accuracy row uses);
``spectra_ref``   one atom per trajectory drawn from the reference component
                  weights, or
``spectra_ps``    the same with path-space estimated weights.

Conventions follow the other runners so the rows are comparable with
the rest of the study: terminal draw and noise keys as in
``scripts/sample_single_factor.py`` (``PRNGKey(seed * 1_000_003 + obs_seed)``),
atom labels as in ``scripts/mixture_weights_and_controlled.py``
(``PRNGKey(7 + seed)``), PriorGuide's scalar ``Sigma_post`` from a 5000 x 500
uniform base draw, and the standard output layout so ``compute_metrics.py``
scores the samples unchanged.

    python scripts/sample_k_scaling.py --weights reference --ks 2,4,8,16 \\
        --model-id 0 --checkpoint checkpoints/two_moons/model_0.pkl \\
        --out results/k_scaling
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

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, e2e, hooks  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.baselines import draw_base_bank  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra import weights
from spectra.utils import file_hash, git_state, peak_gpu_memory_mb  # noqa: E402
from scripts import k_scaling_config as KG  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_weights(kind: str, root: Path, k: int, obs_seed: int, model_id: str):
    """``(log_pi, provenance)`` for the atom draw, read from an existing weight file."""
    pid = KG.prior_id(k)
    if kind == "reference":
        path = (root / "references" / KG.TASK / f"{KG.PRIOR_TYPE}_{pid}"
                / f"obs_{obs_seed}_weights.json")
    elif kind == "pathspace":
        path = (root / "weights"
                / f"pathspace_{KG.TASK}_K{k}_p{pid}_o{obs_seed}_model{model_id}.json")
    else:
        raise ValueError(f"unknown weight source {kind!r}")
    if not path.is_file():
        raise SystemExit(f"missing {kind} weights: {path}")
    # A failed estimate must not reach the sampler: its logits can be finite and
    # equal, which would turn into a uniform component assignment without an error.
    log_pi, _ = weights.load_weight_record(path, expected_k=k)
    return log_pi, {"weight_source": kind, "weight_file": str(path),
                    "weight_file_sha256_16": file_hash(path),
                    "pi": np.exp(log_pi).tolist(),
                    "k_eff": KG.k_eff(np.exp(log_pi))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ks", default=",".join(str(k) for k in KG.K_P2))
    p.add_argument("--obs-seeds", default=",".join(str(o) for o in KG.OBS_P2))
    p.add_argument("--model-id", default="0")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--sampler-seeds", default=",".join(str(s) for s in KG.SAMPLER_SEEDS))
    p.add_argument("--weights", default="reference", choices=("reference", "pathspace"))
    p.add_argument("--methods", default=",".join(KG.METHODS_P2))
    p.add_argument("--num-samples", type=int, default=KG.NUM_SAMPLES)
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--out", default=KG.ROOT)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    ks = [int(v) for v in args.ks.split(",") if v.strip()]
    obs_seeds = [int(v) for v in args.obs_seeds.split(",") if v.strip()]
    seeds = [int(v) for v in args.sampler_seeds.split(",") if v.strip()]
    want = [m for m in args.methods.split(",") if m]
    spectra_row = "spectra_ref" if args.weights == "reference" else "spectra_ps"
    ckpt_path = args.checkpoint or KG.CHECKPOINT.format(m=args.model_id)
    task = benchmark.get_task(KG.TASK)
    ckpt = load_checkpoint(ckpt_path)
    backbone = Backbone(ckpt, mode="conditional", nn=upstream_nn())
    mean, std = backbone.terminal_moments()
    cfg = e2e.SamplerConfig(num_steps=KG.CONFIG[0], grid=KG.GRID,
                            langevin_steps=KG.CONFIG[1])
    tag = KG.tag()
    out_root = Path(args.out)

    common = {
        "round": "round14_p2_k_composition", "task": KG.TASK,
        "theta_dim": task.theta_dim, "x_dim": task.x_dim,
        "model_id": args.model_id, "checkpoint": str(Path(ckpt_path).resolve()),
        "checkpoint_sha256_16": file_hash(ckpt_path),
        "prior_type": KG.PRIOR_TYPE, "conditioning": "conditional",
        "num_samples": args.num_samples,
        "langevin_semantics": ("delta = eta g(t)^2 dt / 2 (released PriorGuide "
                               "corrector); see e2e.langevin_step_size"),
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
        "jax": jax.__version__, "python": sys.version.split()[0],
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
        "schedule": {"sigma_min": ckpt.schedule.sigma_min,
                     "sigma_max": ckpt.schedule.sigma_max,
                     "t_min": ckpt.schedule.t_min, "t_max": ckpt.schedule.t_max},
    }

    for k in ks:
        pid = KG.prior_id(k)
        shift = build_prior_shift(KG.TASK, KG.PRIOR_TYPE, pid)
        atoms = atoms_mod.atoms_from_shift(shift)
        if atoms.num_atoms != k:
            raise SystemExit(f"K={k} cell expanded into {atoms.num_atoms} atoms")
        for obs_seed in obs_seeds:
            theta_true, x_o = benchmark.load_observation(
                KG.TASK, KG.PRIOR_TYPE, pid, obs_seed)
            log_pi, weight_meta = load_weights(args.weights, out_root, k, obs_seed,
                                               args.model_id)
            built, meta = {}, {}
            if "pg_exact" in want:
                t0 = time.perf_counter()
                pg_bank, pg_bank_wall = draw_base_bank(
                    backbone, jax.random.PRNGKey((90000 + obs_seed) % (2 ** 31)),
                    args.pg_var_samples, x_o,
                    e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform",
                                      langevin_steps=0))
                total_variance = float(np.var(pg_bank))
                print(f"[K={k} obs {obs_seed}] PG Sigma_post bank in "
                      f"{time.perf_counter() - t0:.1f}s (var={total_variance:.5g})",
                      flush=True)
                built["pg_exact"] = hooks.make_pg_hook(
                    backbone, x_o, shift.components,
                    reverse_cov=hooks.upstream_reverse_cov(total_variance,
                                                           task.theta_dim))
                meta["pg_exact"] = {"pg_total_variance": total_variance,
                                    "precompute_s": pg_bank_wall}
            for seed in seeds:
                k_init, k_noise = jax.random.split(
                    jax.random.PRNGKey((seed * 1_000_003 + obs_seed) % (2 ** 31)))
                x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
                diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples,
                                            task.theta_dim, cfg.langevin_steps)
                rows = dict(built)
                labels = None
                if spectra_row in want or "spectra" in want:
                    labels = hooks.draw_atom_labels(
                        jax.random.PRNGKey(7 + seed), jnp.asarray(log_pi),
                        args.num_samples)
                    rows[spectra_row] = hooks.make_spectra_hook(backbone, x_o, atoms,
                                                             labels)
                for name, (hook, ops, hook_meta) in rows.items():
                    row_name = f"{name}_ms{seed}"
                    out = (out_root / "samples" / KG.TASK / row_name
                           / f"model_{args.model_id}" / f"{KG.PRIOR_TYPE}_{pid}"
                           / f"obs_{obs_seed}")
                    if args.skip_existing and (out / f"{tag}.json").is_file():
                        continue
                    t1 = time.perf_counter()
                    res = e2e.reverse_sample(
                        e2e.instrument_hook(hook, ops), schedule=backbone.schedule,
                        cfg=cfg, x_init=x_init, diffusion_noise=diff,
                        langevin_noise=lang, snapshot_fracs=())
                    wall = time.perf_counter() - t1
                    nfe_static = e2e.nfe_accounting(cfg, ops)
                    nfe_runtime = e2e.nfe_from_aux(res.step_aux)
                    out.mkdir(parents=True, exist_ok=True)
                    payload = {"samples": res.samples, "x_o": np.asarray(x_o),
                               "theta_true": np.asarray(theta_true),
                               "times": res.times}
                    if name != "pg_exact" and labels is not None:
                        payload["atom_labels"] = np.asarray(labels)
                    np.savez_compressed(out / f"{tag}.npz", **payload)
                    man = {**common, "K": k, "prior_id": pid, "obs_seed": obs_seed,
                           "method": row_name, "method_family": name,
                           "method_meta": hook_meta, "sampler_seed": seed,
                           "sampler": cfg.as_dict(), "nfe": nfe_static,
                           "nfe_runtime": nfe_runtime,
                           "nfe_runtime_equals_static": nfe_runtime == nfe_static,
                           "wall_clock_s": wall, "wall_clock_includes_compile": True,
                           "wall_clock_per_1000_samples_s":
                               wall * 1000.0 / args.num_samples,
                           "prior_json_sha256_16": benchmark.prior_json_hash(
                               KG.TASK, KG.PRIOR_TYPE, pid),
                           "nan_or_inf": int((~np.isfinite(res.samples)).sum()),
                           "peak_gpu_memory_mb": peak_gpu_memory_mb(),
                           **({} if name == "pg_exact" else weight_meta),
                           **meta.get(name, {})}
                    (out / f"{tag}.json").write_text(
                        json.dumps(man, indent=2, default=str))
                    print(f"[K={k} obs {obs_seed}][{row_name}] {wall:.1f}s "
                          f"nfe={nfe_runtime.get('hook_calls')} "
                          f"nan={man['nan_or_inf']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
