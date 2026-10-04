#!/usr/bin/env python
"""End-to-end sampling on the compiled correlated cell.

The target prior is outside the exact-atom family, so the methods receive
different inputs:

* Spectra runs on the compiled dictionary, so its error includes the
  compilation error.
* PriorGuide and PG-FullCov are Gaussian-closure methods that accept an
  arbitrary covariance, so they get the true correlated Gaussian ratio rather
  than the compiled approximation.  The prior-JSON schema only carries a
  per-dimension sigma, so the full-covariance ratio is built directly here.
* Base is the unguided sampler.

Samples are written in the standard layout.  ``scripts/score_correlated.py``
(and ``scripts/score_correlated_k64.py`` for ``K = 64``) score them against both
the compiled-target and the true-target reference; the general
``scripts/compute_metrics.py`` scores one reference per cell and is not used for
this study.

    python scripts/sample_correlated.py --prior-id 21 --obs-seed 1000003 \\
        --checkpoint checkpoints/two_moons/model_0.pkl --model-id 0 \\
        --out results/correlated_prior/reference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, compile_atoms, e2e, hooks, weights  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.ratios import GaussianComponents  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def true_ratio_components(box) -> tuple:
    """The true correlated Gaussian ratio as a single full-covariance component."""
    prior = compile_atoms.design_prior((np.asarray(box[0], float),
                                        np.asarray(box[1], float)))
    comps = GaussianComponents(
        log_weights=jnp.zeros(1),
        means=jnp.asarray(prior.mu)[None, :],
        covs=jnp.asarray(prior.cov)[None, :, :])
    return prior, comps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="two_moons")
    p.add_argument("--prior-id", type=int, required=True)
    p.add_argument("--obs-seed", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--weights", default=None,
                   help="Spectra atom weights (log_pi JSON); defaults to the "
                        "compiled-target reference next to the reference npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    shift = build_prior_shift(args.task, "mixture", args.prior_id)
    atoms = atoms_mod.atoms_from_shift(shift)
    theta_true, x_o = benchmark.load_observation(args.task, "mixture",
                                                 args.prior_id, args.obs_seed)
    box = benchmark.training_support(args.task)
    prior, true_comps = true_ratio_components(box)

    ckpt = load_checkpoint(args.checkpoint)
    backbone = Backbone(ckpt, mode="conditional", nn=upstream_nn())
    ckpt_hash = benchmark.source_hashes(
        [args.checkpoint])[Path(args.checkpoint).name]
    cfg = e2e.SamplerConfig(num_steps=args.num_steps, grid="power",
                            langevin_steps=0)
    mean, std = backbone.terminal_moments()

    wpath = (Path(args.weights) if args.weights else
             Path(args.out).parent / "references" / args.task
             / f"mixture_{args.prior_id}" / f"obs_{args.obs_seed}_weights.json")
    # Validated, not just read: a failed estimate writes finite equal logits,
    # which would otherwise become a uniform component assignment here.
    log_pi, wblob = weights.load_weight_record(wpath, expected_k=atoms.num_atoms)
    # Component weights estimated under a different model are not the weights of
    # this model's posterior.  Estimator records carry the checkpoint they used;
    # reference records are model-independent and carry none.
    w_ckpt = wblob.get("checkpoint_sha256_16")
    if w_ckpt is not None and w_ckpt != ckpt_hash:
        raise SystemExit(
            f"{wpath} was estimated with checkpoint {w_ckpt}, but this run samples "
            f"with {ckpt_hash} ({args.checkpoint}); rerun the weights for this model")

    # A shared base draw used only for PriorGuide's empirical Sigma_post, as in
    # the other runners.
    base_cfg = e2e.SamplerConfig(num_steps=500, grid="uniform", langevin_steps=0)
    kb0, kb1 = jax.random.split(jax.random.PRNGKey(90000))
    n_base = 5000
    xb = e2e.draw_terminal(kb0, n_base, mean, std)
    db, lb = e2e.make_noise(kb1, base_cfg.num_steps, n_base, backbone.theta_dim, 0)
    hook_b, _, _ = hooks.make_base_hook(backbone, x_o)
    bank = e2e.reverse_sample(hook_b, schedule=backbone.schedule, cfg=base_cfg,
                              x_init=xb, diffusion_noise=db, langevin_noise=lb,
                              snapshot_fracs=()).samples
    total_var = float(np.var(np.asarray(bank)))

    out_root = Path(args.out)
    for seed in [int(v) for v in args.seeds.split(",")]:
        key = jax.random.PRNGKey((seed * 1_000_003 + args.obs_seed) % (2**31))
        k_init, k_noise = jax.random.split(key)
        x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
        diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples,
                                    backbone.theta_dim, 0)
        labels = hooks.draw_atom_labels(jax.random.PRNGKey(7 + seed),
                                        jnp.asarray(log_pi), args.num_samples)

        rows = {
            "base": hooks.make_base_hook(backbone, x_o),
            # PriorGuide and PG-FullCov get the true correlated ratio
            "pg_truecov": hooks.make_pg_hook(
                backbone, x_o, true_comps,
                reverse_cov=hooks.upstream_reverse_cov(total_var,
                                                       backbone.theta_dim)),
            "a_full_truecov": hooks.make_a_full_hook(backbone, x_o, true_comps),
            # Spectra uses the compiled dictionary
            "eamt_compiled_reference": hooks.make_spectra_hook(backbone, x_o, atoms,
                                                            labels),
        }
        for name, (hook, per_hook, hook_meta) in rows.items():
            t0 = time.perf_counter()
            res = e2e.reverse_sample(hook, schedule=backbone.schedule, cfg=cfg,
                                     x_init=x_init, diffusion_noise=diff,
                                     langevin_noise=lang,
                                     snapshot_fracs=e2e.DEFAULT_SNAPSHOT_FRACS)
            wall = time.perf_counter() - t0
            row_name = f"{name}_ms{seed}"
            out = (out_root / "samples" / args.task / row_name
                   / f"model_{args.model_id}" / f"mixture_{args.prior_id}"
                   / f"obs_{args.obs_seed}")
            out.mkdir(parents=True, exist_ok=True)
            tag = f"steps{args.num_steps}_L0_power"
            payload = {"samples": res.samples, "x_o": np.asarray(x_o),
                       "theta_true": np.asarray(theta_true), "times": res.times}
            if name.startswith("eamt"):
                payload["atom_labels"] = np.asarray(labels)
            np.savez_compressed(out / f"{tag}.npz", **payload)
            (out / f"{tag}.json").write_text(json.dumps({
                "round": "round7_compiled", "task": args.task,
                "prior_type": "mixture", "prior_id": args.prior_id,
                "obs_seed": args.obs_seed, "model_id": args.model_id,
                "method": row_name, "method_family": name,
                "method_meta": hook_meta, "conditioning": "conditional",
                "seed": seed,
                "sampler": cfg.as_dict(),
                "nfe": e2e.nfe_accounting(cfg, per_hook),
                "num_samples": args.num_samples,
                "wall_clock_s": wall,
                "wall_clock_per_1000_samples_s": wall * 1000.0 / args.num_samples,
                "nan_or_inf": int((~np.isfinite(res.samples)).sum()),
                "ratio_input": ("true correlated Gaussian (full covariance)"
                                if "truecov" in name else
                                ("compiled positive-atom dictionary, "
                                 f"K={atoms.num_atoms}" if name.startswith("eamt")
                                 else "none")),
                "compiled_prior": prior.as_dict(),
                "atoms": atoms.as_dict() if name.startswith("eamt") else None,
                "eamt_log_pi": log_pi.tolist() if name.startswith("eamt") else None,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256_16": ckpt_hash,
                "upstream_commit": UPSTREAM_COMMIT,
                "repo_git": git_state(REPO),
            }, indent=2, default=str))
            print(f"[seed {seed}][{row_name}] {wall:.1f}s nan="
                  f"{int((~np.isfinite(res.samples)).sum())} -> {out / (tag + '.npz')}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
