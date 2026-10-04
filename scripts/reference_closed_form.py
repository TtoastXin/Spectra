#!/usr/bin/env python
"""Build reference posterior banks in closed form, using numpy only (no JAX or torch).

Gaussian Linear 10D/20D use the closed form; Two Moons uses rejection sampling
with the test prior as proposal.  Every bank carries its own provenance, and the
uniform-training tasks additionally carry the support diagnostics: the prior mass
and the reference-posterior mass outside ``supp(pi_train)``, plus a
support-matched bank against which structured transport is exact.

    python scripts/reference_closed_form.py --task gaussian_linear \\
        --prior-type strong --prior-id 0 --obs-seeds all \\
        --num-samples 10000 --out results/single_factor_controlled_strong
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.references import (  # noqa: E402
    GL_PRIOR_SCALE,
    GL_SIMULATOR_SCALE,
    gaussian_linear_base_posterior,
    gaussian_linear_posterior,
    outside_box_fraction,
    support_matched,
    two_moons_reference,
)
from spectra.simformer import UPSTREAM_COMMIT  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ANALYTIC_TASKS = ("gaussian_linear", "gaussian_linear_high")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--prior-type", required=True, choices=benchmark.PRIOR_TYPES)
    p.add_argument("--prior-id", type=int, required=True)
    p.add_argument("--obs-seeds", default="all")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.task not in ANALYTIC_TASKS + ("two_moons",):
        p.error(
            f"{args.task} has no closed-form reference: OUP and SLCP references come "
            "from scripts/reference_posterior.py, BCI ones from scripts/reference_bci.py"
        )
    seeds = (
        list(benchmark.OBSERVATION_SEEDS) if args.obs_seeds == "all"
        else [int(s) for s in args.obs_seeds.split(",")]
    )
    shift = build_prior_shift(args.task, args.prior_type, args.prior_id)
    out_root = Path(args.out) / "references" / args.task / \
        f"{args.prior_type}_{args.prior_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    for obs_seed in seeds:
        theta_true, x_o = benchmark.load_observation(
            args.task, args.prior_type, args.prior_id, obs_seed)
        x_o = np.asarray(x_o, float)
        rng_seed = (args.seed * 1_000_003 + obs_seed) % (2**31)
        t0 = time.perf_counter()
        meta = {
            "task": args.task, "prior_type": args.prior_type,
            "prior_id": args.prior_id, "obs_seed": obs_seed,
            "num_samples": args.num_samples, "seed": rng_seed,
            "prior_json_sha256_16": benchmark.prior_json_hash(
                args.task, args.prior_type, args.prior_id),
            "upstream_commit": UPSTREAM_COMMIT,
            "repo_git": git_state(REPO),
        }

        if args.task in ANALYTIC_TASKS:
            post = gaussian_linear_posterior(x_o, shift)
            rng = np.random.default_rng(rng_seed)
            samples = post.sample(rng, args.num_samples)
            base_post = gaussian_linear_base_posterior(x_o, shift)
            payload = {
                "samples": samples, "x_o": x_o, "theta_true": np.asarray(theta_true),
                "post_weights": post.weights, "post_means": post.means,
                "post_variances": post.variances,
                "post_mean": post.mean, "post_cov": post.cov,
                # the backbone's own target: posterior under the training prior
                "samples_training_prior": base_post.sample(
                    np.random.default_rng(rng_seed + 1), args.num_samples),
                "training_prior_post_mean": base_post.mean,
                "training_prior_post_cov": base_post.cov,
            }
            meta.update({
                "kind": "analytic",
                "prior_scale": GL_PRIOR_SCALE,
                "simulator_scale": GL_SIMULATOR_SCALE,
                "analytic_mean": post.mean.tolist(),
                "analytic_cov_diag": np.diag(post.cov).tolist(),
                "num_components": int(post.weights.shape[0]),
                "component_weights": post.weights.tolist(),
                "sample_mean_error": float(
                    np.linalg.norm(samples.mean(0) - post.mean)),
                "training_prior_post_mean": base_post.mean.tolist(),
                "training_prior_post_cov_diag": np.diag(base_post.cov).tolist(),
            })
        else:
            ref = two_moons_reference(x_o, shift, args.num_samples, rng_seed)
            samples = ref.samples
            inside = support_matched(samples, shift.train_box)
            # the backbone's own target: the posterior under the uniform training
            # prior, which is what the unguided base Simformer row should be
            # scored against
            base_ref = two_moons_reference(
                x_o, shift, args.num_samples, rng_seed + 1, proposal="training")
            payload = {
                "samples": samples, "x_o": x_o, "theta_true": np.asarray(theta_true),
                "samples_support_matched": inside,
                "samples_training_prior": base_ref.samples,
            }
            meta.update({
                "kind": "rejection",
                "proposals": ref.proposals, "accepted": ref.accepted,
                "acceptance_rate": ref.acceptance_rate,
                "likelihood_bound": ref.bound,
                "upstream_likelihood_bound": ref.upstream_bound,
                "max_acceptance_ratio": ref.max_acceptance_ratio,
                "upstream_mixture_weight_override": False,
                "training_prior_reference": {
                    "proposals": base_ref.proposals, "accepted": base_ref.accepted,
                    "acceptance_rate": base_ref.acceptance_rate,
                    "max_acceptance_ratio": base_ref.max_acceptance_ratio,
                    "seed": base_ref.seed,
                },
                "support": {
                    "training_box_low": shift.train_box[0].tolist(),
                    "training_box_high": shift.train_box[1].tolist(),
                    "prior_mass_outside": shift.target_mass_outside_training_box(),
                    "posterior_mass_outside": outside_box_fraction(
                        samples, shift.train_box),
                    "support_matched_size": int(inside.shape[0]),
                },
            })
        meta["wall_clock_s"] = time.perf_counter() - t0
        path = out_root / f"obs_{obs_seed}.npz"
        np.savez_compressed(path, **payload)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"[{args.task}/{args.prior_type}_{args.prior_id}/obs_{obs_seed}] "
              f"{samples.shape[0]} samples -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
