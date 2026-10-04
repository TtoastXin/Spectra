#!/usr/bin/env python
"""A BCI/BAV target-posterior reference by tempered SMC against the likelihood.

Tempered SMC is used in place of VBMC; PyVBMC can serve as an optional
cross-check.

The Gauss-Hermite order ``--gh-deg`` matters.  The order needed for an accurate
relative log-likelihood field depends on the region of parameter space: near
the strong target prior the field is stable to ~0.02 nats at order 301, while
at wide draws from the training prior the same comparison differs by tens of
nats.  A strong-prior K=1 posterior lies in the stable region, so the reference
is built for the target prior only.  ``--gh-deg`` is recorded; running two
orders and comparing the posteriors is the check.

The likelihood dominates the cost.  It is evaluated in batch on the GPU;
``--theta-chunk`` trades memory for throughput.  The tensor is
``(chunk, 98, n, n)`` in float64, so the chunk has to shrink as the order grows.

    python scripts/reference_bci.py --prior-type strong --prior-id 0 \\
        --obs-seed 1000000 --gh-deg 301 --particles 8000 \\
        --out results/single_factor_controlled_strong
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import bav, benchmark, coordinates, quadrature  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT  # noqa: E402
from spectra.slcp import tempered_smc  # noqa: E402
from spectra.references import seed_stability  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def gaussian_log_prior(mu: np.ndarray, sd: np.ndarray, pi: np.ndarray):
    """Log density of the shipped (possibly mixture) target prior, model coords."""
    mu = np.atleast_2d(np.asarray(mu, float))
    sd = np.atleast_2d(np.asarray(sd, float))
    pi = np.asarray(pi, float)
    log_pi = np.log(pi / pi.sum())

    def fn(theta):
        theta = np.atleast_2d(theta)
        parts = []
        for k in range(mu.shape[0]):
            z = (theta - mu[k]) / sd[k]
            parts.append(log_pi[k] - 0.5 * np.sum(z**2, axis=1)
                         - np.sum(np.log(sd[k]))
                         - 0.5 * theta.shape[1] * math.log(2 * math.pi))
        p = np.stack(parts, axis=0)
        m = p.max(axis=0)
        return m + np.log(np.exp(p - m).sum(axis=0))

    return fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior-type", default="strong")
    p.add_argument("--prior-id", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=1000000)
    p.add_argument("--gh-deg", type=int, default=301)
    p.add_argument("--theta-chunk", type=int, default=16)
    p.add_argument("--device", default="cuda",
                   help="torch backend only; the JAX backend uses whatever "
                        "JAX sees")
    p.add_argument("--backend", default="jax", choices=("jax", "torch"),
                   help="jax by default: the environment ships CPU-only "
                        "torch, so only the JAX path reaches the GPU")
    p.add_argument("--particles", type=int, default=8000)
    p.add_argument("--smc-seeds", default="0,1,2,3")
    p.add_argument("--smc-mcmc", type=int, default=12)
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    warnings.simplefilter("ignore")
    shift = build_prior_shift("bav", args.prior_type, args.prior_id)
    atoms = atoms_mod.atoms_from_shift(shift)
    theta_true, x_o = benchmark.load_observation(
        "bav", args.prior_type, args.prior_id, args.obs_seed)
    theta_true = np.asarray(theta_true, float)
    x_std = np.asarray(x_o, float).reshape(-1)
    raw = bav.unstandardise_responses(x_std)

    cfg = bav.LikelihoodConfig(gh_deg=args.gh_deg, theta_chunk=args.theta_chunk,
                               device=args.device)
    n_calls = {"n": 0}

    def log_lik(theta_model):
        theta_model = np.atleast_2d(theta_model)
        n_calls["n"] += theta_model.shape[0]
        phys = bav.model_to_physical(theta_model)
        if args.backend == "jax":
            return bav.log_likelihood_jax(phys, raw, gh_deg=args.gh_deg,
                                          theta_chunk=args.theta_chunk)
        return bav.log_likelihood(phys, raw, cfg=cfg)

    log_prior = gaussian_log_prior(shift.target_mu, shift.target_sigma,
                                   shift.target_pi)

    t0 = time.perf_counter()
    runs = []
    for s in [int(v) for v in args.smc_seeds.split(",")]:
        rng = np.random.default_rng(10_007 * s + args.obs_seed)

        def draw(num, _rng=rng):
            pi_d = np.asarray(shift.target_pi, float)
            k = _rng.choice(len(pi_d), size=num, p=pi_d / pi_d.sum())
            return (shift.target_mu[k]
                    + shift.target_sigma[k]
                    * _rng.standard_normal((num, shift.theta_dim)))

        res = tempered_smc(log_lik, log_prior, draw,
                           num_particles=args.particles, rng=rng,
                           num_mcmc=args.smc_mcmc, max_stages=400)
        res.seed = s
        runs.append(res)
        print(f"  smc[seed {s}] stages={res.diagnostics['num_stages']} "
              f"minESS={res.diagnostics['min_stage_ess']:.0f} "
              f"acc={res.diagnostics['mean_acceptance']:.3f} "
              f"uniq={res.diagnostics['unique_particles']} "
              f"logZ={res.log_evidence:+.4f}  "
              f"(likelihood calls so far {n_calls['n']:,})", flush=True)
    wall = time.perf_counter() - t0

    pooled = np.concatenate([r.samples for r in runs])
    rng = np.random.default_rng(args.seed * 1_000_003 + args.obs_seed)
    pooled = pooled[rng.permutation(pooled.shape[0])][:args.num_samples]

    resp = quadrature.responsibility_masses_from_samples(
        pooled, atoms, num_shards=len(runs))
    audit = coordinates.audit("bav", shift, atoms)

    payload = {"samples": pooled, "x_o": x_std,
               "theta_true": theta_true}
    for r in runs:
        payload[f"samples_seed_{r.seed}"] = r.samples

    stab = seed_stability([r.samples for r in runs])
    cert = {
        "grade": "stability_reported",
        "method": (f"adaptive tempered SMC against the batched BAV "
                   f"Gauss-Hermite likelihood, order {args.gh_deg}, float64"),
        "gh_deg": args.gh_deg,
        "quadrature_rule": "scipy.special.roots_hermite",
        "backend": args.backend,
        "particles": args.particles, "smc_seeds": args.smc_seeds,
        "num_mcmc_per_stage": args.smc_mcmc,
        "target_stability": stab,
        "target_log_evidence_per_seed": [r.log_evidence for r in runs],
        "target_diagnostics": [r.diagnostics for r in runs],
        "target_mean": pooled.mean(0).tolist(),
        "target_cov": np.cov(pooled.T).tolist(),
        "pi_reference": resp["pi"].tolist(),
        "likelihood_evaluations": n_calls["n"],
        "wall_clock_s": wall,
        "seconds_per_likelihood_evaluation": wall / max(n_calls["n"], 1),
        "reference_is_target_only": (
            "the training-prior posterior is not built this way: "
            "at draws from N(0, I) the relative log-likelihood field is unstable "
            "by tens of nats at every affordable order, so a likelihood-based "
            "base reference could not be validated"),
    }

    out_dir = (Path(args.out) / "references" / "bav"
               / f"{args.prior_type}_{args.prior_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"obs_{args.obs_seed}.npz"
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(json.dumps({
        "task": "bav", "prior_type": args.prior_type, "prior_id": args.prior_id,
        "obs_seed": args.obs_seed, "kind": "smc",
        "num_samples": int(pooled.shape[0]), "seed": args.seed,
        "prior_json_sha256_16": benchmark.prior_json_hash(
            "bav", args.prior_type, args.prior_id),
        "coordinate_audit": audit,
        "certificate": cert,
        "upstream_commit": UPSTREAM_COMMIT,
        "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
    }, indent=2, default=str))
    print(f"[bav/{args.prior_type}_{args.prior_id}/obs_{args.obs_seed}] "
          f"gh={args.gh_deg} meangap={stab.get('max_standardised_mean_gap')} "
          f"{n_calls['n']:,} likelihood evals in {wall:.0f}s -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
