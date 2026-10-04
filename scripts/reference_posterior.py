#!/usr/bin/env python
"""Certified reference posteriors for the benchmark cells.

Two routes, chosen by what the task admits:

``--method quadrature`` (Two Moons, OUP)
    two-dimensional parameter plus a tractable likelihood, so the posterior is
    integrated on a tensor grid.  The samples for C2ST, the normaliser, the atom
    evidences ``Z_k``, the bounded target responsibilities
    ``pi_k = E_q[gamma_k]`` and the per-sample efficiency of the direct bank
    plug-in all come from the same grid with no Monte-Carlo error; the
    refinement drift is reported as the remaining approximation.

``--method smc`` (SLCP)
    five dimensions, exact but unbounded likelihood (``s_1 = theta_3^2 -> 0``
    makes the density diverge, so no rejection envelope exists) and four
    sign-symmetric modes.  Adaptive tempered SMC with random-walk rejuvenation,
    run from several independent seeds so the between-run spread measures the
    reference's own uncertainty.

Both write the standard reference layout (``samples``,
``samples_training_prior``, ``samples_support_matched``) that the metric
scripts read, plus an ``obs_<seed>_weights.json`` in the ``log_pi`` format the
Spectra runners consume.  The ``certificate`` entry of the ``.json`` sidecar
carries the grade and its evidence.

    python scripts/reference_posterior.py --task oup --prior-type mixture \\
        --prior-id 3 --obs-seed 1000000 --method quadrature \\
        --out results/mixture_controlled
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, coordinates, quadrature  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.references import (  # noqa: E402
    seed_stability, support_matched)
from spectra.simformer import UPSTREAM_COMMIT  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOG_ZERO = -700.0


def write_weights(path: Path, atoms, log_pi, *, source: str, extra: dict) -> None:
    lp = np.asarray(log_pi, float)
    lp = np.where(np.isfinite(lp), lp, LOG_ZERO)
    path.write_text(json.dumps({
        "log_pi": lp.tolist(), "pi": np.exp(lp).tolist(),
        "source": source, "num_atoms": atoms.num_atoms, **extra,
    }, indent=2, default=str))


def run_quadrature(args, shift, atoms, x_o, theta_true, out_dir: Path) -> dict:
    task = args.task
    box = quadrature.default_box(task, shift, pad_sd=args.pad_sd)
    t0 = time.perf_counter()

    levels = [int(v) for v in args.levels.split(",")]
    conv = quadrature.convergence_check(task, x_o, shift, kind="target",
                                        box=box, levels=levels)
    n = levels[-1]
    centers, area = quadrature._grid(box, n)
    ll = quadrature.log_likelihood(task, centers, x_o)
    base = quadrature.grid_posterior(task, x_o, shift, kind="base", box=box, n=n,
                                     log_lik=ll, centers=centers, cell_area=area)
    tgt = quadrature.grid_posterior(task, x_o, shift, kind="target", box=box, n=n,
                                    log_lik=ll, centers=centers, cell_area=area)
    tgt_r = quadrature.grid_posterior(task, x_o, shift, kind="target_restricted",
                                      box=box, n=n, log_lik=ll, centers=centers,
                                      cell_area=area)

    rng = np.random.default_rng(args.seed * 1_000_003 + args.obs_seed)
    samples = tgt.sample(rng, args.num_samples)
    base_samples = base.sample(np.random.default_rng(
        args.seed * 1_000_003 + args.obs_seed + 1), args.num_samples)

    lz = quadrature.atom_evidence_on_grid(base, atoms)
    resp = quadrature.responsibility_masses(tgt_r, atoms)
    resp_un = quadrature.responsibility_masses(tgt, atoms)
    # the same bounded estimator on the drawn samples, so the sampling error of
    # a sample-based responsibility reference can be compared against the exact
    # value on a task where the exact value is known
    resp_s = quadrature.responsibility_masses_from_samples(samples, atoms,
                                                           num_shards=4)
    lp = atoms_mod.log_phi(base.centers, atoms)
    eff = np.array([math.exp(2.0 * base.log_expectation(lp[:, k])
                             - base.log_expectation(2.0 * lp[:, k]))
                    for k in range(atoms.num_atoms)])

    payload = {
        "samples": samples, "x_o": x_o, "theta_true": np.asarray(theta_true, float),
        "samples_training_prior": base_samples,
        "grid_centers_shape": np.array(base.centers.shape),
    }
    if shift.train_box is not None:
        payload["samples_support_matched"] = support_matched(samples,
                                                             shift.train_box)
    cert = {
        "grade": "analytic_grade_quadrature",
        "method": "exact-likelihood tensor-grid quadrature, model coordinates",
        "grid_n": n, "levels": levels, "pad_sd": args.pad_sd,
        "box_low": box[0].tolist(), "box_high": box[1].tolist(),
        "refinement": conv["last_refinement"],
        "refinement_levels": conv["levels"],
        "base_edge_mass": base.diagnostics["edge_mass"],
        "target_edge_mass": tgt.diagnostics["edge_mass"],
        "identity_gap_bZ_vs_responsibility": lz["log_odds"] - resp["log_odds"],
        "log_odds_bZ": lz["log_odds"],
        "log_odds_responsibility_restricted": resp["log_odds"],
        "log_odds_responsibility_round3_target": resp_un["log_odds"],
        "log_odds_responsibility_from_samples": resp_s["log_odds"],
        "log_odds_sample_shard_se": resp_s["log_odds_se"],
        "plugin_efficiency": eff.tolist(),
        "predicted_direct_ess_at_5000": (5000.0 * eff).tolist(),
        "log_z": lz["log_z"].tolist(),
        "pi_reference": resp["pi"].tolist(),
        "target_mean": tgt.mean.tolist(),
        "target_cov": tgt.cov.tolist(),
        "base_mean": base.mean.tolist(),
        "base_cov": base.cov.tolist(),
        "posterior_mass_outside_training_box": (
            float(1.0 - math.exp(tgt_r.log_norm - tgt.log_norm))
            if shift.train_box is not None else 0.0),
        "prior_mass_outside_training_box": (
            shift.target_mass_outside_training_box()
            if shift.train_box is not None else 0.0),
        "wall_clock_s": time.perf_counter() - t0,
    }

    if args.compare_rejection and task == "two_moons":
        from spectra.references import two_moons_reference
        ref = two_moons_reference(x_o, shift, args.num_samples,
                                  args.seed * 1_000_003 + args.obs_seed)
        cert["rejection_cross_check"] = {
            "mean_gap": float(np.linalg.norm(ref.samples.mean(0) - tgt.mean)),
            "cov_frobenius_gap": float(np.linalg.norm(
                np.cov(ref.samples.T) - tgt.cov)),
            "acceptance_rate": ref.acceptance_rate,
            "num_samples": int(ref.samples.shape[0]),
            "note": "rejection is exact in distribution; a gap of the size of "
                    "its own Monte-Carlo error confirms the grid",
        }
    return {"payload": payload, "certificate": cert,
            "log_pi_reference": resp["log_pi"].tolist(),
            "weights_extra": {
                "delta_star": float(lz["log_z"][0] - lz["log_z"][1])
                if atoms.num_atoms == 2 else float("nan"),
                "log_b_diff": float(atoms.log_b[0] - atoms.log_b[1])
                if atoms.num_atoms == 2 else float("nan"),
                "log_odds": resp["log_odds"],
                "grade": cert["grade"],
            }}


def run_smc(args, shift, atoms, x_o, theta_true, out_dir: Path) -> dict:
    from spectra import slcp

    t0 = time.perf_counter()
    box_prior = slcp.box_log_prior()
    mix_prior = slcp.mixture_log_prior(
        benchmark.load_prior_json(args.task, args.prior_type, args.prior_id))

    def lik(theta):
        return slcp.log_likelihood(theta, x_o)

    runs = {}
    for kind, log_prior, sampler in (
        ("target", mix_prior, "prior_mixture"),
        ("base", box_prior, "uniform_box"),
    ):
        per_seed = []
        for s in [int(v) for v in args.smc_seeds.split(",")]:
            rng = np.random.default_rng(10_007 * s + args.obs_seed + (0 if kind == "target" else 1))

            def draw(num, _rng=rng, _kind=kind):
                if _kind == "base":
                    return _rng.uniform(-1.0, 1.0, size=(num, shift.theta_dim))
                pi_d = np.asarray(shift.target_pi, float)
                k = _rng.choice(len(pi_d), size=num, p=pi_d / pi_d.sum())
                return (shift.target_mu[k]
                        + shift.target_sigma[k] * _rng.standard_normal(
                            (num, shift.theta_dim)))

            res = slcp.tempered_smc(lik, log_prior, draw,
                                    num_particles=args.particles, rng=rng,
                                    num_mcmc=args.smc_mcmc, max_stages=800,
                                    symmetry_move=slcp.sign_flip_move)
            res.seed = s
            per_seed.append(res)
            print(f"  smc[{kind} seed {s}] stages={res.diagnostics['num_stages']} "
                  f"minESS={res.diagnostics['min_stage_ess']:.0f} "
                  f"acc={res.diagnostics['mean_acceptance']:.3f} "
                  f"uniq={res.diagnostics['unique_particles']} "
                  f"beta={res.diagnostics['final_beta']:.4f} "
                  f"flip={res.diagnostics['mean_flip_acceptance']:.3f} "
                  f"logZ={res.log_evidence:+.4f}", flush=True)
        runs[kind] = per_seed

    tgt_samples = np.concatenate([r.samples for r in runs["target"]])
    base_samples = np.concatenate([r.samples for r in runs["base"]])
    rng = np.random.default_rng(args.seed * 1_000_003 + args.obs_seed)
    tgt_samples = tgt_samples[rng.permutation(tgt_samples.shape[0])][:args.num_samples]
    base_samples = base_samples[
        np.random.default_rng(args.seed + 1).permutation(base_samples.shape[0])
    ][:args.num_samples]

    resp = quadrature.responsibility_masses_from_samples(
        tgt_samples, atoms, num_shards=len(runs["target"]))
    per_run = [quadrature.responsibility_masses_from_samples(r.samples, atoms,
                                                             num_shards=2)
               for r in runs["target"]]
    run_log_odds = [float(r["log_odds"]) for r in per_run]
    plugin = atoms_mod.evidence_plugin(base_samples, atoms)

    payload = {
        "samples": tgt_samples, "x_o": x_o,
        "theta_true": np.asarray(theta_true, float),
        "samples_training_prior": base_samples,
    }
    # per-seed blocks so the metrics stage can measure the reference's own
    # resolution (reference-vs-reference C2ST against a split-half null)
    for r in runs["target"]:
        payload[f"samples_seed_{r.seed}"] = r.samples
    for r in runs["base"]:
        payload[f"samples_training_prior_seed_{r.seed}"] = r.samples
    if shift.train_box is not None:
        payload["samples_support_matched"] = support_matched(tgt_samples,
                                                             shift.train_box)
    spread = (float(np.std(run_log_odds, ddof=1)) if len(run_log_odds) > 1
              else float("nan"))
    if atoms.num_atoms == 2:
        grade = ("exact_or_high_confidence"
                 if (len(run_log_odds) > 1 and spread < 0.1)
                 else "high_quality_approximate")
    else:
        # ``run_log_odds`` is a two-atom quantity (NaN at K=1), so single-factor
        # references are graded downstream against a reference-vs-reference C2ST
        # and its split-half null instead.
        grade = "stability_reported"
    sym = [float(np.mean(r.samples[:, 2] > 0)) for r in runs["base"]]
    cert = {
        "grade": grade,
        "method": ("adaptive tempered SMC against the exact SLCP likelihood, "
                   "with the exact sign-flip mode-jumping move"),
        "base_sign_symmetry_frac_theta3": sym,
        "base_symmetry_error": float(max(abs(v - 0.5) for v in sym)),
        "particles": args.particles, "smc_seeds": args.smc_seeds,
        "num_mcmc_per_stage": args.smc_mcmc,
        "run_log_odds": run_log_odds,
        "run_log_odds_spread_sd": spread,
        "log_odds_pooled": resp["log_odds"],
        "log_odds_shard_se": resp["log_odds_se"],
        "pi_reference": resp["pi"].tolist(),
        "target_log_evidence_per_seed": [r.log_evidence for r in runs["target"]],
        "base_log_evidence_per_seed": [r.log_evidence for r in runs["base"]],
        "target_diagnostics": [r.diagnostics for r in runs["target"]],
        "base_diagnostics": [r.diagnostics for r in runs["base"]],
        "direct_plugin_on_base_reference": {
            "log_z": plugin["log_z"].tolist(), "ess": plugin["ess"].tolist(),
            "log_odds": atoms_mod.log_odds(atoms, plugin["log_z"]),
        },
        "target_stability": seed_stability([r.samples for r in runs["target"]]),
        "base_stability": seed_stability([r.samples for r in runs["base"]]),
        "target_mean": tgt_samples.mean(0).tolist(),
        "target_cov": np.cov(tgt_samples.T).tolist(),
        "base_mean": base_samples.mean(0).tolist(),
        "prior_mass_outside_training_box": (
            shift.target_mass_outside_training_box()
            if shift.train_box is not None else 0.0),
        "wall_clock_s": time.perf_counter() - t0,
    }
    return {"payload": payload, "certificate": cert,
            "log_pi_reference": np.log(resp["pi"]).tolist(),
            "weights_extra": {"log_odds": resp["log_odds"], "grade": grade,
                              "log_b_diff": float(atoms.log_b[0] - atoms.log_b[1])
                              if atoms.num_atoms == 2 else float("nan")}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--prior-type", default="mixture")
    p.add_argument("--prior-id", type=int, required=True)
    p.add_argument("--obs-seed", type=int, required=True)
    p.add_argument("--method", required=True, choices=("quadrature", "smc"))
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--levels", default="1024,2048,4096")
    p.add_argument("--pad-sd", type=float, default=6.0)
    p.add_argument("--compare-rejection", action="store_true")
    p.add_argument("--particles", type=int, default=20000)
    p.add_argument("--smc-seeds", default="0,1,2")
    p.add_argument("--smc-mcmc", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    shift = build_prior_shift(args.task, args.prior_type, args.prior_id)
    atoms = atoms_mod.atoms_from_shift(shift)
    theta_true, x_o = benchmark.load_observation(
        args.task, args.prior_type, args.prior_id, args.obs_seed)
    x_o = np.asarray(x_o, float)

    audit = coordinates.audit(args.task, shift, atoms)
    if audit["status"] != "native_exact":
        print(f"WARNING coordinate check status = {audit['status']}", flush=True)

    out_dir = (Path(args.out) / "references" / args.task
               / f"{args.prior_type}_{args.prior_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = run_quadrature if args.method == "quadrature" else run_smc
    res = runner(args, shift, atoms, x_o, theta_true, out_dir)

    path = out_dir / f"obs_{args.obs_seed}.npz"
    np.savez_compressed(path, **res["payload"])
    meta = {
        "task": args.task, "prior_type": args.prior_type,
        "prior_id": args.prior_id, "obs_seed": args.obs_seed,
        "kind": args.method, "num_samples": args.num_samples, "seed": args.seed,
        "prior_json_sha256_16": benchmark.prior_json_hash(
            args.task, args.prior_type, args.prior_id),
        "coordinate_audit": audit,
        "certificate": res["certificate"],
        "upstream_commit": UPSTREAM_COMMIT,
        "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))]),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    write_weights(out_dir / f"obs_{args.obs_seed}_weights.json", atoms,
                  res["log_pi_reference"],
                  source=f"round7_reference_{args.method}",
                  extra={"task": args.task, "prior_type": args.prior_type,
                         "prior_id": args.prior_id, "obs_seed": args.obs_seed,
                         **res["weights_extra"]})
    c = res["certificate"]
    print(f"[{args.task}/{args.prior_type}_{args.prior_id}/obs_{args.obs_seed}] "
          f"grade={c['grade']} log_odds={c.get('log_odds_bZ', c.get('log_odds_pooled')):+.6f} "
          f"-> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
