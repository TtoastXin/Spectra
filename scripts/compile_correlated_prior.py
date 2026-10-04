#!/usr/bin/env python
"""Compile an out-of-native correlated Gaussian prior into positive atoms.

The other studies use priors that are already isotropic Gaussian mixtures, i.e.
inside the exact-atom family.  This one takes a correlated Gaussian
localisation, which is outside it, and separates three questions:

1. how well a prior-only dictionary approximates the ratio
   (``eps_r`` and the resulting ``TV(q, q~) <= eps_r / Z`` bound);
2. how far the compiled target is from the true target, computed exactly on the
   same grid the reference uses;
3. what the compiled atoms cost in atom mass and component transport.

Layers 1 and 2 need no checkpoint and no sampler, so they run here; layer 3's
end-to-end rows use the ordinary Spectra runner, because the compiled
dictionary is written out as an ordinary isotropic mixture prior (at ``K = 27``
and ``K = 64``).

The dictionary is determined by the prior, the model coordinates, a fixed
``lambda / lambda_min(Sigma)`` ratio and a fixed node-density criterion.  No
observation, posterior or metric enters its construction.

    python scripts/compile_correlated_prior.py --task two_moons --obs-seed 1000003 \\
        --donor-prior-id 2 --grid-n 4096 --local-data data \\
        --out results/correlated_prior/compilation
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, compile_atoms, quadrature  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.references import support_matched  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# prior ids outside the shipped 0..9 range so nothing upstream is shadowed
CAPACITY_PRIOR_ID = {"small": 20, "medium": 21}


def grid_posterior_from_log_prior(task, x_o, log_prior_vals, box, n, log_lik,
                                  centers, area):
    """Normalised posterior for an arbitrary prior evaluated on the grid."""
    log_joint = np.asarray(log_lik, float) + np.asarray(log_prior_vals, float)
    lse = quadrature._logsumexp(log_joint)
    post = quadrature.GridPosterior(
        task=task, kind="custom", box=(np.asarray(box[0], float),
                                       np.asarray(box[1], float)),
        n=n, centers=centers, log_w=log_joint - lse,
        log_norm=lse + math.log(area), cell_area=area)
    post.diagnostics["edge_mass"] = post.edge_mass()
    return post


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="two_moons",
                   choices=sorted(quadrature.LOG_LIKELIHOOD))
    p.add_argument("--obs-seed", type=int, default=1000003)
    p.add_argument("--donor-prior-id", type=int, default=2,
                   help="whose shipped observation is reused; the compiled "
                        "prior is observation-blind, so theta_true belongs to "
                        "the donor prior and only x_o is used")
    p.add_argument("--grid-n", type=int, default=4096)
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--local-data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    local = Path(args.local_data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    box = benchmark.training_support(args.task)
    if box is None:
        raise SystemExit(f"{args.task} has no bounded training prior")
    prior = compile_atoms.design_prior((np.asarray(box[0], float),
                                        np.asarray(box[1], float)))
    _, x_o = benchmark.load_observation(args.task, "mixture",
                                        args.donor_prior_id, args.obs_seed)
    x_o = np.asarray(x_o, float)

    # grid shared by every layer
    grid_box = (np.asarray(box[0], float) - 0.3, np.asarray(box[1], float) + 0.3)
    centers, area = quadrature._grid(grid_box, args.grid_n)
    ll = quadrature.log_likelihood(args.task, centers, x_o)
    log_r_true = prior.log_prob(centers)
    true_post = grid_posterior_from_log_prior(args.task, x_o, log_r_true,
                                              grid_box, args.grid_n, ll,
                                              centers, area)

    report = {
        "task": args.task, "obs_seed": args.obs_seed,
        "donor_prior_id": args.donor_prior_id,
        "observation_note": ("x_o is reused from the donor cell; theta_true was "
                             "drawn from the donor prior, so rmse_to_true_theta "
                             "is not meaningful for the compiled rows"),
        "prior": prior.as_dict(),
        "prior_mass_inside_training_box": prior.mass_inside_box(),
        "grid_n": args.grid_n,
        "grid_box_low": grid_box[0].tolist(), "grid_box_high": grid_box[1].tolist(),
        "true_target_edge_mass": true_post.diagnostics["edge_mass"],
        "true_target_mean": true_post.mean.tolist(),
        "true_target_cov": true_post.cov.tolist(),
        "capacities": {},
        "naive_isotropic_tensor_baseline": {},
    }

    # baseline: the naive rule with one node order for all axes
    for order in (3, 6):
        d0 = compile_atoms.compile_dictionary(prior, spread_factor=0.0,
                                              isotropic_tensor_order=order)
        e0 = compile_atoms.approximation_error(d0, num=200_000)
        report["naive_isotropic_tensor_baseline"][f"order_{order}"] = {
            "num_atoms": d0.num_atoms,
            "eps_r_over_Z_bound_uniform": e0["eps_r_over_Z_bound_uniform"],
        }

    rng = np.random.default_rng(args.obs_seed)
    np.savez_compressed(out / f"true_target_{args.task}_obs{args.obs_seed}.npz",
                        samples=true_post.sample(rng, args.num_samples),
                        x_o=x_o, log_norm=np.array(true_post.log_norm))

    for name, sf in compile_atoms.CAPACITIES.items():
        d = compile_atoms.compile_dictionary(prior, spread_factor=sf)
        err = compile_atoms.approximation_error(d, num=200_000)
        pid = CAPACITY_PRIOR_ID[name]

        log_r_comp = d.log_prob(centers)
        comp_post = grid_posterior_from_log_prior(args.task, x_o, log_r_comp,
                                                  grid_box, args.grid_n, ll,
                                                  centers, area)
        tv = 0.5 * float(np.sum(np.abs(np.exp(true_post.log_w)
                                       - np.exp(comp_post.log_w))))

        # the compiled dictionary, written as an ordinary isotropic mixture so
        # the standard runners consume it unchanged
        pdir = local / "priors" / args.task
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"mixture_{pid}.json").write_text(
            json.dumps(d.as_prior_json(args.task), indent=2))
        odir = local / "observations" / args.task / f"mixture_{pid}"
        odir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            benchmark.observation_path(args.task, "mixture",
                                       args.donor_prior_id, args.obs_seed),
            odir / f"obs_{args.obs_seed}.json")

        # reference masses for the compiled atoms, exact on the grid
        shift = build_prior_shift(args.task, "mixture", pid)
        atoms = atoms_mod.atoms_from_shift(shift)
        base_post = quadrature.grid_posterior(
            args.task, x_o, shift, kind="base", box=grid_box, n=args.grid_n,
            log_lik=ll, centers=centers, cell_area=area)
        lz = quadrature.atom_evidence_on_grid(base_post, atoms)
        resp = quadrature.responsibility_masses(comp_post, atoms)
        lp = atoms_mod.log_phi(base_post.centers, atoms)
        eff = np.array([math.exp(2.0 * base_post.log_expectation(lp[:, k])
                                 - base_post.log_expectation(2.0 * lp[:, k]))
                        for k in range(atoms.num_atoms)])

        ref_dir = out.parent / "references" / args.task / f"mixture_{pid}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        samples = comp_post.sample(np.random.default_rng(args.obs_seed + pid),
                                   args.num_samples)
        payload = {"samples": samples, "x_o": x_o,
                   "theta_true": np.zeros(centers.shape[1]),
                   "samples_training_prior": base_post.sample(
                       np.random.default_rng(args.obs_seed + pid + 1),
                       args.num_samples),
                   "samples_support_matched": support_matched(samples, box)}
        np.savez_compressed(ref_dir / f"obs_{args.obs_seed}.npz", **payload)
        (ref_dir / f"obs_{args.obs_seed}.json").write_text(json.dumps({
            "kind": "compiled_target_quadrature", "capacity": name,
            "num_atoms": d.num_atoms, "grid_n": args.grid_n,
            "note": "reference for the COMPILED target, not the true target",
        }, indent=2))
        lpn = np.asarray(resp["log_pi"], float)
        (ref_dir / f"obs_{args.obs_seed}_weights.json").write_text(json.dumps({
            "log_pi": np.where(np.isfinite(lpn), lpn, -700.0).tolist(),
            "pi": resp["pi"].tolist(),
            "source": f"round7_compiled_reference_{name}",
            "num_atoms": atoms.num_atoms,
            "task": args.task, "prior_type": "mixture", "prior_id": pid,
            "obs_seed": args.obs_seed,
        }, indent=2))

        report["capacities"][name] = {
            "prior_id": pid, "num_atoms": d.num_atoms,
            "spread_factor": sf, "lambda": d.lam,
            "sqrt_lambda": math.sqrt(d.lam), "c_ratio": d.c_ratio,
            "min_weight": float(np.exp(d.log_b).min()),
            "prior_only_error": err,
            "tv_true_vs_compiled_target": tv,
            "tv_bound_from_eps_r": err["eps_r_over_Z_bound_uniform"],
            "compiled_target_edge_mass": comp_post.diagnostics["edge_mass"],
            "compiled_target_mean": comp_post.mean.tolist(),
            "mean_gap_true_vs_compiled": float(
                np.linalg.norm(comp_post.mean - true_post.mean)),
            "reference_pi_entropy_nats": float(
                -np.sum(resp["pi"] * np.log(np.maximum(resp["pi"], 1e-300)))),
            "reference_pi_max": float(resp["pi"].max()),
            "reference_pi_effective_atoms": float(np.exp(
                -np.sum(resp["pi"] * np.log(np.maximum(resp["pi"], 1e-300))))),
            "predicted_direct_ess_at_5000_min": float(5000.0 * eff.min()),
            "predicted_direct_ess_at_5000_at_dominant_atom": float(
                5000.0 * eff[int(np.argmax(resp["pi"]))]),
            "log_z_range": [float(lz["log_z"].min()), float(lz["log_z"].max())],
        }
        print(f"[{name}] K={d.num_atoms} sqrt(lam)={math.sqrt(d.lam):.4f} "
              f"eps_r/Z={err['eps_r_over_Z_bound_uniform']:.5f} "
              f"TV(true, compiled)={tv:.5f} "
              f"mean_gap={report['capacities'][name]['mean_gap_true_vs_compiled']:.5f} "
              f"eff_atoms={report['capacities'][name]['reference_pi_effective_atoms']:.2f} "
              f"directESS@5000={report['capacities'][name]['predicted_direct_ess_at_5000_at_dominant_atom']:.1f}",
              flush=True)

    report["repo_git"] = git_state(REPO)
    report["source_sha256_16"] = benchmark.source_hashes(
        [Path(__file__), *sorted((REPO / "spectra").glob("*.py"))])
    (out / f"compilation_{args.task}_obs{args.obs_seed}.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"wrote -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
