#!/usr/bin/env python
"""Anisotropy study: exact target vs compiled target, on the grid, no sampler.

For one observation and every cell, on a tensor grid over the training box
padded to cover every target's bulk (``mu +/- 6 sd`` per axis, as
``quadrature.default_box``), refined 2048 -> 4096:

* exact target prior ``N(mu, Sigma_gamma)`` and the compiled dictionary ``r~``;
* prior TV between them, both on the full grid and restricted to the training
  box (``support_matched``, each renormalised there);
* approximation-induced posterior TV between ``p_0 * r`` and ``p_0 * r~``
  (Two Moons exact likelihood, uniform training prior), full and
  ``support_matched``;
* moments / edge mass of the exact target posterior.

It also writes the exact-target reference bank for every anisotropic cell in the
standard layout (``references/two_moons/aniso_<id>/obs_<seed>.npz`` with
``samples``, ``samples_support_matched``, ``theta_true``), drawn from the
finest grid with ``reference_posterior.py``'s seed convention, so
``compute_metrics.py`` scores every anisotropy row against the exact target
unchanged.  The ``gamma = 1`` control's reference is ``reference_posterior.py`` on
``mild_20`` (the shift-severity protocol); cell 60 is evaluated here only as a
cross-check of that grid.

    python scripts/anisotropy_representation.py --obs-seed 1000000
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

from spectra import benchmark, compile_atoms, quadrature  # noqa: E402
from spectra.references import support_matched  # noqa: E402
from spectra.utils import git_state  # noqa: E402
from scripts import anisotropy_config as G  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PAD_SD = 6.0
CHUNK = 1 << 20


def exact_log_prior(cell: dict, pts: np.ndarray) -> np.ndarray:
    prior = compile_atoms.CorrelatedGaussianPrior(
        mu=np.asarray(cell["mu"], float), cov=np.asarray(cell["cov"], float),
        box=None, rule="")
    return prior.log_prob(pts)


def compiled_log_prior(prior_json: dict, pts: np.ndarray) -> np.ndarray:
    """Chunked ``log sum_k b_k N(theta; u_k, lambda I)``; exact, just memory-bounded."""
    mu = np.asarray(prior_json["mu"], float)
    lam = float(prior_json["compilation"]["lambda"])
    log_b = np.log(np.asarray(prior_json["pi"], float))
    d = pts.shape[1]
    out = np.empty(pts.shape[0])
    norm = -0.5 * d * math.log(2 * math.pi * lam)
    for s in range(0, pts.shape[0], CHUNK):
        y = pts[s:s + CHUNK]
        r2 = ((y[:, None, :] - mu[None, :, :]) ** 2).sum(-1)       # (n, K)
        parts = log_b[None, :] - 0.5 * r2 / lam + norm
        m = parts.max(1)
        out[s:s + CHUNK] = m + np.log(np.exp(parts - m[:, None]).sum(1))
    return out


def normalise(logv: np.ndarray, mask=None) -> np.ndarray:
    lv = np.where(mask, logv, -np.inf) if mask is not None else logv
    m = np.max(lv)
    w = np.exp(lv - m)
    return w / w.sum()


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def moments(w: np.ndarray, pts: np.ndarray) -> dict:
    m = w @ pts
    d = pts - m
    return {"mean": m.tolist(), "cov": ((w[:, None] * d).T @ d).tolist()}


def grid_box(cells: dict) -> tuple:
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    for c in cells.values():
        mu, sd = np.asarray(c["mu"], float), np.asarray(c["per_axis_sd"], float)
        lo, hi = np.minimum(lo, mu - PAD_SD * sd), np.maximum(hi, mu + PAD_SD * sd)
    return lo, hi


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs-seed", type=int, required=True)
    p.add_argument("--root", default=G.ROOT)
    p.add_argument("--levels", default=",".join(str(n) for n in G.GRID_LEVELS))
    p.add_argument("--num-samples", type=int, default=G.REF_NUM_SAMPLES)
    args = p.parse_args()
    root = Path(args.root)
    t0 = time.perf_counter()

    cells = G.load_cells()
    comp_json = {c: benchmark.load_prior_json(G.TASK, "mixture", c)
                 for c in G.COMPILED_CELLS}
    theta_true, x_o = benchmark.load_observation(G.TASK, G.SOURCE_PRIOR_TYPE,
                                                 G.SOURCE_PRIOR_ID, args.obs_seed)
    x_o = np.asarray(x_o, float)
    train_lo, train_hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    box = grid_box(cells)
    levels = [int(v) for v in args.levels.split(",")]

    per_level = {}
    finest = {}
    for n in levels:
        centers, area = quadrature._grid(box, n)
        inside = np.all((centers >= train_lo) & (centers <= train_hi), axis=-1)
        ll = quadrature.log_likelihood(G.TASK, centers, x_o)
        rows = {}
        for cid, cell in cells.items():
            lp = exact_log_prior(cell, centers)
            post_full = normalise(ll + lp)
            post_sm = normalise(ll + lp, inside)
            row = {
                "exact_posterior_edge_mass_full": float(
                    post_full.reshape(n, n)[[0, -1], :].sum()
                    + post_full.reshape(n, n)[1:-1, [0, -1]].sum()),
                "exact_posterior_mass_outside_training_box": float(post_full[~inside].sum()),
                "exact_prior_mass_outside_training_box": float(
                    normalise(lp)[~inside].sum()),
                "exact_posterior_full": moments(post_full, centers),
                "exact_posterior_support_matched": moments(post_sm, centers),
            }
            if cid in comp_json:
                lpc = compiled_log_prior(comp_json[cid], centers)
                row.update({
                    "num_atoms": len(comp_json[cid]["mu"]),
                    "prior_tv_full": tv(normalise(lp), normalise(lpc)),
                    "prior_tv_support_matched": tv(normalise(lp, inside),
                                                   normalise(lpc, inside)),
                    "posterior_tv_full": tv(post_full, normalise(ll + lpc)),
                    "posterior_tv_support_matched": tv(post_sm, normalise(ll + lpc, inside)),
                    "max_abs_log_prior_gap_where_prior_mass": float(np.max(np.abs(
                        (lp - lpc)[normalise(lp) > 1e-8 * normalise(lp).max()]))),
                })
            rows[cid] = row
            if n == levels[-1]:
                finest[cid] = (centers, ll + lp, area)
        per_level[n] = rows
        print(f"[obs {args.obs_seed}] level {n}: " + " ".join(
            f"c{c}:TVpost_sm={r.get('posterior_tv_support_matched', float('nan')):.5f}"
            for c, r in rows.items()), flush=True)

    # refinement drift (the only approximation the grid adds)
    drift = {}
    if len(levels) >= 2:
        a, b = per_level[levels[-2]], per_level[levels[-1]]
        for cid in cells:
            dd = {"d_mean_full_max": float(np.max(np.abs(
                np.subtract(b[cid]["exact_posterior_full"]["mean"],
                            a[cid]["exact_posterior_full"]["mean"]))))}
            for k in ("prior_tv_support_matched", "posterior_tv_support_matched",
                      "posterior_tv_full"):
                if k in b[cid]:
                    dd[f"d_{k}"] = b[cid][k] - a[cid][k]
            drift[cid] = dd

    # exact-target reference banks for the anisotropic cells
    n = levels[-1]
    for cid in G.ANISO_CELLS:
        centers, log_joint, area = finest[cid]
        lse = quadrature._logsumexp(log_joint)
        post = quadrature.GridPosterior(
            task=G.TASK, kind="exact_anisotropic_target", box=box, n=n, centers=centers,
            log_w=log_joint - lse, log_norm=lse + math.log(area), cell_area=area)
        rng = np.random.default_rng(0 * 1_000_003 + args.obs_seed)
        samples = post.sample(rng, args.num_samples)
        ref_dir = root / "references" / G.TASK / f"aniso_{cid}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ref_dir / f"obs_{args.obs_seed}.npz", samples=samples,
                            samples_support_matched=support_matched(samples,
                                                                    (train_lo, train_hi)),
                            x_o=x_o, theta_true=np.asarray(theta_true, float))
        (ref_dir / f"obs_{args.obs_seed}.json").write_text(json.dumps({
            "kind": "exact_anisotropic_target_quadrature", "cell": cid,
            "grid_n": n, "box_low": box[0].tolist(), "box_high": box[1].tolist(),
            "levels": levels, "refinement": drift.get(cid),
            "edge_mass": per_level[n][cid]["exact_posterior_edge_mass_full"],
            "num_samples": args.num_samples, "sample_seed": args.obs_seed,
            "note": "reference for the EXACT anisotropic target p_0 * N(mu, Sigma_gamma)",
            "theta_true_note": "theta_true was drawn from the shipped mild_0 prior",
        }, indent=2))

    out = root / "representation"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"task": G.TASK, "obs_seed": args.obs_seed, "grid_box_low": box[0].tolist(),
           "grid_box_high": box[1].tolist(), "levels": levels,
           "per_level": {str(k): {str(c): v for c, v in r.items()}
                         for k, r in per_level.items()},
           "refinement_drift": {str(c): v for c, v in drift.items()},
           "repo_git": git_state(REPO),
           "source_sha256_16": benchmark.source_hashes(
               [Path(__file__), REPO / "scripts" / "anisotropy_config.py",
                *sorted((REPO / "spectra").glob("*.py"))]),
           "wall_clock_s": time.perf_counter() - t0}
    (out / f"representation_obs{args.obs_seed}.json").write_text(
        json.dumps(rec, indent=2, default=str))
    print(f"-> {out / f'representation_obs{args.obs_seed}.json'} "
          f"({rec['wall_clock_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
