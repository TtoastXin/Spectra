#!/usr/bin/env python
"""Score the compiled cell against both the compiled and the true target.

``compute_metrics.py`` scores every row against one reference per cell.  The
compiled cell has two references, and the difference between them is the
compilation error:

``compiled target``   ``p_0 * r~``, the best Spectra can reproduce;
``true target``       ``p_0 * r``, the requested posterior.

Scoring each method against both separates transport error from the error of
the dictionary's representation of the prior.  PriorGuide and PG-FullCov are
given the true ratio, so for them the true target is the relevant reference and
the compiled target is shown only for completeness.

    python scripts/score_correlated.py --root results/correlated_prior \\
        --task two_moons --obs-seed 1000003 --prior-ids 20,21 \\
        --out results/correlated_prior/metrics/compiled_obs1000003.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra.references import two_moons_mode_mass  # noqa: E402
from spectra.simformer import _upstream_root  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
C2ST_SEEDS = (0, 1, 2, 3, 4)


def upstream_metrics():
    root = _upstream_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sim.methods import metrics as m  # noqa: E402
    return m


def moment_errors(a: np.ndarray, b: np.ndarray) -> dict:
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a.T), np.cov(b.T)
    return {
        "mean_err_l2": float(np.linalg.norm(ma - mb)),
        "mean_err_rel": float(np.linalg.norm(ma - mb)
                              / max(np.linalg.norm(mb), 1e-12)),
        "cov_err_fro": float(np.linalg.norm(ca - cb)),
        "cov_err_rel": float(np.linalg.norm(ca - cb)
                             / max(np.linalg.norm(cb), 1e-12)),
    }


def score(samples, ref, m, n_jobs=1) -> dict:
    vals = []
    for seed in C2ST_SEEDS:
        idx = np.random.default_rng(seed).choice(
            ref.shape[0], size=min(samples.shape[0], ref.shape[0]), replace=False)
        vals.append(float(m.compute_c2st(samples, ref[idx], seed=seed,
                                         classifier_kwargs={"n_jobs": n_jobs})))
    out = {"c2st": float(np.mean(vals)),
           "c2st_sd_over_ref_subsamples": float(np.std(vals, ddof=1)),
           "mmtv": float(m.compute_mmtv(samples, ref))}
    out.update(moment_errors(samples, ref))
    out["mode_mass"] = two_moons_mode_mass(samples)
    out["reference_mode_mass"] = two_moons_mode_mass(ref)
    out["mode_mass_error"] = abs(out["mode_mass"] - out["reference_mode_mass"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--task", default="two_moons")
    p.add_argument("--obs-seed", type=int, default=1000003)
    p.add_argument("--prior-ids", default="20,21")
    p.add_argument("--samples-dir", default="benchmark_e2e",
                   help="directory under --root holding samples/<task>/...")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    root = Path(args.root)
    m = upstream_metrics()
    true_ref = np.load(
        root / "compilation" / f"true_target_{args.task}_obs{args.obs_seed}.npz"
    )["samples"]

    rows = []
    for pid in [int(v) for v in args.prior_ids.split(",")]:
        comp_ref = np.load(root / "references" / args.task / f"mixture_{pid}"
                           / f"obs_{args.obs_seed}.npz")["samples"]
        cap = json.loads((root / "references" / args.task / f"mixture_{pid}"
                          / f"obs_{args.obs_seed}.json").read_text())
        # the two references against each other: the compilation error itself,
        # measured in the same metric the methods are measured in
        base_row = {"prior_id": pid, "capacity": cap.get("capacity"),
                    "num_atoms": cap.get("num_atoms"),
                    "method": "(compiled target vs true target)",
                    "reference_kind": "true_target", "seed": -1}
        rows.append({**base_row, **score(comp_ref, true_ref, m)})
        print(f"[compiled_{pid}] compiled vs true target: "
              f"C2ST={rows[-1]['c2st']:.4f}", flush=True)

        for sdir in sorted((root / args.samples_dir / "samples" / args.task).glob(
                f"*/model_0/mixture_{pid}/obs_{args.obs_seed}")):
            npz = sdir / "steps100_L0_power.npz"
            if not npz.is_file():
                continue
            meta = json.loads(npz.with_suffix(".json").read_text())
            s = np.load(npz)["samples"]
            for kind, ref in (("compiled_target", comp_ref),
                              ("true_target", true_ref)):
                rows.append({
                    "prior_id": pid, "capacity": cap.get("capacity"),
                    "num_atoms": cap.get("num_atoms"),
                    "method": meta["method"],
                    "method_family": meta.get("method_family"),
                    "ratio_input": meta.get("ratio_input"),
                    "seed": meta.get("seed"),
                    "reference_kind": kind,
                    "wall_clock_per_1000_samples_s": meta.get(
                        "wall_clock_per_1000_samples_s"),
                    "nan_or_inf": meta.get("nan_or_inf"),
                    **score(s, ref, m),
                })
            print(f"  {meta['method']:<32} "
                  f"C2ST(compiled)={rows[-2]['c2st']:.4f} "
                  f"C2ST(true)={rows[-1]['c2st']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps({
        "kind": "round7_compiled_scoring", "task": args.task,
        "obs_seed": args.obs_seed, "repo_git": git_state(REPO),
        "note": ("every row is scored against both the compiled target and the "
                 "true target; the first row of each block is the two "
                 "references against each other, i.e. the compilation error in "
                 "the same metric"),
    }, indent=2, default=str))
    print(f"wrote {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
