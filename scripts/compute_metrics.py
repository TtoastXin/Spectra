#!/usr/bin/env python
"""Posterior metrics, scored in a separate torch-allowed process.

Sampling and scoring are separate jobs: the samplers run under JAX on the GPU,
this one runs torch/sbi on the CPU, and the two do not share a process or a
device.  The shipped helpers imported here (``spectra.benchmark`` and
``spectra.references``) pull in JAX transitively, but nothing in this script
puts a sampler on a device.  The metric implementations are upstream's
``priorg.sim.methods.metrics``, so the numbers stay comparable to the paper.

C2ST is balanced 1000-vs-1000: the reference bank is larger than a method's
sample set, and training a classifier on 1000 vs 10000 would score class
imbalance rather than distributional difference.  Five fixed-seed reference
subsamples are drawn and both the mean and the spread across them are reported.
MMTV and the moment diagnostics use the whole reference bank.

    python scripts/compute_metrics.py --root results/single_factor_controlled_strong \\
        --task gaussian_linear \\
        --out results/single_factor_controlled_strong/metrics/gaussian_linear.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark  # noqa: E402
from spectra.references import two_moons_mode_mass  # noqa: E402
from spectra.simformer import _upstream_root  # noqa: E402

C2ST_SUBSAMPLES = 5
C2ST_SEEDS = (11, 22, 33, 44, 55)

# One C2ST costs ~3.4 s and does not speed up with more cores (a random forest on
# 2000x10 is overhead-bound), so a full task is hours.  Rows are therefore
# appended to a JSONL sidecar as they are produced and re-runs skip what is
# already there; a job that hits its wall clock loses one row, not all of them.
ROW_KEY = ("method", "model_id", "prior_type", "prior_id", "obs_seed",
           "num_steps", "langevin_steps", "grid", "reference_kind")


def _upstream_metrics(root=None):
    path = _upstream_root(root)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from sim.methods import metrics as m

    return m


def c2st_corrected(x, y, seed, n_folds=5, n_jobs=1):
    """Same protocol as upstream, with the per-feature standard deviation in the z-score.

    Upstream's ``compute_c2st`` sets ``X_std = np.mean(X, axis=0)``.  With the
    random-forest classifier the benchmark uses, a per-feature affine rescaling
    cannot change any split, so the two agree; this variant is computed
    alongside as a check.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold, cross_val_score

    x, y = np.asarray(x), np.asarray(y)
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-14, 1.0, sd)
    x, y = (x - mu) / sd, (y - mu) / sd
    data = np.concatenate([x, y])
    target = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    clf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, data, target, cv=cv, scoring="accuracy")))


def moment_errors(samples, ref):
    m_s, m_r = samples.mean(0), ref.mean(0)
    c_s, c_r = np.cov(samples.T), np.cov(ref.T)
    return {
        "mean_err_l2": float(np.linalg.norm(m_s - m_r)),
        "mean_err_rel": float(
            np.linalg.norm(m_s - m_r) / max(np.linalg.norm(m_r), 1e-12)),
        "cov_err_fro": float(np.linalg.norm(np.atleast_2d(c_s - c_r))),
        "cov_err_rel": float(
            np.linalg.norm(np.atleast_2d(c_s - c_r))
            / max(np.linalg.norm(np.atleast_2d(c_r)), 1e-12)),
    }


NAN_METRIC_KEYS = ("c2st", "c2st_sd_over_ref_subsamples", "c2st_corrected_zscore",
                   "mmtv", "rmse_to_true_theta", "gskl", "mean_err_l2",
                   "mean_err_rel", "cov_err_fro", "cov_err_rel",
                   "n_distinct_samples", "n_degenerate_axes",
                   "kde_metrics_defined")


def evaluate(samples, ref_bank, theta_true, m, box=None, n_jobs=1,
             check_corrected=False, task=None):
    n = samples.shape[0]
    if not np.isfinite(samples).all():
        # A method that diverges (some diagnostic routes do at high noise) has
        # no distance to a reference.  Report a NaN row flagged ``diverged``
        # instead of letting the classifier raise and lose the whole file.
        return {
            **{k: float("nan") for k in NAN_METRIC_KEYS},
            "n_method_samples": int(n),
            "n_reference_samples": int(ref_bank.shape[0]),
            "nan_or_inf": int((~np.isfinite(samples)).sum()),
            "diverged": 1,
        }
    c2st_up, c2st_fix = [], []
    for seed in C2ST_SEEDS[:C2ST_SUBSAMPLES]:
        idx = np.random.default_rng(seed).choice(
            ref_bank.shape[0], size=min(n, ref_bank.shape[0]), replace=False)
        sub = ref_bank[idx]
        c2st_up.append(float(m.compute_c2st(
            samples, sub, seed=seed, classifier_kwargs={"n_jobs": n_jobs})))
        if check_corrected:
            c2st_fix.append(c2st_corrected(samples, sub, seed=seed, n_jobs=n_jobs))
    # A method can also collapse onto (nearly) a point mass.  SIR does so once
    # the importance weights degenerate (on GL-20D the 20000-point bank
    # resamples to a single distinct draw).  Upstream's MMTV then runs a KDE
    # with zero bandwidth, whose linear binning indexes with INT64_MIN and
    # raises, and the Gaussian symmetric KL becomes singular.  Both are reported
    # as NaN with the ``kde_metrics_defined`` flag; C2ST and the moment errors
    # are well defined against a point mass and are kept.
    n_degenerate_axes = int((samples.std(axis=0) < 1e-12).sum())
    n_distinct = int(np.unique(samples, axis=0).shape[0])
    kde_safe = n_degenerate_axes == 0 and n_distinct > 2
    row = {
        "c2st": float(np.mean(c2st_up)),
        "c2st_sd_over_ref_subsamples": float(np.std(c2st_up, ddof=1)),
        "c2st_corrected_zscore": float(np.mean(c2st_fix)) if c2st_fix else float("nan"),
        "mmtv": float(m.compute_mmtv(samples, ref_bank)) if kde_safe else float("nan"),
        "rmse_to_true_theta": float(m.compute_rmse(theta_true, samples)),
        "gskl": float(m.compute_gskl(samples, ref_bank)) if kde_safe else float("nan"),
        "n_method_samples": int(n),
        "n_reference_samples": int(ref_bank.shape[0]),
        "nan_or_inf": int((~np.isfinite(samples)).sum()),
        "n_distinct_samples": n_distinct,
        "n_degenerate_axes": n_degenerate_axes,
        "kde_metrics_defined": int(kde_safe),
        "diverged": 0,
    }
    row.update(moment_errors(samples, ref_bank))
    if box is not None:
        row["method_mass_outside_training_box"] = benchmark.outside_box_fraction(
            samples, box)
        row["reference_mass_outside_training_box"] = benchmark.outside_box_fraction(
            ref_bank, box)
    if task == "two_moons":
        row["mode_mass"] = two_moons_mode_mass(samples)
        row["reference_mode_mass"] = two_moons_mode_mass(ref_bank)
        row["mode_mass_error"] = abs(row["mode_mass"] - row["reference_mode_mass"])
    return row


_WORKER = {}


def _worker_init(task):
    """One upstream-metrics import per worker; sklearn stays single-threaded.

    A single C2ST costs ~3.4 s and does not speed up with more cores (a random
    forest on 2000x10 is overhead-bound), so the only useful parallelism is
    across rows.
    """
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _WORKER["m"] = _upstream_metrics()
    _WORKER["box"] = benchmark.training_support(task)
    _WORKER["task"] = task


def _worker_eval(item, n_jobs, check_corrected, want):
    """``want`` lists the reference kinds still missing for this file."""
    npz_path, ref_path, base = item
    data, ref = np.load(npz_path), np.load(ref_path)
    m, box, task = _WORKER["m"], _WORKER["box"], _WORKER["task"]
    ev = dict(n_jobs=n_jobs, check_corrected=check_corrected, task=task)
    out = []
    if "full" in want:
        out.append({**base, **evaluate(data["samples"], ref["samples"],
                                       data["theta_true"], m, box, **ev)})
    if "training_prior" in want and "samples_training_prior" in ref:
        out.append({**base, "reference_kind": "training_prior",
                    **evaluate(data["samples"], ref["samples_training_prior"],
                               data["theta_true"], m, box, **ev)})
    if "support_matched" in want and "samples_support_matched" in ref:
        out.append({**base, "reference_kind": "support_matched",
                    **evaluate(data["samples"], ref["samples_support_matched"],
                               data["theta_true"], m, box, **ev)})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="results root of one study")
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--out", required=True)
    p.add_argument("--support-matched", action="store_true",
                   help="also score against the box-truncated reference")
    p.add_argument("--workers", type=int, default=1,
                   help="evaluate this many files in parallel; a single C2ST "
                        "does not speed up with more cores (see _worker_init)")
    p.add_argument("--c2st-jobs", type=int, default=8,
                   help="RandomForest n_jobs; changes speed, not the score")
    p.add_argument("--overwrite", action="store_true",
                   help="ignore the JSONL sidecar and recompute every row")
    p.add_argument("--check-corrected-c2st", action="store_true",
                   help="also compute C2ST with the per-feature standard deviation "
                        "in the z-score; with the random-forest classifier it "
                        "gives the same score, so it is off by default")
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".jsonl")
    rows, done = [], set()
    if sidecar.is_file() and not args.overwrite:
        for line in sidecar.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done.add(tuple(str(r.get(k)) for k in ROW_KEY))
        print(f"resuming: {len(rows)} rows already in {sidecar}", flush=True)
    sink = sidecar.open("w" if args.overwrite else "a")

    def emit(row):
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    work = []
    for npz_path in sorted((root / "samples" / args.task).glob("*/*/*/*/*.npz")):
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.is_file():
            print(f"skip {npz_path}: no manifest", flush=True)
            continue
        meta = json.loads(meta_path.read_text())
        ref_path = (root / "references" / args.task
                    / f"{meta['prior_type']}_{meta['prior_id']}"
                    / f"obs_{meta['obs_seed']}.npz")
        if not ref_path.is_file():
            print(f"skip {npz_path}: missing reference {ref_path}", flush=True)
            continue
        base = {
            "task": args.task, "method": meta["method"],
            "conditioning": meta.get("conditioning"), "model_id": meta["model_id"],
            "prior_type": meta["prior_type"], "prior_id": meta["prior_id"],
            "obs_seed": meta["obs_seed"],
            "num_steps": meta["sampler"]["num_steps"],
            "langevin_steps": meta["sampler"]["langevin_steps"],
            "grid": meta["sampler"]["grid"],
            "wall_clock_s": meta["wall_clock_s"],
            "wall_clock_per_1000_samples_s": meta["wall_clock_per_1000_samples_s"],
            "base_score_calls": meta["nfe"].get("base_score_calls"),
            "denoiser_jacobian_calls": meta["nfe"].get("denoiser_jacobian_calls", 0),
            "bank_kernel_calls": meta["nfe"].get("bank_kernel_calls", 0),
            "reference_kind": "full",
        }
        # ``method_family`` rather than ``method``: the runners suffix every row
        # with its sampler seed (``base_ms0``), and keying on the raw name would
        # drop the base-vs-base-reference rows used as the backbone
        # qualification diagnostic.
        is_base = meta.get("method_family", meta["method"]) == "base"
        want = [k for k in ("full",
                            *(["training_prior"] if is_base else []),
                            *(["support_matched"] if args.support_matched else []))
                if tuple(str({**base, "reference_kind": k}[j]) for j in ROW_KEY) not in done]
        if want:
            work.append(((str(npz_path), str(ref_path), base), want))

    print(f"{len(work)} files to evaluate with {args.workers} worker(s)", flush=True)
    fn = partial(_worker_eval, n_jobs=args.c2st_jobs,
                 check_corrected=args.check_corrected_c2st)
    if args.workers <= 1:
        _worker_init(args.task)
        for item, want in work:
            for r in fn(item, want=want):
                emit(r)
                print(f"{r['method']:<12} {r['prior_type']}_{r['prior_id']} "
                      f"obs{r['obs_seed']} steps{r['num_steps']} {r['reference_kind']}"
                      f"  C2ST={r['c2st']:.4f}", flush=True)
    else:
        # arviz (imported through sbi by the metrics) writes a once-a-day stamp
        # file on import; workers importing it at the same time race on that file.
        # Importing it here first writes the stamp once.
        import arviz  # noqa: F401
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init,
                                 initargs=(args.task,)) as pool:
            futures = [pool.submit(fn, item, want=want) for item, want in work]
            for i, fut in enumerate(futures, 1):
                for r in fut.result():
                    emit(r)
                print(f"[{i}/{len(futures)}] {rows[-1]['method']:<10} "
                      f"{rows[-1]['prior_type']}_{rows[-1]['prior_id']} "
                      f"obs{rows[-1]['obs_seed']} steps{rows[-1]['num_steps']}"
                      f"  C2ST={rows[-1]['c2st']:.4f}", flush=True)

    sink.close()
    if not rows:
        print("no rows produced", flush=True)
        return 1
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
