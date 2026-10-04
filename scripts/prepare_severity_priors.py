#!/usr/bin/env python
"""Write and check the five shift-severity cells of each sweep task.

The sweep needs five single-factor Gaussian test priors per task that share the
shipped ``mild_0`` centre and observations and differ only in width.  The new
priors, the copied observations and a copy of each task's training prior (the
loaders read a prior's training prior from the same root) are written into a
fresh ``--out`` directory; the shipped ``data/`` directory is only read.

Everything is then re-read through the ordinary loaders and checked:

* ``sigma / sd_training == lambda`` for every cell and axis;
* ``sigma == sigma_mild * lambda / 0.5`` exactly, so ``lambda = 0.50`` is
  exactly the shipped mild width;
* the centre is exactly the shipped ``mild_0`` centre;
* every copied observation is identical to its ``mild_0`` source;
* the exact ratio is a single exponential-quadratic factor with
  ``kappa = 1 / sigma^2`` and a valid transport guard at every noise level.

Any failure exits non-zero.  ``--self-test`` builds a throwaway copy, tampers
with one width and one observation, and requires the checks to reject it.

    python scripts/prepare_severity_priors.py --out <new directory>
    python scripts/prepare_severity_priors.py --self-test --out <new directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import severity_config as G  # noqa: E402
from spectra import benchmark  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SIGMA_TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_data_root(task: str) -> Path:
    """Where the shipped ``mild_0`` prior and observations of ``task`` live."""
    return benchmark.data_root(task)


def copy_training_prior(out: Path, source: Path, task: str) -> str:
    """The loaders read the training prior from the same root as the new cells."""
    dst = out / "priors" / task / "training.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "priors" / task / "training.json", dst)
    return str(dst)


def training_marginal_sd(task: str, source: Path) -> np.ndarray:
    js = json.loads((source / "priors" / task / "training.json").read_text())
    if js["dist"] != "uniform":
        raise RuntimeError(f"{task}: the sweep assumes a uniform training prior, "
                           f"got {js['dist']!r}")
    low = np.asarray(js["low"], float)
    high = np.asarray(js["high"], float)
    return (high - low) / math.sqrt(12.0)


def build(out: Path) -> dict:
    """Write every new prior/observation cell; return the provenance record."""
    out.mkdir(parents=True, exist_ok=True)
    record = {"data_root": str(out), "tasks": {}, "written": [], "copied": []}

    for task in G.TASKS:
        source = _source_data_root(task)
        record["copied"].append(copy_training_prior(out, source, task))
        src_prior_path = source / "priors" / task / f"{G.PRIOR_TYPE}_{G.SOURCE_PRIOR_ID}.json"
        src_prior = json.loads(src_prior_path.read_text())
        if src_prior["dist"] != "gaussian":
            raise RuntimeError(f"{task}: source prior is {src_prior['dist']!r}, "
                               "the sweep only scales a single Gaussian factor")
        mu = [float(v) for v in src_prior["mu"]]
        sigma_mild = [float(v) for v in src_prior["sigma"]]
        sd_train = training_marginal_sd(task, source)
        obs_src_dir = (source / "observations" / task
                       / f"{G.PRIOR_TYPE}_{G.SOURCE_PRIOR_ID}")
        obs_files = sorted(obs_src_dir.glob("obs_*.json"))

        cells = {}
        for lam in G.LAMBDAS:
            pid = G.prior_id(lam)
            scale = lam / G.ANCHOR_LAMBDA
            sigma = [s * scale for s in sigma_mild]
            payload = {
                "task": task,
                "dist": "gaussian",
                "type": G.PRIOR_TYPE,
                "mu": mu,
                "sigma": sigma,
                # provenance; the loaders read only task/dist/mu/sigma
                "round14_lambda": lam,
                "round14_sigma_over_training_sd": [
                    float(s / sd) for s, sd in zip(sigma, sd_train)],
                "round14_source_prior": str(src_prior_path.relative_to(source)),
                "round14_source_prior_sha256": sha256(src_prior_path),
                "round14_rule": ("sigma = sigma_mild * lambda / 0.5 at the shipped "
                                 "mild_0 centre; observations are the mild_0 ones"),
                "generated_by": "scripts/round14_prepare_data.py",
            }
            prior_out = out / "priors" / task / f"{G.PRIOR_TYPE}_{pid}.json"
            prior_out.parent.mkdir(parents=True, exist_ok=True)
            prior_out.write_text(json.dumps(payload, indent=4) + "\n")
            record["written"].append(str(prior_out))

            obs_out_dir = out / "observations" / task / f"{G.PRIOR_TYPE}_{pid}"
            obs_out_dir.mkdir(parents=True, exist_ok=True)
            copied = []
            for src in obs_files:
                dst = obs_out_dir / src.name
                shutil.copyfile(src, dst)
                copied.append(str(dst))
                record["written"].append(str(dst))
            cells[str(pid)] = {
                "lambda": lam, "prior_id": pid, "prior_json": str(prior_out),
                "prior_json_sha256": sha256(prior_out),
                "sigma": sigma, "n_observations": len(copied),
                "observation_source": str(obs_src_dir),
            }
        record["tasks"][task] = {
            "source_data_root": str(source),
            "source_prior": str(src_prior_path),
            "source_prior_sha256": sha256(src_prior_path),
            "training_marginal_sd": sd_train.tolist(),
            "sigma_mild_shipped": sigma_mild,
            "sigma_mild_over_training_sd": [float(s / sd)
                                            for s, sd in zip(sigma_mild, sd_train)],
            "centre": mu,
            "observations": [p.name for p in obs_files],
            "cells": cells,
        }
    return record


def verify(out: Path) -> dict:
    """Re-read every cell through the ordinary loaders; raise on any mismatch."""
    from spectra.prior_shift import build_prior_shift  # noqa: E402

    checks, failures = [], []

    def check(ok: bool, what: str, detail=None):
        checks.append({"check": what, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(f"{what}: {detail}")

    for task in G.TASKS:
        source = _source_data_root(task)
        src_prior = json.loads(
            (source / "priors" / task
             / f"{G.PRIOR_TYPE}_{G.SOURCE_PRIOR_ID}.json").read_text())
        mu_src = [float(v) for v in src_prior["mu"]]
        sigma_mild = [float(v) for v in src_prior["sigma"]]
        sd_train = training_marginal_sd(task, source)
        obs_src_dir = (source / "observations" / task
                       / f"{G.PRIOR_TYPE}_{G.SOURCE_PRIOR_ID}")

        for lam in G.LAMBDAS:
            pid = G.prior_id(lam)
            cell = f"{task}/{G.PRIOR_TYPE}_{pid}"
            js = benchmark.load_prior_json(task, G.PRIOR_TYPE, pid, root=out)
            mu = [float(v) for v in js["mu"]]
            sigma = [float(v) for v in js["sigma"]]

            check(mu == mu_src, f"{cell}: centre is the shipped mild_0 centre",
                  {"loaded": mu, "shipped": mu_src})
            want = [s * (lam / G.ANCHOR_LAMBDA) for s in sigma_mild]
            check(sigma == want, f"{cell}: sigma = sigma_mild * lambda / 0.5",
                  {"loaded": sigma, "expected": want})
            ratios = [s / sd for s, sd in zip(sigma, sd_train)]
            check(all(abs(r - lam) <= 1e-8 for r in ratios),
                  f"{cell}: sigma / sd_training == lambda",
                  {"ratios": ratios, "lambda": lam})
            if abs(lam - G.ANCHOR_LAMBDA) < 1e-15:
                check(sigma == sigma_mild,
                      f"{cell}: anchor reproduces the shipped mild width exactly",
                      {"loaded": sigma, "shipped": sigma_mild})

            for src in sorted(obs_src_dir.glob("obs_*.json")):
                dst = (out / "observations" / task / f"{G.PRIOR_TYPE}_{pid}" / src.name)
                same = dst.is_file() and sha256(dst) == sha256(src)
                check(same, f"{cell}/{src.name}: identical to the mild_0 source",
                      {"copy": str(dst), "source": str(src)})

            shift = build_prior_shift(task, G.PRIOR_TYPE, pid, root=out)
            eq = shift.expquad
            check(eq is not None, f"{cell}: ratio is one exponential-quadratic factor")
            if eq is not None:
                kappa_want = 1.0 / (sigma[0] ** 2)
                check(abs(eq.kappa - kappa_want) <= 1e-9 * kappa_want,
                      f"{cell}: kappa == 1 / sigma^2",
                      {"kappa": eq.kappa, "expected": kappa_want})
                check(eq.guard(0.0) and eq.guard(1e6),
                      f"{cell}: transport guard 1 + kappa tau > 0 holds")
            n_obs = len(G.obs_seeds(task))
            for obs_seed in G.obs_seeds(task):
                theta, x_o = benchmark.load_observation(task, G.PRIOR_TYPE, pid, obs_seed,
                                                        root=out)
                theta0, x0 = benchmark.load_observation(
                    task, G.PRIOR_TYPE, G.SOURCE_PRIOR_ID, obs_seed, root=source)
                check(bool(np.array_equal(np.asarray(theta), np.asarray(theta0))
                           and np.array_equal(np.asarray(x_o), np.asarray(x0))),
                      f"{cell}/obs_{obs_seed}: loads the mild_0 observation")
            check(n_obs > 0, f"{cell}: observation list is non-empty")

    if failures:
        raise RuntimeError(f"{len(failures)} severity data checks failed:\n  "
                           + "\n  ".join(failures[:20]))
    return {"n_checks": len(checks), "all_ok": True, "checks": checks}


def self_test(workdir: Path) -> dict:
    """Show the checks can fail: tamper with a good tree and require rejection."""
    out = {}
    for name, tamper in (("bad_sigma", "sigma"), ("bad_observation", "obs")):
        tmp = Path(tempfile.mkdtemp(prefix=f"r14_{name}_", dir=str(workdir)))
        build(tmp)
        verify(tmp)  # a clean tree must pass first
        task, pid = "two_moons", G.prior_id(G.LAMBDAS[-1])
        if tamper == "sigma":
            path = tmp / "priors" / task / f"{G.PRIOR_TYPE}_{pid}.json"
            js = json.loads(path.read_text())
            js["sigma"] = [float(v) * 1.05 for v in js["sigma"]]
            path.write_text(json.dumps(js, indent=4) + "\n")
        else:
            path = next((tmp / "observations" / task
                         / f"{G.PRIOR_TYPE}_{pid}").glob("obs_*.json"))
            js = json.loads(path.read_text())
            js["x"] = [float(v) + 0.1 for v in np.asarray(js["x"], float).ravel()]
            path.write_text(json.dumps(js, indent=4) + "\n")
        try:
            verify(tmp)
        except RuntimeError as exc:
            out[name] = {"rejected": True, "tampered": str(path),
                         "error_head": str(exc).splitlines()[0]}
        else:
            out[name] = {"rejected": False, "tampered": str(path)}
        shutil.rmtree(tmp, ignore_errors=True)
    out["all_rejected"] = all(v["rejected"] for k, v in out.items()
                              if isinstance(v, dict))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True,
                   help="new directory for the cells (never the shipped data/)")
    p.add_argument("--report", default=None,
                   help="where to write the provenance/check report "
                        "(default <out>/prepare_report.json)")
    p.add_argument("--self-test", action="store_true",
                   help="also demonstrate that the checks reject bad input")
    args = p.parse_args()

    out = Path(args.out).resolve()
    shipped = benchmark.DATA_DIR.resolve()
    if out == shipped or shipped in out.parents:
        p.error(f"refusing to write into the shipped data directory: {out}")

    record = build(out)
    checks = verify(out)
    payload = {"record": record, "verification": checks}
    if args.self_test:
        with tempfile.TemporaryDirectory(prefix="r14_selftest_") as tmp:
            payload["self_test"] = self_test(Path(tmp))
        if not payload["self_test"]["all_rejected"]:
            print(json.dumps(payload["self_test"], indent=2))
            raise SystemExit("self-test did not reject tampered input")

    report = Path(args.report) if args.report else (
        out / "prepare_report.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, default=str))
    n_prior = sum(len(v["cells"]) for v in record["tasks"].values())
    print(f"wrote {n_prior} prior cells and {len(record['written']) - n_prior} "
          f"observation copies under {out}")
    print(f"{checks['n_checks']} checks passed; report -> {report}")
    if args.self_test:
        for k, v in payload["self_test"].items():
            if isinstance(v, dict):
                print(f"self-test {k}: rejected={v['rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
