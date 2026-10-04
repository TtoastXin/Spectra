#!/usr/bin/env python
"""Write and check the Two Moons mixture cells of the component-count study.

As in ``prepare_severity_priors.py``, the cells, their observations (copies of
the shipped ``mild_0`` ones) and a copy of the training prior go into a fresh
``--out`` directory, the shipped ``data/`` directory is only read, and every
cell is re-read through the ordinary loaders and checked.

Checked here:

* the prior id does not shadow a shipped cell (upstream ``mixture_0..9``);
* the centres of every ``K`` are nested subsets of one 64-point circle, i.e. a
  subset of the ``K = 64`` set;
* weights are equal and sum to one; every component has the same width, equal
  to the shipped ``strong_0`` width;
* the copied observations are identical to their ``mild_0`` sources;
* the exact ratio has ``K`` atoms with ``kappa = 1 / sigma^2`` and positive
  weights, so PriorGuide gets the exact mixture and Spectra the same components.

``--self-test`` tampers with a good tree (a rotated centre, an unequal weight)
and requires the checks to reject it.

    python scripts/prepare_k_mixtures.py --out <new directory>
    python scripts/prepare_k_mixtures.py --self-test --out <new directory>
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

from scripts import k_scaling_config as KG  # noqa: E402
from scripts import severity_config as SG  # noqa: E402
from spectra import benchmark  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_data() -> Path:
    """The shipped ``data/`` directory; only read."""
    return benchmark.data_root(KG.TASK)


def build(out: Path, *, tamper: str | None = None) -> dict:
    src = source_data()
    mild = json.loads((src / "priors" / KG.TASK
                       / f"{SG.PRIOR_TYPE}_{SG.SOURCE_PRIOR_ID}.json").read_text())
    strong_path = src / "priors" / KG.TASK / "strong_0.json"
    strong = json.loads(strong_path.read_text())
    centre = [float(v) for v in mild["mu"]]
    sigma = float(strong["sigma"][0])
    if sigma != KG.SIGMA:
        raise RuntimeError(f"shipped strong width {sigma!r} != KG.SIGMA "
                           f"{KG.SIGMA!r}; the shipped priors no longer match "
                           "k_scaling_config")
    obs_src_dir = (src / "observations" / KG.TASK
                   / f"{SG.PRIOR_TYPE}_{SG.SOURCE_PRIOR_ID}")

    record = {"data_root": str(out), "task": KG.TASK, "centre": centre,
              "sigma": sigma, "radius": KG.RADIUS, "base_points": KG.BASE_POINTS,
              "source_prior": str(src / "priors" / KG.TASK
                                  / f"{SG.PRIOR_TYPE}_{SG.SOURCE_PRIOR_ID}.json"),
              "strong_width_source": str(strong_path), "cells": {}, "written": []}
    # the loaders read a prior's training prior from the same root as the prior
    train = out / "priors" / KG.TASK / "training.json"
    train.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "priors" / KG.TASK / "training.json", train)
    record["written"].append(str(train))

    for k in sorted(set(KG.K_P1) | set(KG.K_P2)):
        pid = KG.prior_id(k)
        shipped = src / "priors" / KG.TASK / f"{KG.PRIOR_TYPE}_{pid}.json"
        if shipped.is_file() and "round14_k" not in json.loads(shipped.read_text()):
            raise RuntimeError(f"prior id {pid} is another shipped cell; refusing to shadow it")
        mu = KG.centres(k, centre)
        pi = [1.0 / k] * k
        if tamper == "rotate" and k == 8:
            mu = [[x + 0.01, y] for x, y in mu]
        if tamper == "weights" and k == 8:
            pi = [2.0 / (k + 1) if i == 0 else 1.0 / (k + 1) for i in range(k)]
        payload = {
            "task": KG.TASK, "dist": "mixture", "type": KG.PRIOR_TYPE,
            "mu": mu, "sigma": [[sigma, sigma] for _ in range(k)], "pi": pi,
            "round14_k": k, "round14_radius": KG.RADIUS,
            "round14_base_points": KG.BASE_POINTS, "round14_centre": centre,
            "round14_rule": ("equal-weight components on every 64/K-th point of one "
                             "fixed 64-point circle of radius R about the mild_0 "
                             "centre; common width = shipped strong_0 width"),
            "round14_neighbour_separation_in_sigma": KG.neighbour_separation(k),
            "generated_by": "scripts/round14_k_prepare.py",
        }
        prior_out = out / "priors" / KG.TASK / f"{KG.PRIOR_TYPE}_{pid}.json"
        prior_out.parent.mkdir(parents=True, exist_ok=True)
        prior_out.write_text(json.dumps(payload, indent=4) + "\n")
        record["written"].append(str(prior_out))

        obs_dir = out / "observations" / KG.TASK / f"{KG.PRIOR_TYPE}_{pid}"
        obs_dir.mkdir(parents=True, exist_ok=True)
        for obs_seed in sorted(set(KG.OBS_P1) | set(KG.OBS_P2)):
            s = obs_src_dir / f"obs_{obs_seed}.json"
            shutil.copyfile(s, obs_dir / s.name)
            record["written"].append(str(obs_dir / s.name))
        record["cells"][str(k)] = {
            "K": k, "prior_id": pid, "prior_json": str(prior_out),
            "prior_json_sha256": sha256(prior_out),
            "neighbour_separation_in_sigma": KG.neighbour_separation(k),
        }
    return record


def verify(out: Path) -> dict:
    from spectra import atoms as atoms_mod  # noqa: E402
    from spectra.prior_shift import build_prior_shift  # noqa: E402

    src = source_data()
    mild = json.loads((src / "priors" / KG.TASK
                       / f"{SG.PRIOR_TYPE}_{SG.SOURCE_PRIOR_ID}.json").read_text())
    centre = [float(v) for v in mild["mu"]]
    full = KG.centres(KG.BASE_POINTS, centre)
    full_set = {(round(x, 15), round(y, 15)) for x, y in full}
    obs_src_dir = (src / "observations" / KG.TASK
                   / f"{SG.PRIOR_TYPE}_{SG.SOURCE_PRIOR_ID}")

    checks, failures = [], []

    def check(ok, what, detail=None):
        checks.append({"check": what, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(f"{what}: {detail}")

    for k in sorted(set(KG.K_P1) | set(KG.K_P2)):
        pid = KG.prior_id(k)
        cell = f"{KG.TASK}/{KG.PRIOR_TYPE}_{pid}"
        js = benchmark.load_prior_json(KG.TASK, KG.PRIOR_TYPE, pid, root=out)
        mu = [[float(a), float(b)] for a, b in js["mu"]]
        pi = [float(v) for v in js["pi"]]
        sig = np.asarray(js["sigma"], float)

        check(len(mu) == k and len(pi) == k, f"{cell}: has K={k} components",
              {"n_mu": len(mu), "n_pi": len(pi)})
        check(all(abs(v - 1.0 / k) <= 1e-15 for v in pi) and abs(sum(pi) - 1) <= 1e-12,
              f"{cell}: equal weights summing to one", {"pi": pi[:4]})
        check(np.allclose(sig, KG.SIGMA, rtol=0, atol=0),
              f"{cell}: common width is the shipped strong width",
              {"sigma": sig.reshape(-1)[:2].tolist(), "expected": KG.SIGMA})
        if k == 1:
            check(mu[0] == centre, f"{cell}: K=1 sits at the mild_0 centre",
                  {"mu": mu[0], "centre": centre})
        else:
            subset = all((round(x, 15), round(y, 15)) in full_set for x, y in mu)
            check(subset, f"{cell}: centres are a subset of the 64-point circle")
            radii = [math.hypot(x - centre[0], y - centre[1]) for x, y in mu]
            check(all(abs(r - KG.RADIUS) <= 1e-12 for r in radii),
                  f"{cell}: every centre is at radius R", {"radii": radii[:3]})
            check(mu == KG.centres(k, centre),
                  f"{cell}: centres match the nested rule")

        shift = build_prior_shift(KG.TASK, KG.PRIOR_TYPE, pid, root=out)
        atoms = atoms_mod.atoms_from_shift(shift)
        check(atoms.num_atoms == k, f"{cell}: ratio expands into K atoms",
              {"num_atoms": atoms.num_atoms})
        kappa_want = 1.0 / (KG.SIGMA ** 2)
        check(np.allclose(np.asarray(atoms.kappa), kappa_want, rtol=1e-9),
              f"{cell}: kappa == 1 / sigma^2",
              {"kappa": np.asarray(atoms.kappa)[:2].tolist(), "expected": kappa_want})
        check(bool(np.all(np.isfinite(np.asarray(atoms.log_b)))),
              f"{cell}: all atom weights are finite and positive")
        outside = shift.target_mass_outside_training_box()
        check(outside is not None and outside < 0.01,
              f"{cell}: prior mass outside the training box stays negligible",
              {"mass_outside": outside})

        for obs_seed in sorted(set(KG.OBS_P1) | set(KG.OBS_P2)):
            dst = out / "observations" / KG.TASK / f"{KG.PRIOR_TYPE}_{pid}" \
                / f"obs_{obs_seed}.json"
            s = obs_src_dir / f"obs_{obs_seed}.json"
            check(dst.is_file() and sha256(dst) == sha256(s),
                  f"{cell}/obs_{obs_seed}: identical to the mild_0 source")

    # nesting across K, not just against the 64-point set
    for small, big in ((2, 4), (4, 8), (8, 16), (16, 32), (32, 64)):
        a = {tuple(np.round(v, 15)) for v in KG.centres(small, centre)}
        b = {tuple(np.round(v, 15)) for v in KG.centres(big, centre)}
        check(a <= b, f"K={small} centres are nested inside K={big}")

    if failures:
        raise RuntimeError(f"{len(failures)} K-family checks failed:\n  "
                           + "\n  ".join(failures[:20]))
    return {"n_checks": len(checks), "all_ok": True, "checks": checks}


def self_test(workdir: Path) -> dict:
    out = {}
    for name in ("rotate", "weights"):
        tmp = Path(tempfile.mkdtemp(prefix=f"r14k_{name}_", dir=str(workdir)))
        build(tmp)
        verify(tmp)  # clean tree must pass
        build(tmp, tamper=name)
        try:
            verify(tmp)
        except RuntimeError as exc:
            out[name] = {"rejected": True, "error_head": str(exc).splitlines()[1].strip()}
        else:
            out[name] = {"rejected": False}
        shutil.rmtree(tmp, ignore_errors=True)
    out["all_rejected"] = all(v["rejected"] for k, v in out.items()
                              if isinstance(v, dict))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True,
                   help="new directory for the cells (never the shipped data/)")
    p.add_argument("--report", default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    out = Path(args.out).resolve()
    shipped = benchmark.DATA_DIR.resolve()
    if out == shipped or shipped in out.parents:
        p.error(f"refusing to write into the shipped data directory: {out}")

    record = build(out)
    checks = verify(out)
    payload = {"record": record, "verification": checks}
    if args.self_test:
        with tempfile.TemporaryDirectory(prefix="r14k_selftest_") as tmp:
            payload["self_test"] = self_test(Path(tmp))
        if not payload["self_test"]["all_rejected"]:
            print(json.dumps(payload["self_test"], indent=2))
            raise SystemExit("self-test did not reject tampered input")

    report = Path(args.report) if args.report else (
        out / "prepare_report.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {len(record['cells'])} K cells under {out}")
    for k, v in record["cells"].items():
        print(f"  K={k:<3} prior_id={v['prior_id']:<4} "
              f"neighbour separation {v['neighbour_separation_in_sigma']:.2f} sigma")
    print(f"{checks['n_checks']} checks passed; report -> {report}")
    if args.self_test:
        for k, v in payload["self_test"].items():
            if isinstance(v, dict):
                print(f"self-test {k}: rejected={v['rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
