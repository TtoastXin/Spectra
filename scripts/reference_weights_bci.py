#!/usr/bin/env python
"""Reference component weights for the BCI two-component mixture grid.

``scripts/reference_bci.py`` writes the tempered-SMC posterior samples for the
ten shipped BCI (``bav``) two-component mixture priors on three observations
each, but not the ``obs_<seed>_weights.json`` that the mixture runner reads (the
other five tasks get one from their own reference job).  This script produces
that file for the thirty cells without new sampling:

* the samples are the existing references, symlinked into the output root;
* the estimator is ``quadrature.responsibility_masses_from_samples``, the same
  call ``scripts/reference_posterior.py`` uses for the SLCP references;
* the grade follows the SMC convention: the spread of the per-seed log-odds.

The result is a finite-sample responsibility estimate on the reference
posterior sample, not an analytically exact alpha.

The reference tree is only read.  The output root gets
``references/bav/mixture_<pid>/`` with symlinks to the reference ``.npz`` /
``.json`` next to the new weights file, so ``--ref-roots`` and
``compute_metrics.py`` both resolve, plus the geometry table and, when
``--screen`` is given, a consistency check against a screening ``pi_k`` table.

    python scripts/reference_weights_bci.py \\
        --references results/mixture_controlled \\
        --out results/mixture_controlled
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, coordinates, quadrature  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASK = "bav"
OBS = (1000000, 1000001, 1000002)
LOG_ZERO = -700.0


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def link(src: Path, dst: Path) -> None:
    if dst.exists() and dst.resolve() == src.resolve():
        return
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--references", required=True,
                   help="root holding references/bav/mixture_* from the BCI reference job")
    p.add_argument("--screen", default=None,
                   help="optional screening table (prior, obs_seed, pi_0, pi_1) for a "
                        "consistency check")
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    r8, out = Path(args.references), Path(args.out)
    (out / "references" / TASK).mkdir(parents=True, exist_ok=True)
    (out / "refcheck").mkdir(parents=True, exist_ok=True)

    screen = {}
    for r in (csv.DictReader(open(args.screen)) if args.screen else ()):
        if r["task"] == TASK:
            screen[(r["prior"], int(r["obs_seed"]))] = r

    geometry, audit, missing = [], [], []
    for pid in range(10):
        src_dir = r8 / "references" / TASK / f"mixture_{pid}"
        dst_dir = out / "references" / TASK / f"mixture_{pid}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shift = build_prior_shift(TASK, "mixture", pid)
        atoms = atoms_mod.atoms_from_shift(shift)
        if atoms.num_atoms != 2:
            raise SystemExit(f"mixture_{pid}: expected 2 atoms, got {atoms.num_atoms}")
        for obs in OBS:
            npz, meta_json = src_dir / f"obs_{obs}.npz", src_dir / f"obs_{obs}.json"
            if not (npz.is_file() and meta_json.is_file()):
                missing.append(str(npz))
                continue
            link(npz, dst_dir / f"obs_{obs}.npz")
            link(meta_json, dst_dir / f"obs_{obs}.json")

            d = np.load(npz)
            samples = np.asarray(d["samples"], float)
            seed_keys = sorted(k for k in d.files if k.startswith("samples_seed_"))
            resp = quadrature.responsibility_masses_from_samples(
                samples, atoms, num_shards=args.num_shards)
            per_seed = [quadrature.responsibility_masses_from_samples(
                np.asarray(d[k], float), atoms, num_shards=2) for k in seed_keys]
            run_log_odds = [float(r["log_odds"]) for r in per_seed]
            spread = (float(np.std(run_log_odds, ddof=1)) if len(run_log_odds) > 1
                      else float("nan"))
            grade = ("exact_or_high_confidence"
                     if (len(run_log_odds) > 1 and spread < 0.1)
                     else "high_quality_approximate")

            pi = np.asarray(resp["pi"], float)
            with np.errstate(divide="ignore"):
                lp = np.log(pi)
            lp = np.where(np.isfinite(lp), lp, LOG_ZERO)
            meta = json.loads(meta_json.read_text())
            payload = {
                "log_pi": lp.tolist(), "pi": np.exp(lp).tolist(),
                "source": "round12_responsibility_on_round8_bci_reference_smc",
                "num_atoms": int(atoms.num_atoms),
                "task": TASK, "prior_type": "mixture", "prior_id": pid, "obs_seed": obs,
                "log_odds": float(resp["log_odds"]),
                "grade": grade,
                "log_b_diff": float(atoms.log_b[0] - atoms.log_b[1]),
                # provenance and finite-sample resolution of this estimate
                "estimator": "quadrature.responsibility_masses_from_samples",
                "estimator_note": ("finite-sample responsibility mass on the certified "
                                   "posterior sample; not an analytically exact alpha"),
                "num_samples": int(resp["num_samples"]),
                "num_shards": int(resp["num_shards"]),
                "mc_se_pi": np.asarray(resp["mc_se_pi"], float).tolist(),
                "log_odds_se": float(resp["log_odds_se"]),
                "per_seed_log_odds": run_log_odds,
                "run_log_odds_spread_sd": spread,
                "reference_npz": str(npz.resolve()),
                "reference_npz_sha256_16": file_hash(npz),
                "reference_certificate_grade": meta.get("certificate", {}).get("grade"),
                "prior_json_sha256_16": benchmark.prior_json_hash(TASK, "mixture", pid),
                "repo_git": git_state(REPO),
            }
            (dst_dir / f"obs_{obs}_weights.json").write_text(
                json.dumps(payload, indent=2, default=str))

            geometry.append({
                "task": TASK, "prior_id": pid, "obs_seed": obs,
                "npz_exists": True, "weights_exists": True,
                "alpha_0": float(pi[0]), "alpha_1": float(pi[1]),
                "alpha_min": float(pi.min()),
                "entropy_nats": float(-np.sum(pi[pi > 0] * np.log(pi[pi > 0]))),
                "stratum": ("substantive_mixture" if pi.min() >= 0.05
                            else "dominant_component"),
                "extreme_allocation_lt_0.01": bool(pi.min() < 0.01),
                "source": payload["source"], "grade": grade,
                "reference_kind": meta.get("kind"),
                "reference_grade": meta.get("certificate", {}).get("grade"),
                "refinement": "",
                "smc_run_log_odds_spread_sd": spread,
                "smc_min_stage_ess": min(
                    (dg.get("min_stage_ess", float("nan"))
                     for dg in meta.get("certificate", {}).get("target_diagnostics", [])),
                    default=None),
            })

            s = screen.get((f"mixture_{pid}", obs))
            if s is not None:
                s_pi = np.array([float(s["pi_0"]), float(s["pi_1"])], float)
                audit.append({
                    "prior_id": pid, "obs_seed": obs,
                    "pi_min_round12": float(pi.min()), "pi_min_round8_screen": float(s_pi.min()),
                    "l1_gap": float(np.abs(pi - s_pi).sum()),
                    "dominant_same": bool(int(np.argmax(pi)) == int(np.argmax(s_pi))),
                    "both_below_0.05": bool(pi.min() < 0.05 and s_pi.min() < 0.05),
                })

    if missing:
        raise SystemExit("missing BCI references:\n" + "\n".join(missing))

    keys = list(geometry[0].keys())
    with open(out / "refcheck" / "reference_geometry_bav_30.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(geometry)

    strata = {}
    for r in geometry:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    summary = {
        "pairs": len(geometry), "pairs_expected": 30, "strata": strata,
        "alpha_min_max_over_cells": max(r["alpha_min"] for r in geometry),
        "alpha_min_cell": max(geometry, key=lambda r: r["alpha_min"])["prior_id"],
        "n_substantive_ge_0.05": sum(1 for r in geometry if r["alpha_min"] >= 0.05),
        "grades": {g: sum(1 for r in geometry if r["grade"] == g)
                   for g in sorted({r["grade"] for r in geometry})},
        "screen_audit": {
            "n_compared": len(audit),
            "dominant_component_agrees": sum(1 for a in audit if a["dominant_same"]),
            "both_below_0.05_agrees": sum(1 for a in audit if a["both_below_0.05"]),
            "max_l1_gap": max((a["l1_gap"] for a in audit), default=float("nan")),
            "max_pi_min_round12": max((a["pi_min_round12"] for a in audit), default=float("nan")),
            "max_pi_min_round8_screen": max((a["pi_min_round8_screen"] for a in audit),
                                            default=float("nan")),
        },
        "repo_git": git_state(REPO),
    }
    (out / "refcheck" / "bav_weights_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    if audit:
        with open(out / "refcheck" / "bav_screen_consistency.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(audit[0].keys()))
            w.writeheader()
            w.writerows(audit)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
