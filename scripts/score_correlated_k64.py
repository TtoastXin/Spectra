#!/usr/bin/env python
"""Score ``spectra_ref`` and ``spectra_ps`` against both correlated-prior references.

Scoring follows ``scripts/score_correlated.py``: every row is scored against the
true target ``p_0 * r`` (the requested correlated Gaussian; primary here) and
the compiled target ``p_0 * r~`` (what the K=64 dictionary represents; a
secondary diagnostic that separates the quadrature approximation from weight
estimation and sampling).  Its ``score`` function (C2ST seeds 0..4 on balanced
reference subsamples, MMTV and moments on the whole bank) is imported
unchanged.  That script only globs ``model_0`` and one root; this wrapper walks
the ``K = 64`` roots for all three checkpoints and both weight sources.

Samples come from ``scripts/sample_correlated.py``, run once with the reference
component weights (``<root>/reference``) and once with the Path-Space weights
(``<root>/pathspace``).  That runner names its Spectra row
``eamt_compiled_reference_ms<s>`` regardless of the weights it was given, so the
weight source is taken from the root and checked: the ``eamt_log_pi`` a
manifest records must equal the ``log_pi`` of that source's weight file, or the
script exits non-zero.  The Base / PG-truecov / PG-FullCov-truecov rows the
runner also writes are not scored; their copies in the two roots are only
checked for identical samples, since both roots share seeds and base bank.

    python scripts/score_correlated_k64.py --root results/correlated_prior --workers 18
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra.utils import git_state  # noqa: E402
from scripts import correlated_k64_config as G  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPECTRA_ROW = "eamt_compiled_reference"
BYPRODUCT_ROWS = ("base", "pg_truecov", "a_full_truecov")

_W = {}


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from scripts.score_correlated import score, upstream_metrics
    _W["m"] = upstream_metrics()
    _W["score"] = score


def references(obs: int, root=None) -> dict:
    """The two reference banks, under the caller's results root."""
    root = Path(root if root is not None else G.REFERENCE_ROOT)
    return {
        "true_target": root / "compilation" / f"true_target_{G.K64_TASK}_obs{obs}.npz",
        "compiled_target": (root / "references" / G.K64_TASK / f"mixture_{G.K64_PRIOR_ID}"
                            / f"obs_{obs}.npz"),
    }


def _score(item):
    base, npz, ref_path = item
    s = np.load(npz)["samples"]
    ref = np.load(ref_path)["samples"]
    return {**base, **_W["score"](s, ref, _W["m"])}


def weights_file(source: str, obs: int, model: int, root: Path) -> Path:
    if source == "reference":
        # same root as the path-space weights below, so a custom --root does not
        # fall back to whatever is in the repository's results/
        return (Path(root) / "references" / G.K64_TASK
                / f"mixture_{G.K64_PRIOR_ID}" / f"obs_{obs}_weights.json")
    return (root / "weights"
            / f"pathspace_{G.K64_TASK}_p{G.K64_PRIOR_ID}_o{obs}_model{model}.json")


def sample_dir(root: Path, source: str, row: str, seed: int, model: int, obs: int) -> Path:
    return (root / source / "samples" / G.K64_TASK / f"{row}_ms{seed}" / f"model_{model}"
            / f"mixture_{G.K64_PRIOR_ID}" / f"obs_{obs}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=G.K64_ROOT)
    p.add_argument("--models", default=",".join(str(m) for m in G.K64_MODELS))
    p.add_argument("--obs", default=",".join(str(o) for o in G.K64_OBS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--self-test", action="store_true",
                   help="also re-run the weights-provenance gate with the sources swapped")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.root)
    models = [int(v) for v in args.models.split(",") if v.strip()]
    obs_list = [int(v) for v in args.obs.split(",") if v.strip()]
    out = Path(args.out) if args.out else root / "metrics" / "k64.csv"

    work, failures, provenance, byproduct = [], [], [], []
    for obs in obs_list:
        refs = references(obs, root)
        work.append(({"row": "(compiled target vs true target)", "weight_source": None,
                      "model_id": None, "obs_seed": obs, "seed": -1,
                      "reference_kind": "true_target"},
                     str(refs["compiled_target"]), str(refs["true_target"])))
        for model in models:
            for source in G.K64_WEIGHT_SOURCES:
                wpath = weights_file(source, obs, model, root)
                if not wpath.is_file():
                    failures.append(f"missing weights {wpath}")
                    continue
                want = [float(v) for v in json.loads(wpath.read_text())["log_pi"]]
                for seed in G.K64_SAMPLER_SEEDS:
                    d = sample_dir(root, source, SPECTRA_ROW, seed, model, obs)
                    npz, man_path = d / f"{G.K64_TAG}.npz", d / f"{G.K64_TAG}.json"
                    if not npz.is_file():
                        failures.append(f"missing sample {npz}")
                        continue
                    man = json.loads(man_path.read_text())
                    got = [float(v) for v in man.get("eamt_log_pi") or []]
                    ok = got == want
                    provenance.append({"weight_source": source, "model_id": model,
                                       "obs_seed": obs, "seed": seed,
                                       "manifest_log_pi_equals_weights_file": ok,
                                       "weights_file": str(wpath)})
                    if not ok:
                        failures.append(f"{npz}: eamt_log_pi is not the {source} weights")
                        continue
                    for kind, ref in refs.items():
                        work.append(({"row": G.K64_ROW_LABEL[source], "weight_source": source,
                                      "model_id": model, "obs_seed": obs, "seed": seed,
                                      "reference_kind": kind,
                                      "nan_or_inf": man.get("nan_or_inf"),
                                      "num_steps": man["sampler"]["num_steps"],
                                      "langevin_steps": man["sampler"]["langevin_steps"],
                                      "sample_file": str(npz)},
                                     str(npz), str(ref)))
            for row in BYPRODUCT_ROWS:
                for seed in G.K64_SAMPLER_SEEDS:
                    a = sample_dir(root, "reference", row, seed, model, obs) / f"{G.K64_TAG}.npz"
                    b = sample_dir(root, "pathspace", row, seed, model, obs) / f"{G.K64_TAG}.npz"
                    if a.is_file() and b.is_file():
                        byproduct.append({"row": row, "model_id": model, "obs_seed": obs,
                                          "seed": seed, "bitwise_across_roots":
                                          np.load(a)["samples"].tobytes()
                                          == np.load(b)["samples"].tobytes()})

    if args.self_test:
        # the provenance gate on real files with the two sources swapped: every
        # comparison must now fail, or the gate could not tell the weights apart
        swapped = []
        for obs in obs_list:
            for model in models:
                for source, other in (("reference", "pathspace"), ("pathspace", "reference")):
                    wpath = weights_file(other, obs, model, root)
                    d = sample_dir(root, source, SPECTRA_ROW, G.K64_SAMPLER_SEEDS[0], model, obs)
                    if not (wpath.is_file() and (d / f"{G.K64_TAG}.json").is_file()):
                        continue
                    want = [float(v) for v in json.loads(wpath.read_text())["log_pi"]]
                    got = [float(v) for v in json.loads(
                        (d / f"{G.K64_TAG}.json").read_text()).get("eamt_log_pi") or []]
                    swapped.append(got == want)
        st_ = {"n_swapped_comparisons": len(swapped), "n_wrongly_accepted": sum(swapped),
               "pass": bool(swapped) and not any(swapped)}
        print(f"[self-test] provenance gate with sources swapped: {st_}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".selftest.json").write_text(json.dumps(st_, indent=2))
        if not st_["pass"]:
            failures.append("self-test: swapped weight sources were not all rejected")

    if failures:
        print("K64 SCORING GATE FAILED:\n  " + "\n  ".join(failures), flush=True)
        (out.parent).mkdir(parents=True, exist_ok=True)
        out.with_suffix(".failures.json").write_text(json.dumps(
            {"failures": failures, "provenance": provenance}, indent=2))
        return 1

    print(f"{len(work)} scorings with {args.workers} worker(s)", flush=True)
    if args.workers <= 1:
        _init()
        rows = [_score(w) for w in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
            rows = list(pool.map(_score, work))
    for r in rows:
        if r["row"].startswith("spectra"):
            print(f"  {r['row']:<11} m{r['model_id']} o{r['obs_seed']} s{r['seed']} "
                  f"{r['reference_kind']:<15} C2ST={r['c2st']:.4f} MMTV={r['mmtv']:.4f}",
                  flush=True)

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
        "kind": "round15_k64_scoring",
        "scorer": "scripts/score_correlated.score (imported unchanged)",
        "primary_reference": "true_target", "secondary_reference": "compiled_target",
        "weight_provenance_gate": provenance,
        "byproduct_rows_bitwise_across_roots": byproduct,
        "repo_git": git_state(REPO),
    }, indent=2, default=str))
    print(f"wrote {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
