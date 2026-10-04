#!/usr/bin/env python
"""Write the anisotropy cells: the exact-target records and the compiled dictionaries.

* ``anisotropy/cells.json`` records each exact target ``aniso_<id>`` (mu = the
  ``mild_20`` centre, diagonal covariance ``Sigma_gamma^(+/-)``).  It is a plain
  record rather than a prior JSON because ``build_prior_shift`` refuses an
  anisotropic target.
* ``priors/two_moons/mixture_<id>.json`` is the tie-up compiled dictionary of each
  compiled cell (medium capacity), written as an ordinary isotropic mixture so
  ``build_prior_shift``, ``atoms_from_shift``, ``reference_posterior.py`` and the
  Spectra hook consume it as is; its observations are copies of the ten
  ``mild_20`` observations (which are the ``mild_0`` ones).

Everything is written into a fresh ``--out`` directory; the shipped ``data/``
directory is only read, and the shipped cells are the output of this script.
Refuses (exit 1) if the compiler gate fails or a source file is missing.  The
dictionaries depend on the prior only; no observation enters.

    python scripts/prepare_anisotropy_priors.py --out <new directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark, compile_atoms  # noqa: E402
from scripts import anisotropy_compiler as RC  # noqa: E402
from scripts import anisotropy_config as G  # noqa: E402


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True,
                   help="new directory for the cells (never the shipped data/)")
    args = p.parse_args()
    data = Path(args.out).resolve()
    shipped = benchmark.DATA_DIR.resolve()
    if data == shipped or shipped in data.parents:
        p.error(f"refusing to write into the shipped data directory: {data}")
    data.mkdir(parents=True, exist_ok=True)

    gate = RC.gate(self_test=True)
    (data / "compiler_gate.json").write_text(json.dumps(gate, indent=2))
    if not gate["pass"]:
        print("compiler gate FAILED: " + "; ".join(gate["failures"]), flush=True)
        return 1

    source = benchmark.data_root(G.TASK)
    src_prior = (source / "priors" / G.TASK
                 / f"{G.SOURCE_PRIOR_TYPE}_{G.SOURCE_PRIOR_ID}.json")
    src_obs = (source / "observations" / G.TASK
               / f"{G.SOURCE_PRIOR_TYPE}_{G.SOURCE_PRIOR_ID}")
    missing = [str(src_prior)] if not src_prior.is_file() else []
    missing += [str(src_obs / f"obs_{o}.json") for o in G.OBS_SEEDS
                if not (src_obs / f"obs_{o}.json").is_file()]
    if missing:
        print("MISSING " + ", ".join(missing), flush=True)
        return 1

    pri = json.loads(src_prior.read_text())
    mu = np.asarray(pri["mu"], float)
    sig = np.asarray(pri["sigma"], float)
    if not np.allclose(sig, sig[0], rtol=0, atol=0):
        print(f"source prior is not isotropic: {sig}", flush=True)
        return 1
    s = float(sig[0])
    box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

    def copy_obs(prior_type: str, pid: int) -> dict:
        dst = data / "observations" / G.TASK / f"{prior_type}_{pid}"
        dst.mkdir(parents=True, exist_ok=True)
        out = {}
        for o in G.OBS_SEEDS:
            shutil.copyfile(src_obs / f"obs_{o}.json", dst / f"obs_{o}.json")
            out[o] = sha16(dst / f"obs_{o}.json")
        return out

    (data / "priors" / G.TASK).mkdir(parents=True, exist_ok=True)
    obs_hashes = {o: sha16(src_obs / f"obs_{o}.json") for o in G.OBS_SEEDS}

    cells = {}
    for cid, spec in G.CELLS.items():
        var = G.cov_diag(cid, s)
        cov = np.diag(var)
        prior = compile_atoms.CorrelatedGaussianPrior(
            mu=mu, cov=cov, box=box,
            rule=(f"Round 16 cell {cid}: mu = Round 14 mild_20 centre; "
                  f"Sigma = diag{tuple(round(v, 12) for v in var)}, "
                  f"gamma = {spec['gamma']}, orientation = {spec['orientation']}"))
        rec = {"cell": cid, **spec, "s": s, "mu": mu.tolist(), "cov": cov.tolist(),
               "per_axis_sd": np.sqrt(var).tolist(),
               "det_cov": float(np.linalg.det(cov)), "s4": s ** 4,
               "condition_number": float(np.linalg.cond(cov)),
               "prior_mass_inside_training_box": prior.mass_inside_box()}
        if cid in G.COMPILED_CELLS:
            d, crec = RC.compile_dictionary(
                prior, spread_factor=compile_atoms.CAPACITIES[G.CAPACITY])
            err = compile_atoms.approximation_error(d)
            js = d.as_prior_json(G.TASK)
            js["compilation"]["node_order_rule"] = "round16 tie_up: floor(snap(raw)) + 2"
            js["compilation"]["node_orders"] = crec["node_orders"]
            js["round16_cell"] = cid
            path = data / "priors" / G.TASK / f"mixture_{cid}.json"
            path.write_text(json.dumps(js, indent=2))
            copy_obs("mixture", cid)
            rec["compiled"] = {**crec, "capacity": G.CAPACITY,
                               "prior_json": str(path.relative_to(data)),
                               "prior_json_sha256_16": sha16(path),
                               "min_prior_coefficient": float(np.exp(d.log_b).min()),
                               "eps_r_bound": err}
            print(f"[cell {cid}] gamma={spec['gamma']} {spec['orientation']:<5} "
                  f"orders={crec['node_orders']} K={crec['num_atoms']} "
                  f"eps_r/Z bound={err['eps_r_over_Z_bound_uniform']:.4f}", flush=True)
        cells[cid] = rec

    record = {
        "round": "round16_anisotropy",
        "source_prior": str(src_prior.relative_to(source)),
        "source_prior_sha256_16": sha16(src_prior),
        "observation_sha256_16": obs_hashes, "cells": cells,
        "compiler_gate": {k: gate[k] for k in ("pass", "fidelity",
                                                "self_test_tie_down_rejected")},
    }
    (data / "anisotropy").mkdir(parents=True, exist_ok=True)
    (data / "anisotropy" / "cells.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"-> {data / 'anisotropy' / 'cells.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
