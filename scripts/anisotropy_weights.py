#!/usr/bin/env python
"""Anisotropy study: practical path-space component weights on the compiled dictionaries.

The estimator is the K-general ``spectra.weights.pathspace_weights``, used
unchanged at the practical budget of the mixture studies (3 estimator seeds x
4096 trajectories x 400 power-grid steps, no Langevin in the chain, chain seed
``1000 * seed + 17 * atom``), as for the correlated prior's K=64 weights.
Every atom is estimated, with no pruning.  Diagnostics against the grid reference weights
(``reference_posterior.py`` on ``mixture_<cell>``) are written next to the
estimate.

    python scripts/anisotropy_weights.py --cells 62,63,64,65 --model-id 0 \\
        --obs-seeds 1000000 --out results/anisotropy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.utils import file_hash, git_state  # noqa: E402
from spectra.weights import pathspace_weights  # noqa: E402
from scripts import anisotropy_config as G  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def reference_weights_path(root: Path, cell: int, obs: int) -> Path:
    return root / "references" / G.TASK / f"mixture_{cell}" / f"obs_{obs}_weights.json"


def weights_path(root: Path, cell: int, obs: int, model: int) -> Path:
    return root / "weights" / f"pathspace_{G.TASK}_c{cell}_o{obs}_model{model}.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", default=",".join(str(c) for c in G.ANISO_CELLS))
    p.add_argument("--model-id", type=int, default=0)
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint of --model-id (default: the shipped one)")
    p.add_argument("--obs-seeds", default=",".join(str(o) for o in G.OBS_SEEDS))
    p.add_argument("--seeds", default=",".join(str(s) for s in G.PS_SEEDS))
    p.add_argument("--trajectories", type=int, default=G.PS_TRAJECTORIES)
    p.add_argument("--steps", type=int, default=G.PS_STEPS)
    p.add_argument("--grid", default=G.PS_GRID)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--out", default=G.ROOT)
    args = p.parse_args()
    root = Path(args.out)
    cells_rec = G.load_cells()
    cells = [int(v) for v in args.cells.split(",") if v.strip()]
    obs_seeds = [int(v) for v in args.obs_seeds.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]

    ckpt_path = args.checkpoint or G.checkpoint(args.model_id)
    backbone = Backbone(load_checkpoint(ckpt_path), mode="conditional", nn=upstream_nn())
    (root / "weights").mkdir(parents=True, exist_ok=True)
    for cell in cells:
        shift = build_prior_shift(G.TASK, "mixture", cell)
        atoms = atoms_mod.atoms_from_shift(shift)
        want_k = cells_rec[cell]["compiled"]["num_atoms"]
        if atoms.num_atoms != want_k:
            raise SystemExit(f"mixture_{cell} expanded into {atoms.num_atoms} atoms, "
                             f"cells.json records {want_k}")
        chain = [1000 * s + 17 * k for s in seeds for k in range(atoms.num_atoms)]
        if len(set(chain)) != len(chain):
            raise SystemExit(f"chain seeds 1000*seed + 17*atom collide at K={atoms.num_atoms}")
        common = {
            "round": "round16_anisotropy_pathspace", "task": G.TASK, "cell": cell,
            "gamma": cells_rec[cell]["gamma"], "orientation": cells_rec[cell]["orientation"],
            "prior_type": "mixture", "prior_id": cell,
            "prior_json_sha256_16": benchmark.prior_json_hash(G.TASK, "mixture", cell),
            "model_id": args.model_id, "checkpoint": str(Path(ckpt_path).resolve()),
            "checkpoint_sha256_16": file_hash(ckpt_path),
            "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
            "devices": [str(d) for d in jax.devices()],
            "source_sha256_16": benchmark.source_hashes(
                [Path(__file__), REPO / "scripts" / "anisotropy_config.py",
                 *sorted((REPO / "spectra").glob("*.py"))]),
            "estimator_source": "spectra.weights.pathspace_weights (imported unchanged)",
            "pruning": f"none -- all {atoms.num_atoms} atoms estimated",
        }
        for obs in obs_seeds:
            path = weights_path(root, cell, obs, args.model_id)
            if args.skip_existing and path.is_file():
                print(f"[cell {cell} obs {obs} model {args.model_id}] skip (exists)", flush=True)
                continue
            ref_path = reference_weights_path(root, cell, obs)
            if not ref_path.is_file():
                raise SystemExit(f"missing reference weights {ref_path}")
            _, x_o = benchmark.load_observation(G.TASK, "mixture", cell, obs)
            print(f"[cell {cell} obs {obs} model {args.model_id}] path-space over "
                  f"{atoms.num_atoms} atoms x {len(seeds)} seeds", flush=True)
            t0 = time.perf_counter()
            res = pathspace_weights(backbone, x_o, atoms, seeds, args.trajectories,
                                    args.steps, args.grid)
            alpha_ref = np.exp(np.asarray(json.loads(ref_path.read_text())["log_pi"], float))
            alpha_ref = alpha_ref / alpha_ref.sum()
            alpha = np.asarray(res["pi"], float)
            finite = bool(np.isfinite(alpha).all())
            ess = [a["u_ess_normalized"] for s in res["per_seed"] for a in s["atoms"]]
            res.update({**common, "obs_seed": obs,
                        "reference_weights_file": str(ref_path),
                        "reference_weights_sha256_16": sha16(ref_path),
                        "reference_pi": alpha_ref.tolist(),
                        "diagnostics": {
                            "l1_error": float(np.abs(alpha - alpha_ref).sum()) if finite else None,
                            "max_abs_error": float(np.max(np.abs(alpha - alpha_ref))) if finite else None,
                            "k_eff_ps": G.k_eff(alpha) if finite else None,
                            "k_eff_ref": G.k_eff(alpha_ref),
                            "n_mass_ge_0p05_ps": int((alpha >= G.MASS_THRESHOLD).sum()) if finite else None,
                            "n_mass_ge_0p05_ref": int((alpha_ref >= G.MASS_THRESHOLD).sum()),
                            "u_ess_normalized_min": float(np.nanmin(ess)),
                            "u_ess_normalized_median": float(np.nanmedian(ess)),
                            "all_weights_finite": finite},
                        "total_wall_clock_s": time.perf_counter() - t0})
            path.write_text(json.dumps(res, indent=2, default=str))
            d = res["diagnostics"]
            print(f"[cell {cell} obs {obs} model {args.model_id}] status={res['status']} "
                  f"L1={d['l1_error']} max={d['max_abs_error']} K_eff PS/ref="
                  f"{d['k_eff_ps']}/{d['k_eff_ref']:.2f} wall {res['wall_clock_s']:.1f}s",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
