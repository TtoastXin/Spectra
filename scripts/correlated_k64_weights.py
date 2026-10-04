#!/usr/bin/env python
"""Practical Path-Space component weights on the ``K = 64`` correlated dictionary.

The prior is the compiled positive-atom dictionary ``two_moons / mixture_21``
(64 isotropic atoms on a Gauss--Hermite grid, read from the repository's
``data/``), the observations are the study's three, and the estimator is the
K-general ``pathspace_weights``, used unchanged at the practical budget (3
estimator seeds x 4096 trajectories x 400 power-grid steps, no Langevin in the
chain, chain seed ``1000 * seed + 17 * atom``).  All 64 atoms are estimated;
none is pruned by its prior coefficient, and a non-finite or collapsed atom is
recorded as is.

    python scripts/correlated_k64_weights.py --model-id 0 --obs-seeds 1000003 --out <root>
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
from scripts import correlated_k64_config as G  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def reference_weights_path(obs: int, root=None) -> Path:
    """Reference weights for one observation, under the caller's results root.

    ``root`` defaults to the configured one; passing ``--out`` elsewhere
    (``SPECTRA_RESULTS``) reads the references written under that same root
    instead of the repository default.
    """
    return (Path(root if root is not None else G.REFERENCE_ROOT) / "references"
            / G.K64_TASK / f"mixture_{G.K64_PRIOR_ID}" / f"obs_{obs}_weights.json")


def load_k64():
    shift = build_prior_shift(G.K64_TASK, G.K64_PRIOR_TYPE, G.K64_PRIOR_ID)
    return shift, atoms_mod.atoms_from_shift(shift)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", type=int, default=0)
    p.add_argument("--obs-seeds", default=",".join(str(o) for o in G.K64_OBS))
    p.add_argument("--seeds", default=",".join(str(s) for s in G.PS_SEEDS))
    p.add_argument("--trajectories", type=int, default=G.PS_TRAJECTORIES)
    p.add_argument("--steps", type=int, default=G.PS_STEPS)
    p.add_argument("--grid", default=G.PS_GRID)
    p.add_argument("--checkpoint", default=None,
                   help="model to estimate with; defaults to the shipped checkpoint of "
                        "--model-id.  Pass the same path the sampler uses, otherwise the "
                        "component weights and the component sampling belong to "
                        "different models")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--out", default=G.K64_ROOT)
    args = p.parse_args()
    out_root = Path(args.out)

    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    chain = [1000 * s + 17 * k for s in seeds for k in range(G.K64_NUM_ATOMS)]
    if len(set(chain)) != len(chain):
        raise SystemExit("chain seeds 1000*seed + 17*atom collide at K=64")
    obs_seeds = [int(v) for v in args.obs_seeds.split(",") if v.strip()]
    ckpt_path = args.checkpoint or G.k64_checkpoint(args.model_id)
    backbone = Backbone(load_checkpoint(ckpt_path), mode="conditional", nn=upstream_nn())
    shift, atoms = load_k64()
    if atoms.num_atoms != G.K64_NUM_ATOMS:
        raise SystemExit(f"mixture_{G.K64_PRIOR_ID} expanded into {atoms.num_atoms} atoms")
    common = {
        "round": "round15_k64_pathspace", "task": G.K64_TASK,
        "prior_type": G.K64_PRIOR_TYPE, "prior_id": G.K64_PRIOR_ID,
        "prior_json_sha256_16": benchmark.prior_json_hash(G.K64_TASK, G.K64_PRIOR_TYPE,
                                                          G.K64_PRIOR_ID),
        "model_id": args.model_id, "checkpoint": str(Path(ckpt_path).resolve()),
        "checkpoint_sha256_16": file_hash(ckpt_path),
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "devices": [str(d) for d in jax.devices()],
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), REPO / "scripts" / "correlated_k64_config.py",
             *sorted((REPO / "spectra").glob("*.py"))]),
        "estimator_source": "spectra/weights.pathspace_weights (imported unchanged)",
        "pruning": "none -- all 64 atoms estimated",
    }
    (out_root / "weights").mkdir(parents=True, exist_ok=True)
    for obs in obs_seeds:
        path = (out_root / "weights"
                / f"pathspace_{G.K64_TASK}_p{G.K64_PRIOR_ID}_o{obs}_model{args.model_id}.json")
        if args.skip_existing and path.is_file():
            print(f"[obs {obs} model {args.model_id}] skip (exists)", flush=True)
            continue
        _, x_o = benchmark.load_observation(G.K64_TASK, G.K64_PRIOR_TYPE, G.K64_PRIOR_ID, obs)
        ref = json.loads(reference_weights_path(obs, out_root).read_text())
        print(f"[obs {obs} model {args.model_id}] path-space over {atoms.num_atoms} atoms x "
              f"{len(seeds)} seeds", flush=True)
        t0 = time.perf_counter()
        res = pathspace_weights(backbone, x_o, atoms, seeds, args.trajectories, args.steps,
                                args.grid)
        alpha_ref = np.exp(np.asarray(ref["log_pi"], float))
        alpha_ref = alpha_ref / alpha_ref.sum()
        alpha = np.asarray(res["pi"], float)
        finite = bool(np.isfinite(alpha).all())
        res.update({**common, "obs_seed": obs,
                    "reference_weights_file": str(reference_weights_path(obs, out_root)),
                    "reference_weights_sha256_16": sha16(reference_weights_path(obs, out_root)),
                    "reference_pi": alpha_ref.tolist(),
                    "diagnostics": {
                        "l1_error": float(np.abs(alpha - alpha_ref).sum()) if finite else None,
                        "max_abs_error": float(np.max(np.abs(alpha - alpha_ref))) if finite else None,
                        "k_eff_ps": G.k_eff(alpha) if finite else None,
                        "k_eff_ref": G.k_eff(alpha_ref),
                        "n_mass_ge_0p05_ps": int((alpha >= G.K64_MASS_THRESHOLD).sum()) if finite else None,
                        "n_mass_ge_0p05_ref": int((alpha_ref >= G.K64_MASS_THRESHOLD).sum()),
                        "all_weights_finite": finite},
                    "total_wall_clock_s": time.perf_counter() - t0})
        path.write_text(json.dumps(res, indent=2, default=str))
        d = res["diagnostics"]
        print(f"[obs {obs} model {args.model_id}] status={res['status']} "
              f"L1={d['l1_error']} max={d['max_abs_error']} "
              f"K_eff PS/ref={d['k_eff_ps']}/{d['k_eff_ref']:.2f} "
              f"estimator wall {res['wall_clock_s']:.1f}s -> {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
