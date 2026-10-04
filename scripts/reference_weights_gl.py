#!/usr/bin/env python
"""Analytic reference atom weights for a Gaussian Linear mixture cell.

``scripts/reference_closed_form.py`` writes the analytic posterior samples for
Gaussian Linear but no ``obs_<seed>_weights.json``.  The mixture runner reads the
weights file uniformly for every task, so this writes it in the same schema from
the closed form (``atoms.analytic_atom_weights``), next to the reference npz.

    python scripts/reference_weights_gl.py --task gaussian_linear --prior-id 0 \\
        --obs-seeds 1000000,1000001,1000002 --out results/mixture_controlled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402

LOG_ZERO = -700.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=("gaussian_linear", "gaussian_linear_high"))
    p.add_argument("--prior-id", type=int, required=True)
    p.add_argument("--obs-seeds", default="1000000,1000001,1000002")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    shift = build_prior_shift(args.task, "mixture", args.prior_id)
    atoms = atoms_mod.atoms_from_shift(shift)
    out_dir = Path(args.out) / "references" / args.task / f"mixture_{args.prior_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for obs in [int(v) for v in args.obs_seeds.split(",") if v.strip()]:
        _, x_o = benchmark.load_observation(args.task, "mixture", args.prior_id, obs)
        pi = atoms_mod.analytic_atom_weights(np.asarray(x_o, float), shift)
        lp = np.log(np.maximum(pi, 0.0))
        lp = np.where(np.isfinite(lp), lp, LOG_ZERO)
        blob = {
            "log_pi": lp.tolist(), "pi": pi.tolist(),
            "source": "round10_gl_analytic", "num_atoms": atoms.num_atoms,
            "task": args.task, "prior_type": "mixture", "prior_id": args.prior_id,
            "obs_seed": obs, "grade": "analytic",
            "log_odds": float(lp[0] - lp[1]) if atoms.num_atoms == 2 else float("nan"),
            "log_b_diff": float(atoms.log_b[0] - atoms.log_b[1]) if atoms.num_atoms == 2 else float("nan"),
        }
        if atoms.num_atoms == 2:
            blob["delta_star"] = blob["log_odds"] - blob["log_b_diff"]
        path = out_dir / f"obs_{obs}_weights.json"
        path.write_text(json.dumps(blob, indent=2))
        print(f"[{args.task}/mixture_{args.prior_id}/obs_{obs}] pi*={np.round(pi, 6).tolist()} -> {path}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
