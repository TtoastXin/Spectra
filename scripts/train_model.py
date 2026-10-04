#!/usr/bin/env python
"""Train one Simformer backbone on a pre-simulated task dataset.  JAX only.

Upstream's benchmark recipe, unchanged: VE-SDE, ``structured_random``
conditioning, ``3 x num_rows`` steps capped at 100k, early stopping on the
validation ratio.  ``--seed`` doubles as the model id, matching how upstream's
runners select ``models/<task>/model_<id>.pkl``.  Upstream ships only
``gaussian_linear/model_0.pkl``, without its training seed, so the model seeds
here are our own and the upstream checkpoint serves only as a parity check.

    python scripts/train_model.py \
        --data <training data>/gaussian_linear/sims10000_seed0.npz \
        --task gaussian_linear --seed 0 --out checkpoints

Two Moons was trained with ``XLA_FLAGS=--xla_gpu_enable_triton_gemm=false``:
in our environment (JAX 0.4.28 on an NVIDIA A100) the default XLA:GPU stack
segfaults in its backward pass.  The shipped Two Moons checkpoints were trained
that way, and retraining under the flag reproduces them exactly.  The other
tasks were trained with the default flags, under which XLA autotunes its GPU
kernels on the device; retraining them on other hardware gives an equally
trained model, not an exact copy of the shipped one.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark  # noqa: E402
from spectra.simformer import (  # noqa: E402
    UPSTREAM_COMMIT,
    SimformerConfig,
    TrainConfig,
    VESchedule,
    _upstream_root,
    upstream_nn,
)
from spectra.training import train_model  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps-override", type=int, default=None)
    args = p.parse_args()

    root = _upstream_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sim.utils.conditional_mask import get_condition_mask_fn  # noqa: E402

    spec = benchmark.get_task(args.task)
    blob = np.load(args.data)
    theta, x = blob["theta"], blob["x"]
    if theta.shape[-1] != spec.theta_dim or x.shape[-1] != spec.x_dim:
        p.error(f"{args.data} does not match {args.task} dims")

    nn = upstream_nn()
    schedule = VESchedule()
    model_cfg = SimformerConfig()
    train_cfg = TrainConfig(num_sims=theta.shape[0])
    cmask_fn = get_condition_mask_fn("structured_random")

    print(f"training {args.task} seed {args.seed} on {theta.shape[0]} rows "
          f"({jax.devices()})", flush=True)
    t0 = time.perf_counter()
    res = train_model(
        theta, x, seed=args.seed, schedule=schedule, model_cfg=model_cfg,
        train_cfg=train_cfg, nn=nn, condition_mask_fn=cmask_fn,
        steps_override=args.steps_override,
    )
    wall = time.perf_counter() - t0

    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"model_{args.seed}.pkl"
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, res.params),
        "theta_dim": spec.theta_dim, "x_dim": spec.x_dim,
        "schedule": asdict(schedule), "model_cfg": asdict(model_cfg),
        "marginal_end_mean": res.marginal_end_mean,
        "marginal_end_std": res.marginal_end_std,
        "meta": {
            "task": args.task, "seed": args.seed, "num_sims": int(theta.shape[0]),
            "train_cfg": asdict(train_cfg),
            "condition_mask_fn": "structured_random",
            "steps_run": res.steps_run, "early_stopped": res.early_stopped,
            "timing": res.timing, "wall_clock_s": wall,
            "history": res.history,
            "data_path": str(Path(args.data).resolve()),
            "upstream_commit": UPSTREAM_COMMIT,
            "repo_git": git_state(REPO),
            "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
            "host": platform.node(),
        },
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    path.with_suffix(".json").write_text(json.dumps(payload["meta"], indent=2))
    print(f"steps_run={res.steps_run} early_stopped={res.early_stopped} "
          f"{wall / 60:.1f} min -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
