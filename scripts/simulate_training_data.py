#!/usr/bin/env python
"""Generate one Simformer training set from the upstream task definitions.

Split out from training because the sbibm tasks (Two Moons, Gaussian Linear)
simulate through torch/pyro while training is JAX-only, and the two cannot share
a process on this cluster.  OUP is pure JAX upstream but is drawn here too, so
that every task follows one path: simulate -> ``.npz`` -> train.

The random stream mirrors ``priorg/train.py``:  ``set_seed(seed)`` seeds torch
(which is what the sbibm simulators use), and ``key_data`` is the first
split of ``PRNGKey(seed)`` (which is what the JAX simulators use).

    python scripts/simulate_training_data.py --task gaussian_linear --seed 0 \
        --num-sims 10000 --upstream <prior-guide checkout> --out <directory>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT  # noqa: E402
from spectra.utils import git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(benchmark.TASKS))
    p.add_argument("--seed", type=int, required=True,
                   help="upstream's cfg.seed; doubles as the model id here")
    p.add_argument("--num-sims", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--upstream", default=None)
    args = p.parse_args()

    root = benchmark.upstream_root(args.upstream)
    for entry in (str(root), str(root / "priorg")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import jax  # noqa: E402
    import torch  # noqa: E402

    torch.set_num_threads(1)
    from priorg.sim.tasks.task import get_task, set_seed  # noqa: E402

    spec = benchmark.get_task(args.task)
    # ``priorg.sim.tasks.task.set_seed`` seeds torch/numpy/random but returns
    # nothing; ``priorg/train.py`` wraps it and additionally builds the JAX key
    # on CPU.  Reproduce that here.
    set_seed(args.seed)
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(args.seed)
    key, key_data = jax.random.split(key)

    t0 = time.perf_counter()
    task = get_task(name=args.task)
    data = task.get_data(args.num_sims, key=key_data)
    theta = np.asarray(data["theta"], dtype=np.float32)
    x = np.asarray(data["x"], dtype=np.float32)
    wall = time.perf_counter() - t0

    if theta.shape != (args.num_sims, spec.theta_dim):
        raise RuntimeError(f"theta shape {theta.shape} != {(args.num_sims, spec.theta_dim)}")
    if x.shape != (args.num_sims, spec.x_dim):
        raise RuntimeError(f"x shape {x.shape} != {(args.num_sims, spec.x_dim)}")
    if not (np.isfinite(theta).all() and np.isfinite(x).all()):
        raise RuntimeError("simulator produced non-finite values")

    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sims{args.num_sims}_seed{args.seed}.npz"
    np.savez_compressed(path, theta=theta, x=x)
    meta = {
        "task": args.task, "seed": args.seed, "num_sims": args.num_sims,
        "theta_dim": spec.theta_dim, "x_dim": spec.x_dim,
        "theta_mean": theta.mean(0).tolist(), "theta_std": theta.std(0).tolist(),
        "x_mean": x.mean(0).tolist()[:8], "x_std": x.std(0).tolist()[:8],
        "wall_clock_s": wall, "simulator": "upstream_literal",
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "torch": torch.__version__, "jax": jax.__version__,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"{args.task} seed {args.seed}: {theta.shape} / {x.shape} in {wall:.1f}s "
          f"-> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
