#!/usr/bin/env python
"""The positive-mixture compiler with the node-count tie resolved upward.

``compile_atoms.compile_dictionary`` sets the Gauss--Hermite order per principal
axis of ``Sigma_c`` as ``ceil(spread * sd_c / sqrt(lambda)) + 1``.  With
``lambda = 0.9 lambda_min(Sigma)`` the short axis always has
``sd_c / sqrt(lambda) = sqrt(0.1 / 0.9) = 1/3`` exactly, so for every 2-D prior
the criterion is exactly 1 (small) or 2 (medium) and the ceiling adds a node or
not depending on the last floating-point bit.  The correlated prior's
dictionary happens to land on ``2.0000000000000018`` -> 4 short-axis nodes ->
``K = 64``.

The shared module is left unchanged, so existing results do not move.  This
script instead resolves the tie upward (``order = floor(snap(raw)) + 2``,
identical to ``ceil(raw) + 1`` for every non-integer ``raw``), which is the side
the correlated prior's dictionaries sit on, and the gate fails unless it
reproduces them (``mixture_20``, ``mixture_21``) exactly.

    python scripts/anisotropy_compiler.py --gate [--self-test]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import benchmark, compile_atoms  # noqa: E402
from scripts import anisotropy_config as G  # noqa: E402

SNAP_EPS = 64  # machine epsilons, relative


def snap(raw: float) -> float:
    nearest = round(raw)
    if abs(raw - nearest) <= SNAP_EPS * np.finfo(float).eps * max(1.0, abs(raw)):
        return float(nearest)
    return float(raw)


def node_order(raw: float, rule: str, max_nodes: int) -> int:
    """``tie_up``: floor(snap(raw)) + 2; ``tie_down``: ceil(snap(raw)) + 1 (self-test only)."""
    r = snap(raw)
    n = math.floor(r) + 2 if rule == "tie_up" else math.ceil(r) + 1
    return int(min(max_nodes, max(1, n)))


def compile_dictionary(prior, *, spread_factor: float, c_ratio: float = compile_atoms.C_RATIO,
                       max_nodes_per_axis: int = 31, rule: str = "tie_up"):
    """``compile_atoms.compile_dictionary`` with only the order rule replaced.

    Every other operation is copied in the same order so that, whenever the two
    rules agree on the orders, the output is identical (checked by
    :func:`gate`).  Returns ``(AtomDictionary, record)``.
    """
    lam_min = float(np.linalg.eigvalsh(prior.cov).min())
    lam = c_ratio * lam_min
    cov_c = prior.cov - lam * np.eye(prior.dim)
    ev, axes = np.linalg.eigh(cov_c)
    if ev.min() < -1e-12:
        raise ValueError(f"Sigma_c is not PSD (min eig {ev.min():.3g})")
    sd = np.sqrt(np.maximum(ev, 0.0))
    raws = [spread_factor * s / math.sqrt(lam) for s in sd]
    orders = [node_order(r, rule, max_nodes_per_axis) for r in raws]

    nodes, weights = [], []
    for i in range(prior.dim):
        xi, wi = np.polynomial.hermite_e.hermegauss(orders[i])
        nodes.append(xi * sd[i])
        weights.append(wi / wi.sum())
    grids = np.meshgrid(*nodes, indexing="ij")
    z = np.stack([g.reshape(-1) for g in grids], axis=-1)
    wt = np.ones(z.shape[0])
    idx = np.indices(orders)
    for i in range(prior.dim):
        wt = wt * weights[i][idx[i].reshape(-1)]
    centers = prior.mu + z @ axes.T
    log_b = np.log(wt / wt.sum())
    d = compile_atoms.AtomDictionary(centers=centers, log_b=log_b, lam=lam,
                                     order=int(max(orders)), c_ratio=c_ratio, prior=prior)
    rec = {"rule": rule, "spread_factor": spread_factor, "c_ratio": c_ratio,
           "lambda": lam, "sqrt_lambda": math.sqrt(lam),
           "sigma_c_axis_sd": sd.tolist(), "sigma_c_axes": axes.tolist(),
           "raw_criterion": [repr(r) for r in raws], "node_orders": orders,
           "num_atoms": int(centers.shape[0])}
    return d, rec


def correlated_prior_check(rule: str) -> list[str]:
    """Reasons the rule fails to reproduce the correlated prior's K=27 / K=64 JSONs."""
    box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    prior = compile_atoms.design_prior(box)
    bad = []
    for name, pid in (("small", 20), ("medium", 21)):
        d, _ = compile_dictionary(prior, spread_factor=compile_atoms.CAPACITIES[name], rule=rule)
        want = json.loads(benchmark.prior_path(G.TASK, "mixture", pid).read_text())
        got = json.loads(json.dumps(d.as_prior_json(G.TASK)))
        for key in ("mu", "sigma", "pi", "compilation"):
            if got[key] != want[key]:
                bad.append(f"{rule}: mixture_{pid} ({name}) field {key!r} differs "
                           f"(K {len(got['mu'])} vs {len(want['mu'])})")
    return bad


def fidelity_check(num: int = 400, seed: int = 12345) -> dict:
    """Where the rules agree on orders, the copy must match the shared compiler exactly."""
    box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    rng = np.random.default_rng(seed)
    agree = mismatch = differ = 0
    for i in range(num):
        sd1, sd2 = np.exp(rng.uniform(np.log(0.03), np.log(0.4), 2))
        th = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        prior = compile_atoms.CorrelatedGaussianPrior(
            mu=rng.uniform(-0.5, 0.5, 2), cov=rot @ np.diag([sd1**2, sd2**2]) @ rot.T,
            box=box, rule=f"fidelity{i}")
        for sf in compile_atoms.CAPACITIES.values():
            old = compile_atoms.compile_dictionary(prior, spread_factor=sf)
            new, _ = compile_dictionary(prior, spread_factor=sf)
            if old.num_atoms != new.num_atoms:
                differ += 1
                continue
            agree += 1
            if not (np.array_equal(old.centers, new.centers)
                    and np.array_equal(old.log_b, new.log_b) and old.lam == new.lam):
                mismatch += 1
    return {"cases": 2 * num, "orders_agree": agree, "orders_differ": differ,
            "bitwise_mismatch_where_orders_agree": mismatch}


def gate(self_test: bool) -> dict:
    res = {"correlated_prior_tie_up": correlated_prior_check("tie_up"),
           "fidelity": fidelity_check()}
    failures = list(res["correlated_prior_tie_up"])
    if res["fidelity"]["bitwise_mismatch_where_orders_agree"]:
        failures.append("copied compiler differs from compile_atoms where orders agree")
    if self_test:
        down = correlated_prior_check("tie_down")
        res["self_test_tie_down_rejected"] = bool(down)
        res["self_test_tie_down_reasons"] = down
        if not down:
            failures.append("self-test: tie_down rule was NOT rejected")
    res["failures"] = failures
    res["pass"] = not failures
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", default=None, help="write the gate record here")
    args = p.parse_args()
    if not args.gate:
        p.error("nothing to do; use --gate")
    res = gate(args.self_test)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    if not res["pass"]:
        print("ANISOTROPY COMPILER GATE FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
