"""Component-weight estimation for multi-component Spectra (Algorithm 1, line 3).

For a positive factor representation ``r ∝ sum_k b_k phi_k`` the posterior
component weights are ``alpha_k ∝ b_k Z_k`` with ``Z_k = E_{p(theta | x)}[phi_k]``.
Two estimators of ``log Z_k`` are provided:

* Direct: the average of ``phi_k`` over banks of base-posterior samples
  (:func:`direct_evidence`);
* path-space: importance sampling on finite reverse chains
  (:func:`spectra.pathspace.pathspace_evidence`).

Each estimator is repeated over estimator seeds and the log-normalisers are
averaged in the log domain before the softmax.  Two implementations of that
aggregation are used for the reported results:

``direct_evidence`` / ``pathspace_evidence`` (``K = 2``)
    the systematic two-component benchmark.  The per-seed log-ratio
    ``Delta = log Z_1 - log Z_2`` is averaged over the seeds whose estimate is
    finite, and the weights are ``softmax_pair(log b_1 - log b_2 + Delta)``.
``pathspace_weights`` (any ``K``)
    the component-count and correlated-Gaussian studies.  Each ``log Z_k`` is
    averaged over seeds with ``nanmean`` and the weights are
    ``softmax(log b + mean log Z)``.

Averaging is linear, so the two rules give the same weights at ``K = 2`` when
every seed is finite.  They differ only in the last floating-point digits of
the softmax, and in what a non-finite estimate removes: the whole seed in the
first rule, that component's value for that seed in the second.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import jax
import numpy as np

from spectra import atoms as atoms_mod
from spectra import e2e, hooks, pathspace

LOG_ZERO = -700.0


def softmax_pair(log_odds: float) -> tuple[np.ndarray, np.ndarray]:
    lp = np.array([log_odds, 0.0])
    if not np.isfinite(log_odds):
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
    lp = lp - (np.max(lp) + math.log(np.sum(np.exp(lp - np.max(lp)))))
    return lp, np.exp(lp)


def direct_evidence(backbone, x_o, atoms, seeds, n, steps):
    """Plug-in (``plugin_base``) evidence on one base-sample bank per seed."""
    cfg = e2e.SamplerConfig(num_steps=steps, grid="uniform", langevin_steps=0)
    log_b_diff = float(atoms.log_b[0] - atoms.log_b[1])
    per, banks, wall = [], [], 0.0
    for s in seeds:
        key0 = jax.random.PRNGKey(20260826 + s)
        key0, key = jax.random.split(key0)
        key, k_init, k_noise = jax.random.split(key, 3)
        mean, std = backbone.terminal_moments()
        x_init = e2e.draw_terminal(k_init, n, mean, std)
        diff, lang = e2e.make_noise(k_noise, steps, n, backbone.theta_dim, 0)
        hook, _, _ = hooks.make_base_hook(backbone, x_o)
        t0 = time.perf_counter()
        res = e2e.reverse_sample(hook, schedule=backbone.schedule, cfg=cfg, x_init=x_init,
                                 diffusion_noise=diff, langevin_noise=lang, snapshot_fracs=())
        wall += time.perf_counter() - t0
        bank = np.asarray(res.samples, float)
        finite = np.isfinite(bank).all(axis=1)
        plug = atoms_mod.evidence_plugin(bank[finite], atoms) if finite.any() else None
        delta = float(plug["log_z"][0] - plug["log_z"][1]) if plug is not None else float("nan")
        per.append({"seed": s, "n": int(n), "n_finite": int(finite.sum()),
                    "log_z": (plug["log_z"].tolist() if plug is not None else None),
                    "ess": (plug["ess"].tolist() if plug is not None else None),
                    "rse": (plug["rse"].tolist() if plug is not None else None),
                    "delta": delta, "log_odds": log_b_diff + delta})
        banks.append(bank)
    deltas = [p["delta"] for p in per if math.isfinite(p["delta"])]
    delta = float(np.mean(deltas)) if deltas else float("nan")
    log_pi, pi = softmax_pair(log_b_diff + delta)
    pooled = np.concatenate(banks)
    pooled = pooled[np.isfinite(pooled).all(axis=1)]
    plug_pooled = atoms_mod.evidence_plugin(pooled, atoms) if pooled.shape[0] else None
    return {
        "source": "round6_direct", "estimator": "plugin_base, Delta averaged over bank seeds",
        "budget": {"seeds": list(seeds), "n_per_seed": int(n), "steps": int(steps),
                   "grid": "uniform"},
        "per_seed": per, "delta_estimate": delta, "delta_estimate_n_seeds": len(deltas),
        "delta_estimate_per_seed": [p["delta"] for p in per],
        "log_b_diff": log_b_diff, "log_odds": log_b_diff + delta,
        "log_pi": [float(v) for v in np.where(np.isfinite(log_pi), log_pi, LOG_ZERO)],
        "pi": [float(v) for v in pi],
        "pooled": ({"n": int(pooled.shape[0]), "log_z": plug_pooled["log_z"].tolist(),
                    "ess": plug_pooled["ess"].tolist(),
                    "delta": float(plug_pooled["log_z"][0] - plug_pooled["log_z"][1])}
                   if plug_pooled is not None else None),
        "ess_min_over_seeds": (min(min(p["ess"]) for p in per if p["ess"]) if any(p["ess"] for p in per)
                               else float("nan")),
        "status": "ok" if math.isfinite(delta) else "nonfinite",
        "wall_clock_s": wall,
    }, pooled


def pathspace_evidence(backbone, x_o, atoms, seeds, trajectories, steps, grid):
    score_p, make_q = pathspace.learned_scores(backbone, x_o, atoms)
    cfg = e2e.SamplerConfig(num_steps=steps, grid=grid, langevin_steps=0)
    term_mean, term_std = backbone.terminal_moments()
    log_b_diff = float(atoms.log_b[0] - atoms.log_b[1])
    per, wall = [], 0.0
    for seed in seeds:
        res = {}
        atom_rows = []
        for k in range(atoms.num_atoms):
            r = pathspace.pathspace_evidence(
                score_p, make_q, schedule=backbone.schedule, cfg=cfg, atoms=atoms, atom=k,
                terminal_mean=term_mean, terminal_std=term_std,
                num_trajectories=trajectories, seed=1000 * seed + 17 * k)
            res[k] = r
            wall += r.wall_clock_s
            wd, idg = r.diagnostics["weight"], r.diagnostics["integrand"]
            atom_rows.append({"atom": k, "log_z": r.log_z,
                              "w_ess_normalized": wd["ess_normalized"],
                              "w_log_w_var": wd.get("log_w_var"),
                              "u_ess_normalized": idg["ess_normalized"],
                              "u_max_normalized_weight": idg.get("max_normalized_weight"),
                              "frac_finite_samples": r.diagnostics["frac_finite_samples"],
                              "wall_clock_s": r.wall_clock_s})
        comb = pathspace.combine_log_odds(res, atoms)
        per.append({"seed": seed, "delta": comb["delta"], "log_odds": comb["log_odds"],
                    "ess_norm_min": min(comb["ess_norm_i"], comb["ess_norm_j"]),
                    "atoms": atom_rows,
                    "status": "ok" if np.isfinite(comb["delta"]) else "nonfinite"})
    deltas = [p["delta"] for p in per if math.isfinite(p["delta"])]
    delta = float(np.mean(deltas)) if deltas else float("nan")
    log_pi, pi = softmax_pair(log_b_diff + delta)
    return {
        "source": "round6_pathspace", "estimator": "path-space IS, Delta averaged over estimator seeds",
        "budget": {"seeds": list(seeds), "trajectories": int(trajectories), "steps": int(steps),
                   "grid": grid, "atoms": int(atoms.num_atoms)},
        "per_seed": per, "delta_estimate": delta, "delta_estimate_n_seeds": len(deltas),
        "delta_estimate_per_seed": [p["delta"] for p in per],
        "log_b_diff": log_b_diff, "log_odds": log_b_diff + delta,
        "log_pi": [float(v) for v in np.where(np.isfinite(log_pi), log_pi, LOG_ZERO)],
        "pi": [float(v) for v in pi],
        "ess_norm_min_over_seeds": (min(p["ess_norm_min"] for p in per) if per else float("nan")),
        "status": "ok" if math.isfinite(delta) else "nonfinite",
        "wall_clock_s": wall,
    }


def weight_error(pi_hat, pi_star):
    if pi_hat is None or pi_star is None:
        return None
    ph, ps = np.asarray(pi_hat, float), np.asarray(pi_star, float)
    if not (np.isfinite(ph).all() and np.isfinite(ps).all()):
        return {"l1": float("nan"), "abs_alpha0": float("nan"), "log_odds_err": float("nan")}
    lo_h = float(np.log(max(ph[0], 1e-300)) - np.log(max(ph[1], 1e-300)))
    lo_s = float(np.log(max(ps[0], 1e-300)) - np.log(max(ps[1], 1e-300)))
    return {"l1": float(np.abs(ph - ps).sum()), "abs_alpha0": float(abs(ph[0] - ps[0])),
            "log_odds_err": lo_h - lo_s,
            "note": "raw; log-odds of near-degenerate references are huge by construction"}


def softmax_log(log_x: np.ndarray):
    lx = np.asarray(log_x, float)
    m = np.max(lx[np.isfinite(lx)]) if np.isfinite(lx).any() else 0.0
    w = np.exp(lx - m)
    tot = w.sum()
    pi = w / tot
    return np.log(pi), pi


# Normalisation tolerance for a stored ``log_pi``.  Every shipped weight record
# is normalised to within 2.2e-16, so this is seven orders of magnitude of
# headroom; a uniform ``LOG_ZERO`` placeholder is off by ~699.
LOG_PI_NORM_TOL = 1e-9


def weight_record_problems(blob, *, expected_k=None):
    """Everything that makes a stored component-weight record unusable.

    Returns a list of human-readable problems; an empty list means the record
    may be consumed.  The checks, and why each one is here:

    * ``status``: the estimators write ``"ok"`` or a failure reason.  A record
      that carries the field must say ``ok``.  Reference-weight records do not
      carry it at all and are accepted without one.
    * ``NaN`` / ``+inf``: not a valid log-probability.
    * all components at ``-inf``: no posterior mass anywhere.  A single
      component at ``-inf`` is fine: a mixture component may legitimately carry
      no mass, and that is not an estimation failure.
    * normalisation: ``logsumexp(log_pi)`` must be 0.  This also catches a
      failed estimate whose entries were replaced by a finite placeholder,
      because equal placeholders do not sum to one.
    """
    problems = []
    log_pi = blob.get("log_pi")
    if log_pi is None:
        return ["record carries no log_pi"]
    status = blob.get("status")
    if status is not None and status != "ok":
        problems.append(f"estimate status={status!r}")
    try:
        a = np.asarray(log_pi, float)
    except (TypeError, ValueError) as exc:
        return problems + [f"log_pi is not numeric: {exc}"]
    # Shape before anything that indexes it: a scalar would raise IndexError and
    # a nested list would pass a length check on its outer dimension.
    if a.ndim != 1:
        return problems + [f"log_pi must be a 1-D vector, got shape {a.shape}"]
    if a.size == 0:
        return problems + ["log_pi is empty"]
    if expected_k is not None and a.shape[0] != expected_k:
        problems.append(f"{a.shape[0]} weights, expected {expected_k}")
    pi = blob.get("pi")
    if pi is not None:
        try:
            q = np.asarray(pi, float)
        except (TypeError, ValueError):
            q = None
        if q is None or q.shape != a.shape:
            problems.append("pi does not have the same shape as log_pi")
        elif np.isfinite(a).all() and not np.allclose(q, np.exp(a), rtol=1e-6, atol=1e-9):
            problems.append("pi is not exp(log_pi)")
    if np.isnan(a).any():
        problems.append("log_pi contains NaN")
    if np.isposinf(a).any():
        problems.append("log_pi contains +inf")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        problems.append("log_pi has no component with finite mass")
    elif not problems:
        m = float(finite.max())
        total = m + math.log(float(np.exp(a - m).sum()))
        if abs(total) > LOG_PI_NORM_TOL:
            problems.append(f"log_pi is not normalised (logsumexp={total:+.6g})")
    return problems


def load_weight_record(path, *, expected_k=None):
    """Read a weight JSON and refuse it unless it is usable.  Returns log_pi."""
    blob = json.loads(Path(path).read_text())
    problems = weight_record_problems(blob, expected_k=expected_k)
    if problems:
        raise SystemExit(f"unusable component weights {path}: " + "; ".join(problems))
    return np.asarray(blob["log_pi"], float), blob


def pathspace_weights(backbone, x_o, atoms, seeds, trajectories, steps, grid):
    """``log pi`` from per-atom path-space evidences, general in ``K``."""
    score_p, make_q = pathspace.learned_scores(backbone, x_o, atoms)
    cfg = e2e.SamplerConfig(num_steps=steps, grid=grid, langevin_steps=0)
    term_mean, term_std = backbone.terminal_moments()
    per, wall = [], 0.0
    log_z = np.full((len(seeds), atoms.num_atoms), np.nan)
    for si, seed in enumerate(seeds):
        atom_rows = []
        for k in range(atoms.num_atoms):
            r = pathspace.pathspace_evidence(
                score_p, make_q, schedule=backbone.schedule, cfg=cfg, atoms=atoms,
                atom=k, terminal_mean=term_mean, terminal_std=term_std,
                num_trajectories=trajectories, seed=1000 * seed + 17 * k)
            log_z[si, k] = r.log_z
            wall += r.wall_clock_s
            wd, idg = r.diagnostics["weight"], r.diagnostics["integrand"]
            atom_rows.append({"atom": k, "log_z": r.log_z,
                              "w_ess_normalized": wd["ess_normalized"],
                              "u_ess_normalized": idg["ess_normalized"],
                              "u_max_normalized_weight": idg.get("max_normalized_weight"),
                              "frac_finite_samples": r.diagnostics["frac_finite_samples"],
                              "wall_clock_s": r.wall_clock_s})
            print(f"    [seed {seed} atom {k}] log_z={r.log_z:+.4f} "
                  f"u_ess={idg['ess_normalized']:.4g} {r.wall_clock_s:.2f}s", flush=True)
        per.append({"seed": seed, "atoms": atom_rows})
    mean_log_z = np.nanmean(log_z, axis=0)
    log_pi, pi = softmax_log(np.asarray(atoms.log_b, float) + mean_log_z)
    ok = bool(np.isfinite(mean_log_z).all())
    return {
        "source": "round14_pathspace",
        "estimator": ("path-space IS per atom, log Zhat averaged over estimator "
                      "seeds, weights = softmax(log b + mean log Zhat); at K=2 this "
                      "is identical to the Round 10 Delta-averaging rule"),
        "budget": {"seeds": list(seeds), "trajectories": int(trajectories),
                   "steps": int(steps), "grid": grid, "atoms": int(atoms.num_atoms)},
        "per_seed": per,
        "log_z_per_seed": log_z.tolist(),
        "mean_log_z": mean_log_z.tolist(),
        "log_b": np.asarray(atoms.log_b, float).tolist(),
        "log_pi": [float(v) for v in np.where(np.isfinite(log_pi), log_pi, LOG_ZERO)],
        "pi": [float(v) for v in pi],
        "status": "ok" if ok else "nonfinite",
        "wall_clock_s": wall,
        "wall_clock_per_atom_seed_s": wall / max(1, len(seeds) * atoms.num_atoms),
    }
