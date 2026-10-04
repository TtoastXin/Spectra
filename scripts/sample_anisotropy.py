#!/usr/bin/env python
"""Sample the anisotropic cells with PriorGuide and Spectra at (25, 8).

A thin runner over the existing library, following ``sample_k_scaling.py``'s
conventions so the rows are comparable with the other studies: terminal draw
and noise keys ``PRNGKey(seed * 1_000_003 + obs_seed)``, atom labels
``PRNGKey(7 + seed)``, PriorGuide's scalar ``Sigma_post`` from a 5000 x 500
uniform base draw, the standard output layout.  Per ``(cell, obs, seed)``:

``pg_exact``     PriorGuide's closure on the exact anisotropic ratio, passed as
                 one full-covariance ``GaussianComponents`` built from
                 ``cells.json`` (the prior-JSON schema and ``build_prior_shift``
                 cannot express it);
``spectra_ref``  exact per-atom transport on the compiled dictionary
                 ``mixture_<cell>`` with the grid reference component weights;
``spectra_ps``   the same with the path-space weights.

Manifests carry ``prior_type = "aniso"``, ``prior_id = cell`` so
``compute_metrics.py`` scores every row against the exact target reference
``references/two_moons/aniso_<cell>``.  Cell 60 (gamma = 1, isotropic) is
accepted for ``pg_exact`` only, as a parity check against
``sample_single_factor.py`` on ``mild_20``.

    python scripts/sample_anisotropy.py --cells 62,63,64,65 --model-id 0 --obs-seeds 1000000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import atoms as atoms_mod  # noqa: E402
from spectra import benchmark, e2e, hooks, weights as weights_mod  # noqa: E402
from spectra.backbone import Backbone, load_checkpoint  # noqa: E402
from spectra.baselines import draw_base_bank  # noqa: E402
from spectra.prior_shift import build_prior_shift  # noqa: E402
from spectra.ratios import GaussianComponents  # noqa: E402
from spectra.simformer import UPSTREAM_COMMIT, upstream_nn  # noqa: E402
from spectra.utils import file_hash, git_state, peak_gpu_memory_mb  # noqa: E402
from scripts import anisotropy_config as G  # noqa: E402
from scripts.anisotropy_weights import reference_weights_path, weights_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
METHODS = ("pg_exact", "spectra_ref", "spectra_ps")


def exact_components(cell: dict) -> GaussianComponents:
    """The exact anisotropic ratio (uniform training prior: r = the target Gaussian)."""
    return GaussianComponents(
        log_weights=jnp.asarray(np.zeros(1)),
        means=jnp.asarray(np.asarray(cell["mu"], float)[None, :]),
        covs=jnp.asarray(np.asarray(cell["cov"], float)[None, :, :]))


def load_weights(root: Path, kind: str, cell: int, obs: int, model: int, k: int):
    path = (reference_weights_path(root, cell, obs) if kind == "spectra_ref"
            else weights_path(root, cell, obs, model))
    if not path.is_file():
        raise SystemExit(f"missing {kind} weights: {path}")
    # A failed estimate must not reach the sampler: its logits can be finite and
    # equal, which would turn into a uniform component assignment without an error.
    log_pi, _ = weights_mod.load_weight_record(path, expected_k=k)
    pi = np.exp(log_pi)
    return log_pi, {"weight_source": "reference" if kind == "spectra_ref" else "pathspace",
                    "weight_file": str(path), "weight_file_sha256_16": file_hash(path),
                    "pi": pi.tolist(), "k_eff": G.k_eff(pi)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", default=",".join(str(c) for c in G.ANISO_CELLS))
    p.add_argument("--obs-seeds", default=",".join(str(o) for o in G.OBS_SEEDS))
    p.add_argument("--model-id", type=int, default=0)
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint of --model-id (default: the shipped one)")
    p.add_argument("--sampler-seeds", default=",".join(str(s) for s in G.SAMPLER_SEEDS))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--num-samples", type=int, default=G.NUM_SAMPLES)
    p.add_argument("--pg-var-samples", type=int, default=5000)
    p.add_argument("--pg-var-steps", type=int, default=500)
    p.add_argument("--out", default=G.ROOT)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    root = Path(args.out)
    cells_rec = G.load_cells()
    cells = [int(v) for v in args.cells.split(",") if v.strip()]
    obs_seeds = [int(v) for v in args.obs_seeds.split(",") if v.strip()]
    seeds = [int(v) for v in args.sampler_seeds.split(",") if v.strip()]
    want = [m for m in args.methods.split(",") if m]
    if set(want) - set(METHODS):
        p.error(f"unknown methods {sorted(set(want) - set(METHODS))}")
    for c in cells:
        if cells_rec[c]["role"] != "anisotropic" and set(want) != {"pg_exact"}:
            p.error(f"cell {c} is not anisotropic; only pg_exact (parity) is allowed")

    ckpt_path = args.checkpoint or G.checkpoint(args.model_id)
    task = benchmark.get_task(G.TASK)
    ckpt = load_checkpoint(ckpt_path)
    backbone = Backbone(ckpt, mode="conditional", nn=upstream_nn())
    mean, std = backbone.terminal_moments()
    cfg = e2e.SamplerConfig(num_steps=G.CONFIG[0], grid=G.GRID, langevin_steps=G.CONFIG[1])
    tag = G.tag()

    common = {
        "round": "round16_anisotropy", "task": G.TASK,
        "theta_dim": task.theta_dim, "x_dim": task.x_dim,
        "model_id": args.model_id, "checkpoint": str(Path(ckpt_path).resolve()),
        "checkpoint_sha256_16": file_hash(ckpt_path),
        "prior_type": "aniso", "conditioning": "conditional",
        "num_samples": args.num_samples,
        "langevin_semantics": ("delta = eta g(t)^2 dt / 2 (released PriorGuide "
                               "corrector); see e2e.langevin_step_size"),
        "upstream_commit": UPSTREAM_COMMIT, "repo_git": git_state(REPO),
        "source_sha256_16": benchmark.source_hashes(
            [Path(__file__), REPO / "scripts" / "anisotropy_config.py",
             *sorted((REPO / "spectra").glob("*.py"))]),
        "cells_json_sha256_16": file_hash(G.CELLS_JSON),
        "jax": jax.__version__, "python": sys.version.split()[0],
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
        "schedule": {"sigma_min": ckpt.schedule.sigma_min,
                     "sigma_max": ckpt.schedule.sigma_max,
                     "t_min": ckpt.schedule.t_min, "t_max": ckpt.schedule.t_max},
    }

    compiled = {}
    for c in cells:
        if cells_rec[c]["role"] == "anisotropic":
            shift = build_prior_shift(G.TASK, "mixture", c)
            atoms = atoms_mod.atoms_from_shift(shift)
            if atoms.num_atoms != cells_rec[c]["compiled"]["num_atoms"]:
                raise SystemExit(f"mixture_{c}: {atoms.num_atoms} atoms, cells.json says "
                                 f"{cells_rec[c]['compiled']['num_atoms']}")
            compiled[c] = atoms

    for obs_seed in obs_seeds:
        theta_true, x_o = benchmark.load_observation(
            G.TASK, G.SOURCE_PRIOR_TYPE, G.SOURCE_PRIOR_ID, obs_seed)
        pg = {}
        if "pg_exact" in want:
            t0 = time.perf_counter()
            pg_bank, pg_bank_wall = draw_base_bank(
                backbone, jax.random.PRNGKey((90000 + obs_seed) % (2 ** 31)),
                args.pg_var_samples, x_o,
                e2e.SamplerConfig(num_steps=args.pg_var_steps, grid="uniform",
                                  langevin_steps=0))
            total_variance = float(np.var(pg_bank))
            pg = {"pg_total_variance": total_variance, "precompute_s": pg_bank_wall}
            print(f"[obs {obs_seed}] PG Sigma_post bank in {time.perf_counter() - t0:.1f}s "
                  f"(var={total_variance:.5g})", flush=True)
        for c in cells:
            built, meta, weights = {}, {}, {}
            if "pg_exact" in want:
                built["pg_exact"] = hooks.make_pg_hook(
                    backbone, x_o, exact_components(cells_rec[c]),
                    reverse_cov=hooks.upstream_reverse_cov(pg["pg_total_variance"],
                                                           task.theta_dim))
                meta["pg_exact"] = {**pg, "ratio": "exact anisotropic Gaussian, full covariance"}
            for kind in ("spectra_ref", "spectra_ps"):
                if kind in want:
                    weights[kind] = load_weights(root, kind, c, obs_seed, args.model_id,
                                                 compiled[c].num_atoms)
            for seed in seeds:
                k_init, k_noise = jax.random.split(
                    jax.random.PRNGKey((seed * 1_000_003 + obs_seed) % (2 ** 31)))
                x_init = e2e.draw_terminal(k_init, args.num_samples, mean, std)
                diff, lang = e2e.make_noise(k_noise, cfg.num_steps, args.num_samples,
                                            task.theta_dim, cfg.langevin_steps)
                rows, labels = dict(built), {}
                for kind, (log_pi, _) in weights.items():
                    labels[kind] = hooks.draw_atom_labels(
                        jax.random.PRNGKey(7 + seed), jnp.asarray(log_pi), args.num_samples)
                    rows[kind] = hooks.make_spectra_hook(backbone, x_o, compiled[c], labels[kind])
                for name, (hook, ops, hook_meta) in rows.items():
                    row_name = f"{name}_ms{seed}"
                    out = (root / "samples" / G.TASK / row_name / f"model_{args.model_id}"
                           / f"aniso_{c}" / f"obs_{obs_seed}")
                    if args.skip_existing and (out / f"{tag}.json").is_file():
                        continue
                    t1 = time.perf_counter()
                    res = e2e.reverse_sample(
                        e2e.instrument_hook(hook, ops), schedule=backbone.schedule,
                        cfg=cfg, x_init=x_init, diffusion_noise=diff,
                        langevin_noise=lang, snapshot_fracs=())
                    wall = time.perf_counter() - t1
                    nfe_static = e2e.nfe_accounting(cfg, ops)
                    nfe_runtime = e2e.nfe_from_aux(res.step_aux)
                    out.mkdir(parents=True, exist_ok=True)
                    payload = {"samples": res.samples, "x_o": np.asarray(x_o),
                               "theta_true": np.asarray(theta_true), "times": res.times}
                    if name in labels:
                        payload["atom_labels"] = np.asarray(labels[name])
                    np.savez_compressed(out / f"{tag}.npz", **payload)
                    man = {**common, "cell": c, "gamma": cells_rec[c]["gamma"],
                           "orientation": cells_rec[c]["orientation"],
                           "K": compiled[c].num_atoms if c in compiled else 1,
                           "prior_id": c, "obs_seed": obs_seed,
                           "method": row_name, "method_family": name,
                           "method_meta": hook_meta, "sampler_seed": seed,
                           "sampler": cfg.as_dict(), "nfe": nfe_static,
                           "nfe_runtime": nfe_runtime,
                           "nfe_runtime_equals_static": nfe_runtime == nfe_static,
                           "wall_clock_s": wall, "wall_clock_includes_compile": True,
                           "wall_clock_per_1000_samples_s": wall * 1000.0 / args.num_samples,
                           "nan_or_inf": int((~np.isfinite(res.samples)).sum()),
                           "peak_gpu_memory_mb": peak_gpu_memory_mb(),
                           **(weights[name][1] if name in weights else {}),
                           **meta.get(name, {})}
                    (out / f"{tag}.json").write_text(json.dumps(man, indent=2, default=str))
                    print(f"[cell {c} obs {obs_seed}][{row_name}] {wall:.1f}s "
                          f"nfe={nfe_runtime.get('hook_calls')} nan={man['nan_or_inf']}",
                          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
