"""The batched hooks must agree with the single-point reference estimators.

:mod:`spectra.hooks` rewrites PriorGuide and PG-FullCov to act on a whole batch
of sampler states at once.  These tests pin each one to the single-point
implementation in :mod:`spectra.methods`, so a batching mistake cannot change a
benchmark number unnoticed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectra import benchmark, hooks, methods
from spectra.backbone import Backbone, CheckpointInfo
from spectra.learned_eval import PSD_EPS, psd_project
from spectra.ratios import GaussianComponents
from spectra.simformer import (
    SimformerConfig,
    VESchedule,
    build_model,
    upstream_nn,
)

THETA_DIM, X_DIM, BATCH = 3, 2, 6
T = 0.4


@pytest.fixture(scope="module")
def ctx():
    nn = upstream_nn()
    sch = VESchedule()
    cfg = SimformerConfig(token_dim=8, time_embedding_dim=8, num_layers=1, num_heads=2)
    num_nodes = THETA_DIM + X_DIM
    init_fn, _ = build_model(num_nodes, sch, cfg, nn)
    params = init_fn(
        jax.random.PRNGKey(3), jnp.ones((2,)), jnp.zeros((2, num_nodes, 1)),
        jnp.arange(num_nodes), jnp.zeros((2, num_nodes, 1)), meta_data=None,
    )
    ckpt = CheckpointInfo(
        params=params, theta_dim=THETA_DIM, x_dim=X_DIM, schedule=sch, model_cfg=cfg,
        marginal_end_mean=np.zeros(num_nodes), marginal_end_std=np.full(num_nodes, 15.0),
        source="spectra", path="<test>", meta={},
    )
    rng = np.random.default_rng(0)
    comps = GaussianComponents(
        log_weights=jnp.asarray(np.log([0.3, 0.7])),
        means=jnp.asarray(rng.normal(size=(2, THETA_DIM))),
        covs=jnp.asarray(np.stack([np.eye(THETA_DIM) * 0.2, np.eye(THETA_DIM) * 0.5])),
    )
    return {
        "backbone": Backbone(ckpt, mode="conditional", nn=nn),
        "joint": Backbone(ckpt, mode="joint_inpaint", nn=nn),
        "comps": comps,
        "z": jnp.asarray(rng.normal(size=(BATCH, THETA_DIM))),
        "x_o": jnp.asarray(rng.normal(size=(X_DIM,))),
    }


def test_pg_hook_matches_single_point_closure(ctx):
    bb, z, x_o, comps = ctx["backbone"], ctx["z"], ctx["x_o"], ctx["comps"]
    tau = float(bb.schedule.sigma(T) ** 2)
    cov = methods.pg_reverse_cov(tau, THETA_DIM, jnp.eye(THETA_DIM) * 0.9)
    hook, ops, _ = hooks.make_pg_hook(bb, x_o, comps, reverse_cov=cov, clip_x0=None)
    got, aux = hook(z, T)

    s = bb.score_theta_t(z, T, x_o)
    jac = bb.denoiser_jacobian_t(z, T, x_o)
    m = z + tau * s
    want = jnp.stack([
        s[i] + methods.pg_closure(m[i], jac[i], tau, comps, jnp.eye(THETA_DIM) * 0.9)
        for i in range(BATCH)
    ])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)
    assert ops == {"base_score": 1, "denoiser_jacobian": 1}
    assert float(aux["clip_hit_frac"]) == 0.0


def test_pg_hook_clip_reproduces_upstream_x0_clipping(ctx):
    """Upstream clips the guided ``x0`` mean to [-50, 50]; the hook keeps this clipping."""
    bb, z, x_o, comps = ctx["backbone"], ctx["z"], ctx["x_o"], ctx["comps"]
    tau = float(bb.schedule.sigma(T) ** 2)
    cov = methods.pg_reverse_cov(tau, THETA_DIM)
    unclipped, _ = hooks.make_pg_hook(bb, x_o, comps, reverse_cov=cov, clip_x0=None)[0](z, T)
    tiny, aux = hooks.make_pg_hook(bb, x_o, comps, reverse_cov=cov, clip_x0=1e-3)[0](z, T)
    assert float(aux["clip_hit_frac"]) > 0.0
    assert not np.allclose(np.asarray(unclipped), np.asarray(tiny))


def test_a_full_hook_matches_single_point_twisted_mean(ctx):
    bb, z, x_o, comps = ctx["backbone"], ctx["z"], ctx["x_o"], ctx["comps"]
    tau = float(bb.schedule.sigma(T) ** 2)
    hook, ops, _ = hooks.make_a_full_hook(bb, x_o, comps)
    got, _ = hook(z, T)

    s = bb.score_theta_t(z, T, x_o)
    jac = bb.denoiser_jacobian_t(z, T, x_o)
    m = z + tau * s
    _, j_psd, _, _ = psd_project(jac, PSD_EPS)
    want = jnp.stack([
        s[i] + methods.a_full(m[i], tau * j_psd[i], tau, comps) for i in range(BATCH)
    ])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)
    assert ops["eigh"] == 1


def test_conditioning_modes_are_different_functions(ctx):
    """The conditional and joint-inpainting score queries are different functions.

    This is why the benchmark uses the conditional query rather than upstream
    PriorGuide's joint-score query.
    """
    bb, joint, z, x_o = ctx["backbone"], ctx["joint"], ctx["z"], ctx["x_o"]
    a = np.asarray(bb.score_theta_t(z, T, x_o))
    b = np.asarray(joint.score_theta_t(z, T, x_o))
    assert not np.allclose(a, b)
