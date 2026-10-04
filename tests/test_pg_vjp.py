"""PriorGuide's guidance via one VJP instead of a full Jacobian.

The full-Jacobian hook (``make_pg_hook``) forms ``J_m`` with ``jacrev``
(``theta_dim`` backward passes) and then contracts it with ``u``.
``make_pg_vjp_hook`` returns the same ``J_m^T u`` from one pullback.  These tests
compare it with the full-Jacobian hook on a small random Simformer, and check
that the tolerance used here rejects the plausible but wrong derivative
``grad_z [u(m)^T m]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectra import hooks, methods
from spectra.backbone import Backbone, CheckpointInfo
from spectra.ratios import GaussianComponents
from spectra.simformer import SimformerConfig, VESchedule, build_model, upstream_nn

THETA_DIM, X_DIM, BATCH = 3, 2, 6
TIMES = (0.05, 0.4, 0.9)
RTOL, ATOL = 1e-5, 1e-6  # equivalence tolerance for the VJP hook


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
        "comps": comps,
        "cov": hooks.upstream_reverse_cov(0.9, THETA_DIM),
        "z": jnp.asarray(rng.normal(size=(BATCH, THETA_DIM))),
        "x_o": jnp.asarray(rng.normal(size=(X_DIM,))),
    }


@pytest.mark.parametrize("clip_x0", [None, hooks.UPSTREAM_X0_CLIP])
@pytest.mark.parametrize("t", TIMES)
def test_vjp_hook_matches_the_full_jacobian_hook(ctx, t, clip_x0):
    bb, comps, cov, z, x_o = ctx["backbone"], ctx["comps"], ctx["cov"], ctx["z"], ctx["x_o"]
    full_jac, _, _ = hooks.make_pg_hook(bb, x_o, comps, reverse_cov=cov, clip_x0=clip_x0)
    vjp, ops, meta = hooks.make_pg_vjp_hook(bb, x_o, comps, reverse_cov=cov, clip_x0=clip_x0)
    want, aux_want = full_jac(z, t)
    got, aux_got = vjp(z, t)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=RTOL, atol=ATOL)
    for k in ("guidance_norm", "score_norm", "clip_hit_frac"):
        np.testing.assert_allclose(float(aux_got[k]), float(aux_want[k]), rtol=RTOL, atol=ATOL)
    assert "jac_asym" not in aux_got
    assert ops == {"base_score": 1, "score_vjp": 1}
    assert meta["method"] == "pg_vjp"


@pytest.mark.parametrize("t", TIMES)
def test_vjp_guidance_is_the_per_row_jacobian_transpose_times_u(ctx, t):
    """The batched pullback equals ``J_m(z_i)^T u_i`` row by row."""
    bb, comps, cov, z, x_o = ctx["backbone"], ctx["comps"], ctx["cov"], ctx["z"], ctx["x_o"]
    tau = bb.schedule.sigma(t) ** 2
    s, pullback = jax.vjp(lambda zz: bb.score_theta_t(zz, t, x_o), z)
    m = z + tau * s
    u = jnp.stack([methods.gaussian_product_moments(m[i], cov(tau), comps)[1]
                   for i in range(BATCH)])
    got = u + tau * pullback(u)[0]
    jac = bb.denoiser_jacobian_t(z, t, x_o)
    want = jnp.stack([jac[i].T @ u[i] for i in range(BATCH)])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("t", TIMES)
def test_the_tolerance_rejects_differentiating_through_u(ctx, t):
    """Negative control: ``grad_z [u(m(z))^T m(z)]`` adds ``J^T (du/dm)^T m``.

    It is an easy mistake to make here, and the equivalence tolerance must be
    tight enough to tell it apart from ``J^T u``.
    """
    bb, comps, cov, z, x_o = ctx["backbone"], ctx["comps"], ctx["cov"], ctx["z"], ctx["x_o"]
    tau = bb.schedule.sigma(t) ** 2

    def u_of(m):
        return jax.vmap(lambda mi: methods.gaussian_product_moments(mi, cov(tau), comps)[1])(m)

    def objective(zz):
        m = zz + tau * bb.score_theta_t(zz, t, x_o)
        return jnp.sum(u_of(m) * m)

    wrong = jax.grad(objective)(z)
    s = bb.score_theta_t(z, t, x_o)
    m = z + tau * s
    jac = bb.denoiser_jacobian_t(z, t, x_o)
    right = jax.vmap(lambda j, ui: j.T @ ui)(jac, u_of(m))
    assert not np.allclose(np.asarray(wrong), np.asarray(right), rtol=RTOL, atol=ATOL)
