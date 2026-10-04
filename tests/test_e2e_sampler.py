"""Driver-level tests: the wrappers must not move the sampler.

A randomly initialised Simformer is enough here: these check the plumbing
(time grid, reversed clock, RNG pairing, degeneracy at ``r == 1``), not the
quality of any score.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectra import e2e, hooks
from spectra.backbone import Backbone, CheckpointInfo
from spectra.simformer import (
    SimformerConfig,
    VESchedule,
    build_model,
    upstream_nn,
)

THETA_DIM, X_DIM = 3, 4


@pytest.fixture(scope="module")
def backbone():
    nn = upstream_nn()
    sch = VESchedule()
    cfg = SimformerConfig(token_dim=8, time_embedding_dim=8, num_layers=1, num_heads=2)
    num_nodes = THETA_DIM + X_DIM
    init_fn, _ = build_model(num_nodes, sch, cfg, nn)
    params = init_fn(
        jax.random.PRNGKey(0),
        jnp.ones((2,)),
        jnp.zeros((2, num_nodes, 1)),
        jnp.arange(num_nodes),
        jnp.zeros((2, num_nodes, 1)),
        meta_data=None,
    )
    ckpt = CheckpointInfo(
        params=params, theta_dim=THETA_DIM, x_dim=X_DIM, schedule=sch, model_cfg=cfg,
        marginal_end_mean=np.zeros(num_nodes), marginal_end_std=np.full(num_nodes, 15.0),
        source="spectra", path="<test>", meta={},
    )
    return Backbone(ckpt, mode="conditional", nn=nn)


def _run(backbone, hook, cfg, key=0):
    mean, std = backbone.terminal_moments()
    k_init, k_noise = jax.random.split(jax.random.PRNGKey(key))
    x_init = e2e.draw_terminal(k_init, 16, mean, std)
    diff, lang = e2e.make_noise(k_noise, cfg.num_steps, 16, THETA_DIM,
                                cfg.langevin_steps)
    return e2e.reverse_sample(
        hook, schedule=backbone.schedule, cfg=cfg, x_init=x_init,
        diffusion_noise=diff, langevin_noise=lang,
    )


@pytest.mark.parametrize("langevin", (0, 3))
def test_trivial_ratio_transport_reproduces_the_base_sampler(backbone, langevin):
    """``a = 0, kappa = 0`` is ``r == 1``: TQ must reproduce the base sampler exactly."""
    x_o = jnp.arange(X_DIM, dtype=jnp.float32) * 0.1
    cfg = e2e.SamplerConfig(num_steps=12, langevin_steps=langevin)
    base_hook, _, _ = hooks.make_base_hook(backbone, x_o)
    tq_hook, _, _ = hooks.make_tq_hook(backbone, x_o, jnp.zeros(THETA_DIM), 0.0)
    a = _run(backbone, base_hook, cfg)
    b = _run(backbone, tq_hook, cfg)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_paired_rng_makes_two_methods_share_their_terminal_draw(backbone):
    x_o = jnp.zeros(X_DIM)
    cfg = e2e.SamplerConfig(num_steps=8)
    mean, std = backbone.terminal_moments()
    k1, _ = jax.random.split(jax.random.PRNGKey(0))
    k2, _ = jax.random.split(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(
        np.asarray(e2e.draw_terminal(k1, 16, mean, std)),
        np.asarray(e2e.draw_terminal(k2, 16, mean, std)),
    )


def test_power_grid_preserves_the_trained_time_range(backbone):
    cfg = e2e.SamplerConfig(num_steps=50, grid="power", grid_rho=2.0)
    ts = np.asarray(e2e.time_grid(backbone.schedule, cfg))
    assert np.isclose(ts[0], backbone.schedule.t_min)
    assert np.isclose(ts[-1], backbone.schedule.t_max)
    assert np.all(np.diff(ts) > 0)
    # steps are concentrated at low noise
    assert np.diff(ts)[0] < np.diff(ts)[-1]


def test_upstream_grid_moves_the_lower_endpoint():
    """Upstream's literal power grid moves ``t_min``, so the main protocol uses ``power``."""
    sch = VESchedule()
    cfg = e2e.SamplerConfig(num_steps=25, grid="upstream_power", grid_rho=2.0)
    ts = np.asarray(e2e.time_grid(sch, cfg))
    assert ts[0] < sch.t_min
    assert np.isclose(ts[0], sch.t_min**2)


def test_snapshots_land_on_the_requested_remaining_time(backbone):
    x_o = jnp.zeros(X_DIM)
    cfg = e2e.SamplerConfig(num_steps=41)
    hook, _, _ = hooks.make_base_hook(backbone, x_o)
    res = _run(backbone, hook, cfg)
    assert sorted(res.snapshots) == [0.05, 0.2, 0.5, 0.8]
    assert res.step_aux["score_norm"].shape == (cfg.num_steps - 1,)
    ts = res.times
    for frac, snap in res.snapshots.items():
        assert ts[0] <= snap["t"] <= ts[-1]
    # later snapshots are at lower diffusion time
    got = [res.snapshots[f]["t"] for f in (0.8, 0.5, 0.2, 0.05)]
    assert got == sorted(got, reverse=True)


def test_nfe_accounting_separates_operators():
    cfg = e2e.SamplerConfig(num_steps=100, langevin_steps=8)
    acc = e2e.nfe_accounting(cfg, {"base_score": 1, "denoiser_jacobian": 1})
    assert acc["hook_calls"] == 99 * 9
    assert acc["base_score_calls"] == 99 * 9
    assert acc["denoiser_jacobian_calls"] == 99 * 9


def test_transport_query_time_shift_matches_the_noise_round_trip(backbone):
    """``t_rho = t - log(D) / (2 log R)`` is the exact inverse of ``sigma``.

    The hook uses the shift rather than ``t_of_tau(rho)`` so that ``kappa = 0``
    leaves the diffusion time exactly unchanged; both must still agree numerically.
    """
    sch = backbone.schedule
    for t in (1e-5, 0.1, 0.5, 0.9, 1.0):
        for kappa in (0.0, 0.5, 240.0):
            tau = float(sch.sigma(t) ** 2)
            D = 1.0 + kappa * tau
            shift = t - 0.5 * np.log(D) / sch.log_ratio
            roundtrip = float(np.log(np.sqrt(tau / D) / sch.sigma_min) / sch.log_ratio)
            assert abs(shift - roundtrip) < 1e-6 * max(abs(t), 1e-3)
        assert float(sch.t_of_tau(sch.sigma(t) ** 2)) == pytest.approx(t, abs=1e-6)
