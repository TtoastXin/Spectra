"""Tests of the Langevin corrector on an analytic Gaussian oracle.

Trajectories are not compared with the upstream PriorGuide sampler: its RNG key
splitting differs from the pre-drawn noise arrays used here, so equal
trajectories are not expected.  The tests fall into three groups:

1. Driver equivalence.  The instrumented driver gives the same samples as the
   uninstrumented one (a reference copy is kept below) at ``L = 0`` and
   ``L > 0``, and adding NFE counters to a hook does not change any sample.
2. Update formula.  For fixed ``(z, t, eps)`` the corrector performs
   ``z <- z + delta s + sqrt(2 delta + 1e-8) eps`` with ``delta = eta g(t)^2 dt
   / 2``, the update in the released PriorGuide code, and ``g^2 = 2 sigma
   sigma'`` is checked by autodiff.  PriorGuide's Eq. (A1) gives a step half
   this size.
3. Gaussian analytic checks with a non-trivial factor (``kappa > 0``,
   ``a != 0``): the transported Spectra score equals the analytic target score,
   the transported score inside the corrector reproduces the closed-form
   one-step Langevin moments, the full sampler with ``L = 0`` and ``L = 8``
   reaches the target moments, PG-FullCov with the exact Jacobian is exact, and
   the runtime NFE counters equal ``(N - 1)(1 + L)``.

The corrector is a Langevin kernel on the current target noisy state ``z_t`` at
base time ``t``, so its step size and noise use ``g(t)``.  Spectra computes the
target score at ``z_t`` by querying the base score at the transformed point
``(m, rho)``; this changes how the score is computed, not the kernel, so
``g(rho)`` is not substituted into ``delta``.  The one-step-moment test checks
that the transported score, used with the base-time step size, is the correct
Langevin drift for the target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectra import e2e, hooks
from spectra.oracles import (
    GaussianOracleBackbone,
    single_gaussian,
    tilt_mixture,
)
from spectra.ratios import GaussianComponents
from spectra.simformer import VESchedule

D = 3
SCH = VESchedule()
A_TILT = np.array([0.8, -0.5, 0.3])
KAPPA = 2.0


def _base_mixture():
    lower = np.array([[0.6, 0.1, 0.0], [0.0, 0.5, 0.15], [0.1, 0.0, 0.7]])
    cov = lower @ lower.T + 0.15 * np.eye(D)
    mean = np.array([0.4, -0.3, 0.2])
    return single_gaussian(jnp.asarray(mean), jnp.asarray(cov))


@pytest.fixture(scope="module")
def ctx():
    base = _base_mixture()
    target = tilt_mixture(base, jnp.asarray(A_TILT), KAPPA * jnp.eye(D))
    oracle = GaussianOracleBackbone(base, SCH)
    comps = GaussianComponents(
        log_weights=jnp.zeros(1),
        means=jnp.asarray(A_TILT / KAPPA)[None, :],
        covs=(jnp.eye(D) / KAPPA)[None, :, :])
    return {"base": base, "target": target, "oracle": oracle, "comps": comps,
            "x_o": jnp.zeros(2)}


def _target_hook(target):
    score = jax.vmap(target.score, in_axes=(0, None))

    def hook(z, t):
        return score(z, SCH.sigma(t) ** 2), {}

    return hook


def _inputs(cfg, num, key=0, dim=D, mean=None, std=None):
    k_init, k_noise = jax.random.split(jax.random.PRNGKey(key))
    mean = jnp.zeros(dim) if mean is None else jnp.asarray(mean)
    std = jnp.full((dim,), float(SCH.sigma(SCH.t_max))) if std is None else jnp.asarray(std)
    x_init = e2e.draw_terminal(k_init, num, mean, std)
    diff, lang = e2e.make_noise(k_noise, cfg.num_steps, num, dim, cfg.langevin_steps)
    return x_init, diff, lang


def _run(hook, cfg, num, key=0, snapshots=(), **kw):
    x_init, diff, lang = _inputs(cfg, num, key, **kw)
    return e2e.reverse_sample(hook, schedule=SCH, cfg=cfg, x_init=x_init,
                              diffusion_noise=diff, langevin_noise=lang,
                              snapshot_fracs=snapshots)


# ---------------------------------------------------------------- layer 1 --


def _plain_reverse_sample(hook, *, schedule, cfg, x_init, diffusion_noise,
                           langevin_noise, snapshot_fracs=e2e.DEFAULT_SNAPSHOT_FRACS):
    """Reference copy of the uninstrumented driver; returns the samples only."""
    ts = e2e.time_grid(schedule, cfg)
    log_ratio = schedule.log_ratio
    c = jnp.sqrt(2.0 * log_ratio)
    n = cfg.num_steps

    def step(z, payload):
        t_hi, t_lo, eps, lang_eps = payload
        dt = t_hi - t_lo
        sig = schedule.sigma(t_hi)
        diff2 = sig**2 * (2.0 * log_ratio)
        if cfg.langevin_steps > 0:
            lstep = cfg.langevin_ratio * diff2 * dt / 2.0

            def lang(zz, e):
                s, _ = hook(zz, t_hi)
                return zz + lstep * s + jnp.sqrt(2.0 * lstep + 1e-8) * e, None

            z, _ = jax.lax.scan(lang, z, lang_eps)
        score, aux = hook(z, t_hi)
        z = z + dt * diff2 * score + jnp.sqrt(dt) * sig * c * eps
        return z, aux

    t_hi = ts[::-1][:-1]
    t_lo = ts[::-1][1:]
    cuts = e2e.snapshot_indices(n, snapshot_fracs)
    bounds = [0, *cuts, n - 1]
    z = x_init
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        payload = (t_hi[lo:hi], t_lo[lo:hi], diffusion_noise[lo:hi],
                   langevin_noise[lo:hi])
        z, _ = jax.lax.scan(step, z, payload)
    return np.asarray(jax.device_get(z))


@pytest.mark.parametrize("langevin", (0, 3))
@pytest.mark.parametrize("snapshots", ((), e2e.DEFAULT_SNAPSHOT_FRACS))
def test_instrumented_driver_matches_plain_driver(ctx, langevin, snapshots):
    oracle, x_o = ctx["oracle"], ctx["x_o"]
    cfg = e2e.SamplerConfig(num_steps=24, langevin_steps=langevin)
    hook, ops, _ = hooks.make_tq_hook(oracle, x_o, A_TILT, KAPPA)
    x_init, diff, lang = _inputs(cfg, 32)
    kw = dict(schedule=SCH, cfg=cfg, x_init=x_init, diffusion_noise=diff,
              langevin_noise=lang, snapshot_fracs=snapshots)
    plain = _plain_reverse_sample(hook, **kw)
    new = e2e.reverse_sample(hook, **kw).samples
    instrumented = e2e.reverse_sample(e2e.instrument_hook(hook, ops), **kw)
    assert np.array_equal(plain, new)
    assert np.array_equal(new, instrumented.samples)


@pytest.mark.parametrize("langevin", (0, 1, 3, 8))
def test_runtime_nfe_counters_equal_the_static_accounting(ctx, langevin):
    oracle, x_o, comps = ctx["oracle"], ctx["x_o"], ctx["comps"]
    cfg = e2e.SamplerConfig(num_steps=13, langevin_steps=langevin)
    built = {
        "base": hooks.make_base_hook(oracle, x_o),
        "spectra": hooks.make_tq_hook(oracle, x_o, A_TILT, KAPPA),
        "pg": hooks.make_pg_hook(oracle, x_o, comps,
                                 reverse_cov=hooks.upstream_reverse_cov(1.0, D)),
        "a_full": hooks.make_a_full_hook(oracle, x_o, comps),
    }
    for name, (hook, ops, _) in built.items():
        res = _run(e2e.instrument_hook(hook, ops), cfg, 8)
        got = e2e.nfe_from_aux(res.step_aux)
        want = e2e.nfe_accounting(cfg, ops)
        assert got == want, (name, got, want)
        assert got["hook_calls"] == (cfg.num_steps - 1) * (1 + langevin)


def test_uninstrumented_hook_leaves_no_counters(ctx):
    hook, _, _ = hooks.make_base_hook(ctx["oracle"], ctx["x_o"])
    res = _run(hook, e2e.SamplerConfig(num_steps=6, langevin_steps=2), 4)
    assert e2e.nfe_from_aux(res.step_aux) == {}
    assert not any(k.startswith(e2e.NFE_PREFIX) for k in res.step_aux)


# ---------------------------------------------------------------- layer 2 --


@pytest.mark.parametrize("t", (0.05, 0.3, 0.7, 1.0))
def test_step_size_is_eta_g2_dt_over_2_and_g2_is_2_sigma_sigmadot(t):
    """``delta = eta g^2 dt / 2`` (code) vs ``eta sigma sigma' dt / 2`` (PriorGuide Eq. A1)."""
    dt, eta = 0.01, 0.5
    delta = float(e2e.langevin_step_size(SCH, t, dt, eta))
    sigma = float(SCH.sigma(t))
    sigma_dot = float(jax.grad(lambda tt: SCH.sigma(tt))(t))
    g2_autodiff = 2.0 * sigma * sigma_dot
    g2_code = sigma**2 * 2.0 * SCH.log_ratio
    assert g2_code == pytest.approx(g2_autodiff, rel=1e-4)
    assert delta == pytest.approx(eta * g2_autodiff * dt / 2.0, rel=1e-4)
    paper_a1 = eta * sigma_dot * sigma * dt / 2.0
    assert delta / paper_a1 == pytest.approx(2.0, rel=1e-4)
    # eta = 1: Langevin noise variance 2 delta equals the EM step variance g^2 dt
    assert 2.0 * e2e.langevin_step_size(SCH, t, dt, 1.0) == pytest.approx(
        g2_code * dt, rel=1e-5)


def test_corrector_update_matches_the_recorded_upstream_formula():
    """Replay the driver in float64 numpy from the same noise arrays."""
    cfg = e2e.SamplerConfig(num_steps=12, langevin_steps=2, langevin_ratio=0.5)
    # scaled so that dt g^2 |F| << 1 on every step: the replay must compare a
    # stable trajectory, not the rounding of an exploding one
    field = 1e-4 * np.array([[0.7, -0.2, 0.1], [0.0, 0.5, 0.3], [0.2, 0.1, 0.9]])

    def hook(z, t):
        return -(z @ jnp.asarray(field, dtype=z.dtype).T), {}

    x_init, diff, lang = _inputs(cfg, 5, key=3)
    got = e2e.reverse_sample(hook, schedule=SCH, cfg=cfg, x_init=x_init,
                             diffusion_noise=diff, langevin_noise=lang,
                             snapshot_fracs=()).samples

    ts = np.asarray(e2e.time_grid(SCH, cfg), float)
    t_hi, t_lo = ts[::-1][:-1], ts[::-1][1:]
    z = np.asarray(x_init, float)
    diff64, lang64 = np.asarray(diff, float), np.asarray(lang, float)
    lr = SCH.log_ratio
    for i in range(cfg.num_steps - 1):
        dt = t_hi[i] - t_lo[i]
        sig = SCH.sigma_min * (SCH.sigma_max / SCH.sigma_min) ** t_hi[i]
        g2 = sig**2 * 2.0 * lr
        delta = cfg.langevin_ratio * g2 * dt / 2.0
        for j in range(cfg.langevin_steps):
            s = -(z @ field.T)
            z = z + delta * s + np.sqrt(2.0 * delta + 1e-8) * lang64[i, j]
        s = -(z @ field.T)
        z = z + dt * g2 * s + np.sqrt(dt) * sig * np.sqrt(2.0 * lr) * diff64[i]
    np.testing.assert_allclose(got, z, rtol=1e-5, atol=1e-4 * float(np.abs(z).max()))


# ---------------------------------------------------------------- layer 3 --


@pytest.mark.parametrize("t", (0.02, 0.2, 0.5, 0.9))
def test_transported_spectra_score_equals_the_analytic_target_score(ctx, t):
    oracle, target, x_o = ctx["oracle"], ctx["target"], ctx["x_o"]
    hook, _, _ = hooks.make_tq_hook(oracle, x_o, A_TILT, KAPPA)
    tau = float(SCH.sigma(t) ** 2)
    z = target.sample_tau(jax.random.PRNGKey(int(t * 1000)), tau, 256)
    got = np.asarray(hook(z, t)[0])
    want = np.asarray(_target_hook(target)(z, t)[0])
    scale = max(1.0, float(np.abs(want).max()))
    assert np.max(np.abs(got - want)) < 2e-4 * scale


def test_fullcov_with_the_exact_jacobian_is_exact_and_pg_is_finite(ctx):
    """Gaussian base + Gaussian factor: ``s + (m_q - m)/tau`` with ``C_T = tau J``
    is the exact target score (``J`` symmetric), so FullCov must match it
    wherever float32 can resolve ``(m_q - m)/tau``; the next test covers the
    noise levels where it cannot."""
    oracle, target, x_o, comps = ctx["oracle"], ctx["target"], ctx["x_o"], ctx["comps"]
    a_full, _, _ = hooks.make_a_full_hook(oracle, x_o, comps)
    pg, _, _ = hooks.make_pg_hook(oracle, x_o, comps,
                                  reverse_cov=hooks.upstream_reverse_cov(0.6, D))
    for t in (0.5, 0.7, 0.9):
        tau = float(SCH.sigma(t) ** 2)
        z = target.sample_tau(jax.random.PRNGKey(11), tau, 256)
        want = np.asarray(_target_hook(target)(z, t)[0])
        got = np.asarray(a_full(z, t)[0])
        scale = max(1.0, float(np.abs(want).max()))
        assert np.max(np.abs(got - want)) < 1e-3 * scale, t
        assert np.isfinite(np.asarray(pg(z, t)[0])).all()


def test_fullcov_guidance_is_a_one_over_tau_difference_in_float32(ctx):
    """At ``t = 0.1`` (``tau ~ 1e-7``) FullCov's ``(m_q - m)/tau`` divides an O(1)
    float32 cancellation by ``tau`` and its guidance error is O(1); the
    transported Spectra query ``(a - kappa z)/D + s_p(m, rho)/D`` has no such
    division and stays exact.  This is a numerical property of the two
    formulations and says nothing about learned-model accuracy: at these noise
    levels the reverse step ``dt g^2`` is tiny and the samples barely move."""
    oracle, target, x_o, comps = ctx["oracle"], ctx["target"], ctx["x_o"], ctx["comps"]
    a_full, _, _ = hooks.make_a_full_hook(oracle, x_o, comps)
    tq, _, _ = hooks.make_tq_hook(oracle, x_o, A_TILT, KAPPA)
    t = 0.1
    tau = float(SCH.sigma(t) ** 2)
    assert tau < 1e-6
    z = target.sample_tau(jax.random.PRNGKey(12), tau, 256)
    want = np.asarray(_target_hook(target)(z, t)[0])
    err_full = float(np.max(np.abs(np.asarray(a_full(z, t)[0]) - want)))
    err_tq = float(np.max(np.abs(np.asarray(tq(z, t)[0]) - want)))
    assert err_tq < 1e-3
    assert err_full > 10.0 * err_tq


def test_one_langevin_step_with_the_transported_score_has_the_closed_form_moments(ctx):
    """ULA kernel on the noisy target: mean stays, cov -> K C K^T + 2 delta I,
    K = I - delta C^{-1}; the departure from C is delta^2 C^{-1}, so the noise
    level and step are chosen (t = 0.9, dt = 0.1) where delta is comparable to
    C and the check discriminates.  ``s_q`` inside the corrector comes from the
    transported query (m, rho); the step size and noise come from the base time
    t.  Two wrong step sizes are ruled out: a factor-two variant
    (delta -> 2 delta) and a corrector run with g(rho) instead of g(t)
    (delta -> delta / D).
    """
    oracle, target, x_o = ctx["oracle"], ctx["target"], ctx["x_o"]
    hook, _, _ = hooks.make_tq_hook(oracle, x_o, A_TILT, KAPPA)
    t, dt, eta, n = 0.9, 0.1, 0.5, 400_000
    tau = float(SCH.sigma(t) ** 2)
    delta = float(e2e.langevin_step_size(SCH, t, dt, eta))
    k_z, k_e = jax.random.split(jax.random.PRNGKey(5))
    z = target.sample_tau(k_z, tau, n)
    eps = jax.random.normal(k_e, z.shape)
    z_new = np.asarray(z + delta * hook(z, t)[0] + jnp.sqrt(2.0 * delta + 1e-8) * eps, float)

    m = np.asarray(target.means[0], float)
    cov_tau = np.asarray(target.covs[0], float) + tau * np.eye(D)

    def ula_cov(d):
        k_mat = np.eye(D) - d * np.linalg.inv(cov_tau)
        return k_mat @ cov_tau @ k_mat.T + 2.0 * d * np.eye(D)

    cov_want = ula_cov(delta)
    cov_got = np.cov(z_new.T)
    np.testing.assert_allclose(z_new.mean(0), m, atol=5.0 * np.sqrt(cov_want.max() / n))
    rel = lambda c: np.linalg.norm(cov_got - c) / np.linalg.norm(c)
    assert rel(cov_want) < 0.02
    # the check is discriminating: delta^2 C^{-1} is a visible fraction of C
    assert np.linalg.norm(cov_want - cov_tau) / np.linalg.norm(cov_tau) > 0.10
    assert rel(ula_cov(2.0 * delta)) > 0.10          # step size off by a factor of two
    d_big = 1.0 + KAPPA * tau
    assert rel(ula_cov(delta / d_big)) > 0.10        # g(rho) instead of g(t)


@pytest.mark.parametrize("langevin", (0, 8))
def test_full_sampler_with_transported_score_lands_on_the_target(ctx, langevin):
    oracle, target, x_o = ctx["oracle"], ctx["target"], ctx["x_o"]
    cfg = e2e.SamplerConfig(num_steps=100, langevin_steps=langevin)
    hook, ops, _ = hooks.make_tq_hook(oracle, x_o, A_TILT, KAPPA)
    mean0, std0 = oracle.terminal_moments()
    res = _run(e2e.instrument_hook(hook, ops), cfg, 6000, key=1, mean=mean0, std=std0)
    m = np.asarray(target.means[0], float)
    cov = np.asarray(target.covs[0], float)
    got_m, got_c = res.samples.mean(0), np.cov(res.samples.T)
    sd = np.sqrt(np.diag(cov))
    assert np.max(np.abs(got_m - m) / sd) < 0.08
    assert np.linalg.norm(got_c - cov) / np.linalg.norm(cov) < 0.10
    assert e2e.nfe_from_aux(res.step_aux)["hook_calls"] == 99 * (1 + langevin)


@pytest.mark.parametrize("langevin", (0, 4))
def test_transported_and_analytic_target_hooks_give_the_same_trajectories(ctx, langevin):
    oracle, target, x_o = ctx["oracle"], ctx["target"], ctx["x_o"]
    cfg = e2e.SamplerConfig(num_steps=40, langevin_steps=langevin)
    tq, _, _ = hooks.make_tq_hook(oracle, x_o, A_TILT, KAPPA)
    a = _run(tq, cfg, 512, key=2).samples
    b = _run(_target_hook(target), cfg, 512, key=2).samples
    np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)
