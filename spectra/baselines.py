"""Base-posterior banks and the SIR baseline.

``draw_base_bank`` and ``base_bank`` draw clean samples from the unadapted
frozen model with the base hook.  They supply the SIR bank and the bank from
which PriorGuide's scalar reverse covariance takes its variance (5,000 samples
on a 500-point uniform grid).  The two differ in argument order and in the
dtype they return: ``draw_base_bank``, used by the single-factor runners,
returns the sampler's float32 array, while ``base_bank``, used by the
two-component runner, returns float64.  PriorGuide's variance is computed from
the returned array, so both are kept to reproduce each runner exactly.

SIR resamples a base bank with self-normalised weights ``w = r(theta)``
(:func:`snis_weights`).  Bank points outside a uniform training box receive
zero weight and are counted.  :func:`sir_resample` draws with replacement.
"""

from __future__ import annotations

import time

import jax
import numpy as np

from spectra import e2e, hooks


def draw_base_bank(backbone, key, num: int, x_o, cfg: e2e.SamplerConfig):
    """A plain base-posterior draw under a stated protocol, with its own timing."""
    mean, std = backbone.terminal_moments()
    k_init, k_noise = jax.random.split(key)
    x_init = e2e.draw_terminal(k_init, num, mean, std)
    diff, lang = e2e.make_noise(k_noise, cfg.num_steps, num, backbone.theta_dim,
                                cfg.langevin_steps)
    hook, _, _ = hooks.make_base_hook(backbone, x_o)
    t0 = time.perf_counter()
    res = e2e.reverse_sample(hook, schedule=backbone.schedule, cfg=cfg,
                             x_init=x_init, diffusion_noise=diff,
                             langevin_noise=lang, snapshot_fracs=())
    return np.asarray(res.samples), time.perf_counter() - t0


def base_bank(backbone, x_o, key, num, cfg):
    mean, std = backbone.terminal_moments()
    k_init, k_noise = jax.random.split(key)
    x_init = e2e.draw_terminal(k_init, num, mean, std)
    diff, lang = e2e.make_noise(k_noise, cfg.num_steps, num, backbone.theta_dim, 0)
    hook, _, _ = hooks.make_base_hook(backbone, x_o)
    t0 = time.perf_counter()
    res = e2e.reverse_sample(hook, schedule=backbone.schedule, cfg=cfg, x_init=x_init,
                             diffusion_noise=diff, langevin_noise=lang, snapshot_fracs=())
    return np.asarray(res.samples, float), time.perf_counter() - t0


def snis_weights(shift, bank: np.ndarray, box) -> dict:
    """Self-normalised importance weights for the prior change, ``w = r(theta)``.

    ``log_r`` returns the analytic continuation outside a uniform training box,
    where the training prior is zero, so a bank point outside the box would get
    a finite weight in a region where the target has no mass either.  Those
    points are given zero weight and counted; the count is part of the
    diagnostic.
    """
    log_w = np.asarray(shift.log_r(bank), float)
    outside = np.zeros(bank.shape[0], dtype=bool)
    if box is not None:
        low, high = box
        outside = ~np.all((bank >= low) & (bank <= high), axis=1)
        log_w = np.where(outside, -np.inf, log_w)
    m = np.max(log_w)
    if not np.isfinite(m):
        return {"log_w": log_w, "w": np.zeros_like(log_w), "ess": 0.0,
                "degenerate": True, "outside_box": int(outside.sum())}
    w = np.exp(log_w - m)
    s1, s2 = w.sum(), (w**2).sum()
    ess = float(s1**2 / s2) if s2 > 0 else 0.0
    wn = w / s1
    order = np.sort(wn)[::-1]
    return {
        "log_w": log_w,
        "w": wn,
        "ess": ess,
        "ess_frac": ess / bank.shape[0],
        "max_weight": float(order[0]),
        "top10_weight_mass": float(order[: max(1, bank.shape[0] // 100)].sum()),
        "log_w_range": float(np.ptp(log_w[np.isfinite(log_w)])),
        "outside_box": int(outside.sum()),
        "degenerate": False,
    }


def sir_resample(rng, bank: np.ndarray, w: np.ndarray, num: int) -> np.ndarray:
    """Sampling-importance-resampling with replacement.

    This is standard SIR; weight degeneracy shows up as duplicated draws.
    """
    idx = rng.choice(bank.shape[0], size=num, replace=True, p=w)
    return bank[idx], idx
