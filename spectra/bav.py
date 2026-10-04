"""BCI / BAV: coordinates, responses and a batched likelihood.

The task upstream calls ``bav`` is the paper's Bayesian Causal Inference model:
five parameters, 98 responses on a fixed 7x7 audiovisual stimulus grid, and a
likelihood that integrates out two-dimensional sensory noise by Gauss--Hermite
quadrature.

Conventions:

1. Coordinates.  The shipped prior JSONs and the Simformer use model
   coordinates ``y``, where the training prior is ``N(0, I_5)``.  The
   likelihood takes physical (unconstrained) coordinates
   ``theta = y * rescale + shift``, whose entries are
   ``log sigma_V, log sigma_A, log sigma_s, log sigma_m, logit p_same``.
2. Response scaling.  The observation JSONs store responses standardised by
   1M-simulation constants; the likelihood takes raw responses.
3. Quadrature rule.  Upstream builds nodes with
   ``np.polynomial.hermite.hermgauss``, which forms weights as ``1/(f' f')`` and
   overflows: at order 401 it returns 135 NaNs, and the usable orders below that
   do not converge monotonically.  ``scipy.special.roots_hermite`` is stable at
   every order tested (weights sum to sqrt(pi) to 1e-16 through n=1001), so the
   batched path uses it and the order can be chosen freely.

The batched likelihood below vectorises upstream's
``nll_bav_constant_gaussian`` over a leading theta axis, with the same algebra,
the same lapse floor and the same ``+1e-12`` guard.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .coordinates import get_coordinates

# Upstream's fixed constants (priorg/sim/tasks/bav.py), copied rather than
# imported so this module does not import torch at load time.
RHO_A = 4.0 / 3.0
STIM_VALUES = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
LAPSE_FIXED = 0.02  # upstream hard-codes this; theta[4] is p_same, not lapse
MU_P_FIXED = 0.0
NUM_TRIALS = 98
THETA_DIM = 5
RESPONSE_FLOOR = 1e-12  # upstream's guard inside log(prob + eps)
LAPSE_SPREAD = 90.0  # upstream's uniform lapse density, lapse / 90


def upstream_norm_constants(upstream: Optional[str] = None):
    """``(mean, std)`` used to standardise the 98 responses, from the vendored upstream task file."""
    from .simformer import _upstream_root

    root = _upstream_root(upstream)
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sim.tasks.bav import BAV_X_MEAN, BAV_X_STD

    return np.asarray(BAV_X_MEAN, float), np.asarray(BAV_X_STD, float)


# ------------------------------------------------------------- coordinates ---


def model_to_physical(y: np.ndarray) -> np.ndarray:
    """Model coordinates (training prior ``N(0, I)``) to the likelihood's."""
    c = get_coordinates("bav")
    return np.atleast_2d(np.asarray(y, float)) * c.rescale + c.shift


def physical_to_model(theta: np.ndarray) -> np.ndarray:
    c = get_coordinates("bav")
    return (np.atleast_2d(np.asarray(theta, float)) - c.shift) / c.rescale


def unstandardise_responses(x_std: np.ndarray,
                            upstream: Optional[str] = None) -> np.ndarray:
    """Observation-JSON responses back to the raw scale the likelihood wants."""
    mean, std = upstream_norm_constants(upstream)
    return np.asarray(x_std, float) * std + mean


def standardise_responses(x_raw: np.ndarray,
                          upstream: Optional[str] = None) -> np.ndarray:
    mean, std = upstream_norm_constants(upstream)
    return (np.asarray(x_raw, float) - mean) / std


# --------------------------------------------------------------- quadrature --


def gauss_hermite(n: int):
    """Physicists' Gauss--Hermite nodes/weights for ``int e^{-x^2} f(x) dx``.

    ``scipy.special.roots_hermite`` rather than numpy's ``hermgauss``: numpy
    forms the weights as a reciprocal square and overflows to NaN by order 401,
    which is inside the range a convergence study needs.
    """
    from scipy.special import roots_hermite

    y, w = roots_hermite(n)
    return np.asarray(y, float), np.asarray(w, float)


def stimulus_grid():
    """``(S_V, S_A, response_type)`` for the fixed 49 BV + 49 BA trials."""
    stim = np.asarray(STIM_VALUES, float)
    sv, sa = np.meshgrid(stim, stim, indexing="ij")
    sv, sa = sv.reshape(-1), sa.reshape(-1)
    S_V = np.concatenate([sv, sv])
    S_A = np.concatenate([sa, sa])
    rt = np.concatenate([np.zeros(49, int), np.ones(49, int)])
    return S_V, S_A, rt


# --------------------------------------------------------- batched likelihood -


@dataclass(frozen=True)
class LikelihoodConfig:
    gh_deg: int = 101
    theta_chunk: int = 64
    dtype: str = "float64"
    device: str = "cpu"


def log_likelihood(theta_phys: np.ndarray, responses_raw: np.ndarray, *,
                   cfg: LikelihoodConfig = LikelihoodConfig()) -> np.ndarray:
    """``log p(R | theta)`` for a batch of physical-coordinate parameters.

    Returns one scalar per row of ``theta_phys``; the sign is the log
    likelihood, i.e. minus upstream's summed NLL.

    Vectorised over theta by adding a leading axis to upstream's per-trial
    ``(B, N_V, N_A)`` computation and chunking it, because the full tensor is
    ``(T, 98, n, n)``.  Only the theta axis is split; the quadrature sum and the
    trial product are computed whole.
    """
    import torch

    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    theta = np.atleast_2d(np.asarray(theta_phys, float))
    if theta.shape[-1] != THETA_DIM:
        raise ValueError(f"BAV theta is {THETA_DIM}-dimensional, got {theta.shape}")
    R = torch.as_tensor(np.asarray(responses_raw, float).reshape(-1),
                        dtype=dtype, device=device)
    if R.numel() != NUM_TRIALS:
        raise ValueError(f"expected {NUM_TRIALS} responses, got {R.numel()}")

    S_V_np, S_A_np, rt_np = stimulus_grid()
    S_V = torch.as_tensor(S_V_np, dtype=dtype, device=device)
    S_A = torch.as_tensor(S_A_np, dtype=dtype, device=device)
    rt = torch.as_tensor(rt_np, dtype=torch.long, device=device)

    y_np, w_np = gauss_hermite(cfg.gh_deg)
    nodes = torch.as_tensor(y_np, dtype=dtype, device=device)
    weights = torch.as_tensor(w_np, dtype=dtype, device=device)
    weight_mat = (weights[:, None] * weights[None, :]) / math.pi  # (n, n)

    lapse = LAPSE_FIXED
    mu_p = MU_P_FIXED
    out = np.empty(theta.shape[0], float)

    for start in range(0, theta.shape[0], cfg.theta_chunk):
        blk = torch.as_tensor(theta[start:start + cfg.theta_chunk],
                              dtype=dtype, device=device)
        sig_V, sig_A, sig_s, sig_m = torch.exp(blk[:, 0]), torch.exp(blk[:, 1]), \
            torch.exp(blk[:, 2]), torch.exp(blk[:, 3])
        p_same = torch.sigmoid(blk[:, 4])

        v_V, v_A, v_s = sig_V**2, sig_A**2, sig_s**2
        iv_V, iv_A, iv_s = 1.0 / v_V, 1.0 / v_A, 1.0 / v_s

        a_, b_, d_ = v_V + v_s, v_s, v_A + v_s
        det_c1 = a_ * d_ - b_ * b_
        inv00, inv11, inv01 = d_ / det_c1, a_ / det_c1, -b_ / det_c1
        log_norm_c1 = -0.5 * (math.log((2 * math.pi) ** 2) + torch.log(det_c1))

        v_Vbar, v_Abar = v_V + v_s, v_A + v_s
        log_norm_c2_V = -0.5 * (math.log(2 * math.pi) + torch.log(v_Vbar))
        log_norm_c2_A = -0.5 * (math.log(2 * math.pi) + torch.log(v_Abar))

        weight_sum_c1 = iv_V + iv_A + iv_s
        weight_V, weight_A = iv_V + iv_s, iv_A + iv_s

        # (T, 1, n, 1) and (T, 1, 1, n): the quadrature grid is theta-dependent
        # because it is scaled by the sensory noise being integrated out
        rel_V = (sig_V * math.sqrt(2.0))[:, None, None, None] * nodes[None, None, :, None]
        rel_A = (sig_A * math.sqrt(2.0))[:, None, None, None] * nodes[None, None, None, :]

        xV = S_V[None, :, None, None] + rel_V           # (T, B, n, n)
        xA = RHO_A * S_A[None, :, None, None] + rel_A   # (T, B, n, n)

        zV, zA = xV - mu_p, xA - mu_p
        e = lambda t: t[:, None, None, None]  # noqa: E731  (T,) -> (T,1,1,1)
        quad_c1 = e(inv00) * zV * zV + 2 * e(inv01) * zV * zA + e(inv11) * zA * zA
        log_p_c1 = e(log_norm_c1) - 0.5 * quad_c1
        log_p_c2 = (
            e(log_norm_c2_V) - 0.5 * zV**2 / e(1.0 / iv_V + 1.0 / iv_s)
            + e(log_norm_c2_A) - 0.5 * zA**2 / e(1.0 / iv_A + 1.0 / iv_s)
        )

        log_ps = torch.log(p_same)[:, None, None, None]
        log_1mps = torch.log1p(-p_same)[:, None, None, None]
        post_c1 = torch.exp(
            log_ps + log_p_c1
            - torch.logaddexp(log_ps + log_p_c1, log_1mps + log_p_c2))

        mu_c1 = (e(iv_V) * xV + e(iv_A) * xA + e(iv_s) * mu_p) / e(weight_sum_c1)
        mu_c2_V = (e(iv_V) * xV + e(iv_s) * mu_p) / e(weight_V)
        mu_c2_A = (e(iv_A) * xA + e(iv_s) * mu_p) / e(weight_A)
        mu_c2 = torch.where(rt[None, :, None, None] == 0, mu_c2_V, mu_c2_A)

        s_hat = post_c1 * mu_c1 + (1.0 - post_c1) * mu_c2

        sm = e(sig_m)
        ll_r = torch.exp(-0.5 * ((R[None, :, None, None] - s_hat) / sm) ** 2) \
            / (math.sqrt(2.0 * math.pi) * sm)
        prob_r = torch.sum(ll_r * weight_mat[None, None, :, :], dim=(2, 3))  # (T,B)
        prob_r = (1.0 - lapse) * prob_r + lapse / LAPSE_SPREAD
        ll = torch.sum(torch.log(prob_r + RESPONSE_FLOOR), dim=1)  # (T,)
        out[start:start + blk.shape[0]] = ll.detach().cpu().numpy()
    return out


def log_likelihood_model_coords(y: np.ndarray, x_o_standardised: np.ndarray, *,
                                cfg: LikelihoodConfig = LikelihoodConfig(),
                                upstream: Optional[str] = None) -> np.ndarray:
    """The whole chain: model coordinates + standardised responses in, ``log p`` out.

    Callers outside this module should use this entry point so that both
    inverse transforms are always applied.
    """
    theta = model_to_physical(y)
    raw = unstandardise_responses(np.asarray(x_o_standardised).reshape(-1), upstream)
    return log_likelihood(theta, raw, cfg=cfg)


# ------------------------------------------------------------- JAX backend ---
#
# The torch build used here is CPU-only, so the torch path above cannot use a
# GPU, and a tempered-SMC reference needs millions of likelihood evaluations.
# JAX in the same environment has the CUDA 12 plugin, so the same algebra is
# written again against ``jax.numpy``.  The torch version above stays closest
# to upstream's scalar likelihood and serves as the reference for this one.


def _jax_setup():
    import jax

    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)
    return jax


def log_likelihood_jax(theta_phys: np.ndarray, responses_raw: np.ndarray, *,
                       gh_deg: int = 301, theta_chunk: int = 64):
    """``log p(R | theta)`` for a batch of parameters, on whatever JAX sees.

    Same algebra, same constants and the same ``+1e-12`` guard as
    :func:`log_likelihood`.  float64 is enabled because the quadrature
    convergence check compares differences of order 1e-2 nats in a number of
    order 1e2.
    """
    jax = _jax_setup()
    import jax.numpy as jnp

    theta = np.atleast_2d(np.asarray(theta_phys, float))
    if theta.shape[-1] != THETA_DIM:
        raise ValueError(f"BAV theta is {THETA_DIM}-dimensional, got {theta.shape}")
    R = jnp.asarray(np.asarray(responses_raw, float).reshape(-1))
    if R.shape[0] != NUM_TRIALS:
        raise ValueError(f"expected {NUM_TRIALS} responses, got {R.shape[0]}")

    S_V_np, S_A_np, rt_np = stimulus_grid()
    S_V, S_A = jnp.asarray(S_V_np), jnp.asarray(S_A_np)
    rt = jnp.asarray(rt_np)
    y_np, w_np = gauss_hermite(gh_deg)
    nodes, weights = jnp.asarray(y_np), jnp.asarray(w_np)
    weight_mat = (weights[:, None] * weights[None, :]) / math.pi

    def block(blk):
        sig_V, sig_A = jnp.exp(blk[:, 0]), jnp.exp(blk[:, 1])
        sig_s, sig_m = jnp.exp(blk[:, 2]), jnp.exp(blk[:, 3])
        p_same = jax.nn.sigmoid(blk[:, 4])

        v_V, v_A, v_s = sig_V**2, sig_A**2, sig_s**2
        iv_V, iv_A, iv_s = 1.0 / v_V, 1.0 / v_A, 1.0 / v_s
        a_, b_, d_ = v_V + v_s, v_s, v_A + v_s
        det_c1 = a_ * d_ - b_ * b_
        inv00, inv11, inv01 = d_ / det_c1, a_ / det_c1, -b_ / det_c1
        log_norm_c1 = -0.5 * (math.log((2 * math.pi) ** 2) + jnp.log(det_c1))
        log_norm_c2_V = -0.5 * (math.log(2 * math.pi) + jnp.log(v_V + v_s))
        log_norm_c2_A = -0.5 * (math.log(2 * math.pi) + jnp.log(v_A + v_s))
        weight_sum_c1 = iv_V + iv_A + iv_s
        weight_V, weight_A = iv_V + iv_s, iv_A + iv_s

        e = lambda t: t[:, None, None, None]  # noqa: E731
        rel_V = e(sig_V * math.sqrt(2.0)) * nodes[None, None, :, None]
        rel_A = e(sig_A * math.sqrt(2.0)) * nodes[None, None, None, :]
        xV = S_V[None, :, None, None] + rel_V
        xA = RHO_A * S_A[None, :, None, None] + rel_A

        zV, zA = xV - MU_P_FIXED, xA - MU_P_FIXED
        quad_c1 = e(inv00) * zV * zV + 2 * e(inv01) * zV * zA + e(inv11) * zA * zA
        log_p_c1 = e(log_norm_c1) - 0.5 * quad_c1
        log_p_c2 = (e(log_norm_c2_V) - 0.5 * zV**2 / e(1.0 / iv_V + 1.0 / iv_s)
                    + e(log_norm_c2_A) - 0.5 * zA**2 / e(1.0 / iv_A + 1.0 / iv_s))

        log_ps, log_1mps = e(jnp.log(p_same)), e(jnp.log1p(-p_same))
        post_c1 = jnp.exp(log_ps + log_p_c1
                          - jnp.logaddexp(log_ps + log_p_c1, log_1mps + log_p_c2))

        mu_c1 = (e(iv_V) * xV + e(iv_A) * xA + e(iv_s) * MU_P_FIXED) / e(weight_sum_c1)
        mu_c2_V = (e(iv_V) * xV + e(iv_s) * MU_P_FIXED) / e(weight_V)
        mu_c2_A = (e(iv_A) * xA + e(iv_s) * MU_P_FIXED) / e(weight_A)
        mu_c2 = jnp.where(rt[None, :, None, None] == 0, mu_c2_V, mu_c2_A)
        s_hat = post_c1 * mu_c1 + (1.0 - post_c1) * mu_c2

        sm = e(sig_m)
        ll_r = jnp.exp(-0.5 * ((R[None, :, None, None] - s_hat) / sm) ** 2) \
            / (math.sqrt(2.0 * math.pi) * sm)
        prob_r = jnp.sum(ll_r * weight_mat[None, None, :, :], axis=(2, 3))
        prob_r = (1.0 - LAPSE_FIXED) * prob_r + LAPSE_FIXED / LAPSE_SPREAD
        return jnp.sum(jnp.log(prob_r + RESPONSE_FLOOR), axis=1)

    fn = jax.jit(block)
    out = np.empty(theta.shape[0], float)
    for start in range(0, theta.shape[0], theta_chunk):
        blk = theta[start:start + theta_chunk]
        pad = theta_chunk - blk.shape[0]
        # pad to a fixed shape so jit compiles once, not once per ragged tail
        if pad:
            blk = np.concatenate([blk, np.repeat(blk[-1:], pad, axis=0)])
        vals = np.asarray(fn(jnp.asarray(blk)))
        out[start:start + theta_chunk - pad] = vals[: theta_chunk - pad]
    return out


def log_likelihood_model_coords_jax(y: np.ndarray, x_o_standardised: np.ndarray, *,
                                    gh_deg: int = 301, theta_chunk: int = 64,
                                    upstream: Optional[str] = None) -> np.ndarray:
    theta = model_to_physical(y)
    raw = unstandardise_responses(np.asarray(x_o_standardised).reshape(-1), upstream)
    return log_likelihood_jax(theta, raw, gh_deg=gh_deg, theta_chunk=theta_chunk)
