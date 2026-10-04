"""bav_parametric_torch.py
=================================
PyTorch implementation of a Bayesian Audio-Visual (BAV) localisation
model with

* **constant sensory noise** (σ_V, σ_A),
* **Gaussian spatial prior** 𝒩(μ, σ_s²),
* **Model-Averaging causal inference**,
* **Gaussian motor noise** (σ_m), and
* **Fixed auditory rescaling** (rho = 4/3).

The module exposes two vectorised functions that both operate in an
**unbounded parameter space** — a single flat tensor ``theta`` whose
entries live on ℝ but are internally transformed to their proper ranges.

Functions
---------
```
nll_bav_constant_gaussian(theta, R, S_V, S_A, response_types, ...)
    → scalar negative log-likelihood (∑ over trials)

sample_bav_responses(theta, S_V, S_A, response_types, N=1, rng=None)
    → synthetic responses R_sim (shape: (N, batch))
```

Unconstrained parameter vector ``theta`` (length = 7)
----------------------------------------------------
```
Idx  Name        Raw value in θ        Transform           Effective range
---  ----------  --------------------  ------------------  ---------------
0    log_σ_V     any real             σ_V = exp(θ₀)        (0, ∞)
1    log_σ_A     any real             σ_A = exp(θ₁)        (0, ∞)
2    log_σ_s     any real             σ_s = exp(θ₂)        (0, ∞)
3    log_σ_m     any real             σ_m = exp(θ₃)        (0, ∞)
4    logit_lapse any real             lapse  = σ(θ₄)       (0, 1)
5    logit_p     any real             p_same = σ(θ₅)       (0, 1)
6    μ           any real             mu    = θ₆           (−∞, ∞)
```
``σ(z)`` denotes the logistic (sigmoid) function.
"""

# AI Summary: Implements Bayesian audiovisual localisation NLL & sampler with flat-array convenience wrappers.
# Core update: replaces rectangular grid integration with separable Gauss–Hermite
# quadrature (configurable order) for more accurate and efficient expectation
# under Gaussian sensory noise. Adds helper to generate nodes/weights via NumPy.
# Adds stimulus-grid caching plus `nll_bav_constant_gaussian_flat` and
# `sample_bav_responses_flat` for simplified 98-trial API.

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

__all__ = [
    "nll_bav_constant_gaussian",
    "sample_bav_responses",
    "nll_bav_constant_gaussian_flat",
    "sample_bav_responses_flat",
]

two_pi = math.sqrt(2.0 * math.pi)
RHO_A = 4.0 / 3.0  # fixed auditory-axis rescaling factor ρ_A

# ---------------------------------------------------------------------
# Canonical 7×7 stimulus grid & caching for flat-array wrappers
# ---------------------------------------------------------------------

DEFAULT_STIM_VALUES = torch.tensor(
    [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0], dtype=torch.float32
)
_GRID_CACHE: dict[
    tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _stimulus_grid(device: torch.device, dtype: torch.dtype = torch.float32):
    """
    Return (S_V, S_A, response_types) tensors for the fixed 49-trial grid.

    Results are cached per (device, dtype) pair to reduce tensor allocation
    overhead when the wrapper is called repeatedly.
    """
    key = (device, dtype)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    stim = DEFAULT_STIM_VALUES.to(device=device, dtype=dtype)
    grid = torch.cartesian_prod(stim, stim)  # (49, 2)
    S_V_grid, S_A_grid = grid[:, 0], grid[:, 1]

    S_V = torch.cat([S_V_grid, S_V_grid], dim=0)  # 49 BV + 49 BA
    S_A = torch.cat([S_A_grid, S_A_grid], dim=0)

    response_types = torch.cat(
        [
            torch.zeros(49, dtype=torch.long, device=device),  # BV
            torch.ones(49, dtype=torch.long, device=device),  # BA
        ],
        dim=0,
    )

    _GRID_CACHE[key] = (S_V, S_A, response_types)
    return _GRID_CACHE[key]


# -----------------------------------------------------------------------------
# Helper: univariate Gaussian pdf
# -----------------------------------------------------------------------------


def _gaussian_pdf(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:  # noqa: D401,E501
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (two_pi * sigma)


# -----------------------------------------------------------------------------
# Helper: Gauss–Hermite nodes and weights for ∫ e^{−x²} f(x) dx
# -----------------------------------------------------------------------------


def _gauss_hermite_tensor(
    n: int, device: torch.device, dtype: torch.dtype = torch.float32
):
    """Return nodes *y* and weights *w* (both shape (n,)) as torch tensors."""
    y, w = np.polynomial.hermite.hermgauss(n)  # physicists' Hermite; exp(−x²) weight
    nodes = torch.as_tensor(y, dtype=dtype, device=device)
    weights = torch.as_tensor(w, dtype=dtype, device=device)
    return nodes, weights


# -----------------------------------------------------------------------------
# Utility: bounded parameter conversion
# -----------------------------------------------------------------------------


def _unpack_theta(theta: torch.Tensor, device: torch.device):
    """Convert unconstrained θ → bounded parameters (scalars)."""
    theta = theta.to(device)
    sigma_V, sigma_A, sigma_s, sigma_m = torch.exp(theta[0:4])
    # lapse = torch.sigmoid(theta[4])
    lapse = torch.sigmoid(torch.logit(torch.tensor(0.02)))
    p_same = torch.sigmoid(theta[4])
    # mu_p = theta[6]
    mu_p = 0.0
    return sigma_V, sigma_A, sigma_s, sigma_m, lapse, p_same, mu_p


# -----------------------------------------------------------------------------
# Negative log‑likelihood (Gauss–Hermite integration)
# -----------------------------------------------------------------------------


def nll_bav_constant_gaussian(
    theta: torch.Tensor,
    R: torch.Tensor,
    S_V: torch.Tensor,
    S_A: torch.Tensor,
    response_types: torch.Tensor,
    *,
    gh_deg: int = 51,
    chunk_size: Optional[int] = None,
    # deprecated — retained for backward‑compatibility; ignored
    grid_step: float | None = None,
    grid_range_sd: float | None = None,
) -> torch.Tensor:
    """Summed NLL across trials for the BAV model using Gauss–Hermite quadrature.

    Parameters
    ----------
    gh_deg : int, optional
        Number of Gauss–Hermite nodes *per dimension* (default **21**). Larger
        values increase accuracy at higher computational cost.
    chunk_size : int, optional
        If provided, evaluates trials in chunks to reduce GPU memory usage.
    """

    device = R.device
    (
        sigma_V,
        sigma_A,
        sigma_s,
        sigma_m,
        lapse,
        p_same,
        mu_p,
    ) = _unpack_theta(theta, device)

    # Variances / precisions (scalars)
    v_V, v_A, v_s = sigma_V**2, sigma_A**2, sigma_s**2
    iv_V, iv_A, iv_s = 1.0 / v_V, 1.0 / v_A, 1.0 / v_s

    # --- constants for p(x|C=1) --------------------------------------
    a, b, d = v_V + v_s, v_s, v_A + v_s
    det_c1 = a * d - b * b
    inv00, inv11, inv01 = d / det_c1, a / det_c1, -b / det_c1
    log_norm_c1 = -0.5 * (math.log((2 * math.pi) ** 2) + math.log(det_c1))

    # --- constants for p(x|C=2) --------------------------------------
    v_Vbar, v_Abar = v_V + v_s, v_A + v_s
    log_norm_c2_V = -0.5 * (math.log(2 * math.pi) + math.log(v_Vbar))
    log_norm_c2_A = -0.5 * (math.log(2 * math.pi) + math.log(v_Abar))

    weight_sum_c1 = iv_V + iv_A + iv_s
    weight_V, weight_A = iv_V + iv_s, iv_A + iv_s

    # --- Gauss–Hermite grid (shared across trials) -------------------
    nodes_V, w_V = _gauss_hermite_tensor(gh_deg, device, R.dtype)
    nodes_A, w_A = _gauss_hermite_tensor(gh_deg, device, R.dtype)
    # scale nodes to measurement noise distribution:  x = √2 σ • y
    rel_V = sigma_V * math.sqrt(2.0) * nodes_V  # (N_V,)
    rel_A = sigma_A * math.sqrt(2.0) * nodes_A  # (N_A,)
    weight_mat = (w_V[:, None] * w_A[None, :]) / math.pi  # (N_V, N_A)

    # --- chunked computation -----------------------------------------
    if chunk_size is None:
        chunk_size = R.numel()

    nll_total = torch.tensor(0.0, device=device, dtype=R.dtype)
    for start in range(0, R.numel(), chunk_size):
        end = min(start + chunk_size, R.numel())
        nll_total += _chunk_nll(
            S_V[start:end],
            S_A[start:end],
            R[start:end],
            response_types[start:end],
            rel_V,
            rel_A,
            weight_mat,
            mu_p,
            sigma_m,
            lapse,
            p_same,
            iv_V,
            iv_A,
            iv_s,
            weight_sum_c1,
            weight_V,
            weight_A,
            inv00,
            inv11,
            inv01,
            log_norm_c1,
            log_norm_c2_V,
            log_norm_c2_A,
        )

    return nll_total


# -----------------------------------------------------------------------------
# Per‑chunk helper (vectorised across trials)
# -----------------------------------------------------------------------------


def _chunk_nll(
    S_V: torch.Tensor,
    S_A: torch.Tensor,
    R: torch.Tensor,
    rt: torch.Tensor,
    rel_V: torch.Tensor,
    rel_A: torch.Tensor,
    weight_mat: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_m: torch.Tensor,
    lapse: torch.Tensor,
    p_same: torch.Tensor,
    iv_V: float,
    iv_A: float,
    iv_s: float,
    weight_sum_c1: float,
    weight_V: float,
    weight_A: float,
    inv00: float,
    inv11: float,
    inv01: float,
    log_norm_c1: float,
    log_norm_c2_V: float,
    log_norm_c2_A: float,
) -> torch.Tensor:
    """Compute summed NLL for *this* batch of trials using G–H quadrature."""

    # Expand Gauss–Hermite nodes to trials
    xV = S_V[:, None, None] + rel_V[None, :, None]  # (B, N_V, N_A)
    xA = RHO_A * (S_A[:, None, None]) + rel_A[None, None, :]  # (B, N_V, N_A)

    # Posterior P(C=1|x)
    zV, zA = xV - mu_p, xA - mu_p
    quad_c1 = inv00 * zV * zV + 2 * inv01 * zV * zA + inv11 * zA * zA
    log_p_c1 = log_norm_c1 - 0.5 * quad_c1
    log_p_c2 = (
        log_norm_c2_V
        - 0.5 * (zV**2) / (1.0 / iv_V + 1.0 / iv_s)
        + log_norm_c2_A
        - 0.5 * (zA**2) / (1.0 / iv_A + 1.0 / iv_s)
    )

    log_ps = torch.log(p_same)
    post_c1 = torch.exp(
        log_ps
        + log_p_c1
        - torch.logaddexp(log_ps + log_p_c1, torch.log1p(-p_same) + log_p_c2)
    )

    # Posterior means of s
    mu_c1 = (iv_V * xV + iv_A * xA + iv_s * mu_p) / weight_sum_c1
    mu_c2_V = (iv_V * xV + iv_s * mu_p) / weight_V
    mu_c2_A = (iv_A * xA + iv_s * mu_p) / weight_A
    mu_c2 = torch.where(rt[:, None, None] == 0, mu_c2_V, mu_c2_A)

    s_hat = post_c1 * mu_c1 + (1.0 - post_c1) * mu_c2

    # Response likelihood
    ll_r = _gaussian_pdf(R[:, None, None], s_hat, sigma_m)  # (B, N_V, N_A)
    prob_r = torch.sum(ll_r * weight_mat, dim=(1, 2))  # (B,)

    # Lapse mixture & NLL
    prob_r = (1.0 - lapse) * prob_r + lapse / 90.0
    return -torch.sum(torch.log(prob_r + 1e-12))


# ==============================================================
#  BAV RESPONSE SAMPLER  (constant-noise, model-averaging)
# ==============================================================


def sample_bav_responses(
    theta: torch.Tensor,
    S_V: torch.Tensor,
    S_A: torch.Tensor,
    response_types: torch.Tensor,
    *,
    N: int = 1,
) -> torch.Tensor:
    """
    Draw synthetic motor responses for the Bayesian Audio-Visual (BAV)
    localisation model with

    * constant sensory noise (σ_V, σ_A),
    * Gaussian spatial prior 𝒩(μ, σ_s²),
    * **model-averaging** causal inference,
    * Gaussian motor noise (σ_m), and
    * uniform-lapse probability.

    All parameters live in an **unbounded space** exactly like
    `nll_bav_constant_gaussian`:

    ╔════╤══════════════╤════════════════════════════════════╗
    ║idx │ meaning      │   forward transform                ║
    ╟────┼──────────────┼────────────────────────────────────╢
    ║ 0  │ log σ_V      │ σ_V   = exp(θ₀)                   ║
    ║ 1  │ log σ_A      │ σ_A   = exp(θ₁)                   ║
    ║ 2  │ log σ_s      │ σ_s   = exp(θ₂)                   ║
    ║ 3  │ log σ_m      │ σ_m   = exp(θ₃)                   ║
    ║ 4  │ logit(lapse) │ lapse  = sigmoid(θ₄)              ║
    ║ 5  │ logit(pₛ)    │ p_same = sigmoid(θ₅)              ║
    ║ 6  │ μ            │ mu     = θ₆                       ║
    ╚════╧══════════════╧════════════════════════════════════╝

    Parameters
    ----------
    theta : (7,) tensor
        Flat unconstrained parameter vector.
    S_V, S_A : (T,) tensors
        True stimulus locations (deg) for each trial.
    response_types : (T,) tensor
        0 → BV (visual report), 1 → BA (auditory report).
    N : int, optional
        Number of responses *per trial* to sample (default 1).

    Returns
    -------
    R_sim : (N, T) tensor
        Simulated responses (deg).
    """
    device = S_V.device
    theta = theta.to(device)

    # -----------------------------------------------------------
    # 1.  Unpack θ  (all scalar tensors)
    # -----------------------------------------------------------
    sigma_V, sigma_A, sigma_s, sigma_m = torch.exp(theta[:4])
    lapse = torch.sigmoid(torch.logit(torch.tensor(0.02)))
    p_same = torch.sigmoid(theta[4])
    mu_p = 0

    v_V, v_A, v_s = sigma_V**2, sigma_A**2, sigma_s**2
    iv_V, iv_A, iv_s = 1.0 / v_V, 1.0 / v_A, 1.0 / v_s

    # -----------------------------------------------------------
    # 2.  Draw sensory measurements  x_V , x_A
    # -----------------------------------------------------------
    T = S_V.numel()
    x_V = S_V.unsqueeze(0) + sigma_V * torch.randn((N, T), device=device)

    # Auditory rescaling
    x_A = (RHO_A * S_A).unsqueeze(0) + sigma_A * torch.randn((N, T), device=device)

    # -----------------------------------------------------------
    # 3.  Compute posterior Pr(C=1 | x_V, x_A)  (vectorised)
    # -----------------------------------------------------------
    # Constants for p(x | C=1)  ~ N([μ, μ],  Σ_C1 )
    a, b, d = v_V + v_s, v_s, v_A + v_s  # Σ_C1 entries
    det_c1 = a * d - b * b
    inv00, inv11, inv01 = d / det_c1, a / det_c1, -b / det_c1
    log_norm_c1 = -0.5 * (math.log((2 * math.pi) ** 2) + math.log(det_c1))

    # Constants for independent-cause likelihood
    v_V_bar, v_A_bar = v_V + v_s, v_A + v_s
    log_norm_c2_V = -0.5 * (math.log(2 * math.pi) + math.log(v_V_bar))
    log_norm_c2_A = -0.5 * (math.log(2 * math.pi) + math.log(v_A_bar))

    zV = x_V - mu_p
    zA = x_A - mu_p

    quad_c1 = inv00 * zV**2 + 2 * inv01 * zV * zA + inv11 * zA**2
    log_p_c1 = log_norm_c1 - 0.5 * quad_c1

    log_p_c2 = (
        log_norm_c2_V - 0.5 * zV**2 / v_V_bar + log_norm_c2_A - 0.5 * zA**2 / v_A_bar
    )

    #   P(C=1|x)   (shape: N × T)
    logit_pc1 = torch.log(p_same) + log_p_c1 - (torch.log1p(-p_same) + log_p_c2)
    post_c1 = torch.sigmoid(logit_pc1)

    # -----------------------------------------------------------
    # 4.  Posterior means  μ̂_C1  and  μ̂_C2
    # -----------------------------------------------------------
    weight_sum_c1 = iv_V + iv_A + iv_s
    weight_V = iv_V + iv_s
    weight_A = iv_A + iv_s

    mu_c1 = (iv_V * x_V + iv_A * x_A + iv_s * mu_p) / weight_sum_c1
    mu_c2_V = (iv_V * x_V + iv_s * mu_p) / weight_V
    mu_c2_A = (iv_A * x_A + iv_s * mu_p) / weight_A

    # Choose μ̂_C2 according to requested report (BV or BA)
    rt = response_types.unsqueeze(0)  # shape (1, T)
    mu_c2 = torch.where(rt == 0, mu_c2_V, mu_c2_A)

    # Model-averaged estimate
    s_hat = post_c1 * mu_c1 + (1.0 - post_c1) * mu_c2

    # -----------------------------------------------------------
    # 5.  Add motor noise  &  lapses
    # -----------------------------------------------------------
    R_noisy = s_hat + sigma_m * torch.randn_like(s_hat)

    if lapse > 0.0:
        lapse_mask = torch.rand_like(R_noisy) < lapse
        R_uniform = -45.0 + 90.0 * torch.rand_like(R_noisy)
        R_noisy = torch.where(lapse_mask, R_uniform, R_noisy)

    return R_noisy


# ---------------------------------------------------------------------
# Flat-array convenience wrappers
# ---------------------------------------------------------------------


def nll_bav_constant_gaussian_flat(
    theta: torch.Tensor,
    R_flat: torch.Tensor,
    *,
    gh_deg: int = 51,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Wrapper around :func:`nll_bav_constant_gaussian` that accepts a single
    vector ``R_flat`` of length 98 (49 visual-report trials followed by
    49 auditory-report trials).
    """
    if R_flat.numel() != 98:
        raise ValueError("R_flat must contain exactly 98 elements (49 BV + 49 BA).")

    device, dtype = R_flat.device, R_flat.dtype
    S_V, S_A, rt = _stimulus_grid(device, dtype)

    return nll_bav_constant_gaussian(
        theta, R_flat, S_V, S_A, rt, gh_deg=gh_deg, chunk_size=chunk_size
    )


def sample_bav_responses_flat(
    theta: torch.Tensor,
    *,
    N: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Draw synthetic responses arranged in the canonical (N, 98) flat format.
    """
    if device is None:
        device = torch.device("cpu")

    S_V, S_A, rt = _stimulus_grid(device, dtype)
    return sample_bav_responses(theta.to(device), S_V, S_A, rt, N=N)


# ---------------------------------------------------------------------
# BAV task normalizing parameters
# caculated from 1 million simulations
BAV_X_MEAN = [
    -14.742755889892578,
    -12.508395195007324,
    -11.415111541748047,
    -11.793352127075195,
    -11.883299827575684,
    -11.884356498718262,
    -11.900609970092773,
    -9.90496826171875,
    -9.588603019714355,
    -7.753718376159668,
    -7.6542439460754395,
    -7.873549938201904,
    -7.913719654083252,
    -7.922729969024658,
    -4.511630535125732,
    -5.014579772949219,
    -4.620617389678955,
    -3.507732629776001,
    -3.75826096534729,
    -3.9147520065307617,
    -3.9585680961608887,
    -0.0836537778377533,
    -0.24739369750022888,
    -0.601858377456665,
    -0.009597082622349262,
    0.6069590449333191,
    0.24985386431217194,
    0.08988672494888306,
    3.951622724533081,
    3.9189789295196533,
    3.769181966781616,
    3.506509780883789,
    4.616843223571777,
    5.011394500732422,
    4.514711856842041,
    7.9257588386535645,
    7.9214582443237305,
    7.881976127624512,
    7.650680065155029,
    7.739823818206787,
    9.594401359558105,
    9.904319763183594,
    11.895862579345703,
    11.898467063903809,
    11.883268356323242,
    11.788851737976074,
    11.410938262939453,
    12.510252952575684,
    14.743147850036621,
    -15.754658699035645,
    -12.220104217529297,
    -6.639216899871826,
    -0.18425652384757996,
    5.269269943237305,
    10.573128700256348,
    15.860469818115234,
    -15.294130325317383,
    -10.553507804870605,
    -6.450826644897461,
    -0.42264899611473083,
    5.229361057281494,
    10.571131706237793,
    15.858860969543457,
    -15.651045799255371,
    -10.129307746887207,
    -5.279376029968262,
    -0.6234456300735474,
    5.095302581787109,
    10.53922176361084,
    15.84988784790039,
    -15.822528839111328,
    -10.428112030029297,
    -4.854170799255371,
    0.0015709053259342909,
    4.854153633117676,
    10.42347240447998,
    15.822242736816406,
    -15.85260009765625,
    -10.544486045837402,
    -5.09999942779541,
    0.6220424771308899,
    5.2829203605651855,
    10.128799438476562,
    15.647370338439941,
    -15.859649658203125,
    -10.565225601196289,
    -5.22694730758667,
    0.4120264947414398,
    6.455163955688477,
    10.551239967346191,
    15.292581558227539,
    -15.854569435119629,
    -10.56982707977295,
    -5.27540922164917,
    0.18439295887947083,
    6.634737014770508,
    12.220427513122559,
    15.754461288452148,
]

BAV_X_STD = [
    4.902778625488281,
    4.580568313598633,
    5.034599304199219,
    5.116972923278809,
    5.074518203735352,
    5.093060493469238,
    5.049951553344727,
    4.891404628753662,
    4.466029167175293,
    4.292824745178223,
    4.6247358322143555,
    4.595235824584961,
    4.570239543914795,
    4.528079032897949,
    4.426291465759277,
    4.457054138183594,
    4.102391719818115,
    4.201342582702637,
    4.312755584716797,
    4.241082668304443,
    4.17235803604126,
    4.140349388122559,
    4.214919090270996,
    4.184858798980713,
    3.96475887298584,
    4.227117538452148,
    4.2155890464782715,
    4.136454105377197,
    4.212174415588379,
    4.222695827484131,
    4.285384654998779,
    4.162843704223633,
    4.120907306671143,
    4.469606399536133,
    4.424351215362549,
    4.535003185272217,
    4.546449184417725,
    4.582648754119873,
    4.612363338470459,
    4.324263095855713,
    4.450297832489014,
    4.916301250457764,
    5.02979040145874,
    5.061990737915039,
    5.057198524475098,
    5.1228556632995605,
    5.015654563903809,
    4.576381206512451,
    4.914848804473877,
    5.05555534362793,
    4.54797887802124,
    4.595488548278809,
    4.171817779541016,
    4.288653373718262,
    4.8674235343933105,
    5.713928699493408,
    5.628124237060547,
    4.476745128631592,
    4.281273365020752,
    4.237773895263672,
    4.322243690490723,
    4.857677936553955,
    5.7033843994140625,
    5.838009357452393,
    4.861903190612793,
    4.106135845184326,
    4.120853424072266,
    4.378781318664551,
    4.896511554718018,
    5.710056304931641,
    5.735380172729492,
    4.951475143432617,
    4.325631141662598,
    3.979309320449829,
    4.3199872970581055,
    4.946342468261719,
    5.750234127044678,
    5.71847677230835,
    4.888073921203613,
    4.38369083404541,
    4.132542610168457,
    4.1152472496032715,
    4.872477054595947,
    5.84484338760376,
    5.706189155578613,
    4.867284297943115,
    4.341814994812012,
    4.270308971405029,
    4.268865585327148,
    4.487931728363037,
    5.656129360198975,
    5.713083744049072,
    4.877927780151367,
    4.272125244140625,
    4.185139179229736,
    4.579892158508301,
    4.569288730621338,
    5.059678077697754,
]

BAV_THETA_MEAN = [np.log(2), np.log(2), np.log(5), np.log(0.3), 0]
BAV_THETA_STD = [0.35, 0.35, 0.5, 0.35, 1]
