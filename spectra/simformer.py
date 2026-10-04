"""Frozen learned base score: the upstream Simformer on a VE-SDE.

This module provides the learned base score ``s_p_hat(z, tau)`` that the
samplers use in place of the analytic base score ``s_p*(z, tau)``.

What comes from upstream and what is transcribed
------------------------------------------------
The architecture (``ScalarTokenizer`` + ``GaussianFourierEmbedding`` +
``Transformer``) and the denoising-score-matching loss are imported unchanged
from the upstream checkout at commit b4852fc1a3c37eeec71affa0078874e1977d8cbd.
Only the assembly glue of ``methods.scalar_transformer_model`` and the VE
branch of ``methods.init_sde_related`` is transcribed here, because upstream's
``priorg/sim/methods/methods.py`` imports ``sim.tasks.task`` and therefore pulls
torch/sbibm into the process, and torch and JAX cannot share one process on
this cluster.

VE convention
-------------
``tau`` is the noise variance everywhere in this repo, so with

    sigma(t) = sigma_min (sigma_max / sigma_min)^t,   tau(t) = sigma(t)^2,

``p_tau = p_0 * N(0, tau I)``.  Upstream's ``T_min``/``T_max`` bound ``t``, not
``tau``; the evaluation grid tau in {0.05, 0.2, 0.8} maps to a narrow
window inside [T_min, T_max] (see ``t_of_tau``).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp

UPSTREAM_COMMIT = "b4852fc1a3c37eeec71affa0078874e1977d8cbd"
VENDOR_DIR = Path(__file__).resolve().parents[1] / "third_party" / "priorguide"


def _upstream_root(root: str | os.PathLike | None = None) -> Path:
    """The vendored PriorGuide code in ``third_party/priorguide`` unless ``root`` is given."""
    path = VENDOR_DIR if root is None else Path(root).expanduser().resolve()
    if not (path / "sim" / "nn" / "transformers.py").is_file():
        raise RuntimeError(f"{path} does not contain the vendored PriorGuide sim/ package")
    return path


def upstream_nn(root: str | os.PathLike | None = None):
    """Import the pure-JAX upstream building blocks (no torch, no sbibm)."""
    path = _upstream_root(root)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from sim.nn.helpers import GaussianFourierEmbedding
    from sim.nn.loss_fn import denoising_score_matching_loss
    from sim.nn.tokenizer import ScalarTokenizer
    from sim.nn.transformers import Transformer

    return {
        "Transformer": Transformer,
        "ScalarTokenizer": ScalarTokenizer,
        "GaussianFourierEmbedding": GaussianFourierEmbedding,
        "dsm_loss": denoising_score_matching_loss,
    }


# ---------------------------------------------------------------- schedule --


@dataclass(frozen=True)
class VESchedule:
    """Upstream ``benchmark.yaml`` VE-SDE defaults."""

    sigma_min: float = 1e-4
    sigma_max: float = 15.0
    t_min: float = 1e-5
    t_max: float = 1.0
    scale_min: float = 1e-3

    @property
    def log_ratio(self) -> float:
        return math.log(self.sigma_max / self.sigma_min)

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def tau(self, t):
        return self.sigma(t) ** 2

    def t_of_tau(self, tau):
        """Inverse of ``tau(t)``; ``tau`` is the variance, so sigma = sqrt(tau)."""
        return jnp.log(jnp.sqrt(tau) / self.sigma_min) / self.log_ratio

    def weight(self, t):
        """Upstream training weight: ``diffusion(t)**2 = sigma(t)**2 * 2 log(ratio)``."""
        return self.sigma(t) ** 2 * (2.0 * self.log_ratio)

    def output_scale(self, t):
        """Upstream ``output_scale_fn``: divide the network output by clipped sigma."""
        return jnp.clip(self.sigma(t), self.scale_min)


@dataclass(frozen=True)
class SimformerConfig:
    """Upstream ``benchmark.yaml`` ``method.model`` defaults."""

    token_dim: int = 40
    condition_token_dim: int = 10
    condition_token_init_scale: float = 0.1
    condition_token_init_mean: float = 0.0
    condition_mode: str = "concat"
    time_embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 6
    attn_size: int = 10
    widening_factor: int = 3
    num_hidden_layers: int = 1
    skip_connection_attn: bool = True
    skip_connection_mlp: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Upstream ``benchmark.yaml`` ``method.train`` defaults."""

    num_sims: int = 10000
    batch_size: int = 1000
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    clip_max_norm: float = 10.0
    total_number_steps_scaling: int = 3
    max_number_steps: int = 100000
    min_number_steps: int = 5000
    validation_fraction: float = 0.05
    val_repeat: int = 5
    val_every: int = 50
    val_error_ratio: float = 1.1
    stop_early_count: int = 5

    def total_steps(self, num_rows: int) -> int:
        return int(
            max(
                min(num_rows * self.total_number_steps_scaling, self.max_number_steps),
                self.min_number_steps,
            )
        )


# ------------------------------------------------------------------- model --


def build_model(num_nodes: int, schedule: VESchedule, cfg: SimformerConfig, nn: dict):
    """Transcription of upstream ``scalar_transformer_model`` for the VE branch."""
    Transformer = nn["Transformer"]
    ScalarTokenizer = nn["ScalarTokenizer"]
    GaussianFourierEmbedding = nn["GaussianFourierEmbedding"]

    token_dim = cfg.token_dim
    condition_token_dim = cfg.condition_token_dim
    if cfg.condition_mode == "add":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = token_dim
    elif cfg.condition_mode == "none":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = 0
    elif cfg.condition_mode != "concat":
        raise ValueError(f"unknown condition_mode {cfg.condition_mode!r}")

    def model(t, data, data_id, condition_mask, meta_data=None, edge_mask=None):
        _, current_nodes, _ = data.shape  # (batch, nodes, 1)
        data_id = data_id.reshape(-1, current_nodes)
        condition_mask = condition_mask.reshape(-1, current_nodes)

        tokenizer = ScalarTokenizer(token_dim, num_nodes)
        time_embeder = GaussianFourierEmbedding(cfg.time_embedding_dim)

        tokens = tokenizer(data_id, data, meta_data)
        time = time_embeder(t[..., None])

        if cfg.condition_mode != "none":
            condition_token = hk.get_parameter(
                "condition_token",
                shape=[1, 1, condition_token_dim],
                init=hk.initializers.RandomNormal(
                    cfg.condition_token_init_scale, cfg.condition_token_init_mean
                ),
            )
            condition_mask = condition_mask.reshape(-1, current_nodes, 1)
            condition_token = condition_mask * condition_token
            if cfg.condition_mode == "add":
                tokens = tokens + condition_token
            else:
                condition_token = jnp.broadcast_to(
                    condition_token, tokens.shape[:-1] + (condition_token_dim,)
                )
                tokens = jnp.concatenate([tokens, condition_token], -1)

        transformer = Transformer(
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            attn_size=cfg.attn_size,
            widening_factor=cfg.widening_factor,
            num_hidden_layers=cfg.num_hidden_layers,
            act=jax.nn.gelu,
            skip_connection_attn=cfg.skip_connection_attn,
            skip_connection_mlp=cfg.skip_connection_mlp,
        )
        h = transformer(tokens, context=time, mask=edge_mask)
        out = hk.Linear(1)(h)
        return out / schedule.output_scale(t)[..., None, None]

    return hk.without_apply_rng(hk.transform(model))
