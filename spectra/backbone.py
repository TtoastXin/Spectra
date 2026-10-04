"""The frozen Simformer backbone as a batched target-score hook.

Every method queries one object:

    (z_theta, t, x_o) -> s_p(z_theta, t, x_o)

Upstream has two conditioning semantics, and they give different functions:

``conditional`` (the main protocol)
    the real posterior condition mask is handed to the network, so the theta
    block of the output is ``grad_theta log p_t(theta_t | x_o)``.  This is what
    ``AllConditionalScoreModel._sample`` uses, and it is the object the exact
    transport identities are stated for.

``joint_inpaint`` (upstream PriorGuide parity only)
    ``prior_guide_theta_prior_only`` calls ``_init_score(..., condition_mask=
    jnp.zeros_like(condition_mask))`` and pins the observation by hard
    inpainting instead.  Provided so that Protocol B can reproduce upstream.

Checkpoints are read with a stubbing unpickler: upstream's ``__setstate__``
imports ``sim.tasks.task`` and therefore torch/sbibm, which cannot share a
process with JAX on this cluster.  Only plain arrays and config dicts are taken
out, and the model is rebuilt from this repository's transcription of
``scalar_transformer_model``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from spectra.simformer import (
    SimformerConfig,
    VESchedule,
    build_model,
    upstream_nn,
)

CONDITIONING_MODES = ("conditional", "joint_inpaint")


class _StubState:
    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _StubUnpickler(pickle.Unpickler):
    """Load an upstream model pickle without importing torch/sbibm."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("priorg", "sim", "scoresbibm"):
            return type(name, (_StubState,), {})
        return super().find_class(module, name)


@dataclass
class CheckpointInfo:
    """Everything this repository needs out of a Simformer checkpoint."""

    params: dict
    theta_dim: int
    x_dim: int
    schedule: VESchedule
    model_cfg: SimformerConfig
    marginal_end_mean: np.ndarray  # (num_nodes,)
    marginal_end_std: np.ndarray  # (num_nodes,)
    source: str  # "upstream" | "spectra"
    path: str
    meta: dict


def _schedule_from_sde_params(sde: dict) -> VESchedule:
    if sde.get("name", "vesde").lower() != "vesde":
        raise ValueError(f"only the VE SDE is supported, got {sde.get('name')!r}")
    return VESchedule(
        sigma_min=float(sde["sigma_min"]),
        sigma_max=float(sde["sigma_max"]),
        t_min=float(sde["T_min"]),
        t_max=float(sde["T_max"]),
        scale_min=float(sde.get("scale_min", 1e-3)),
    )


def load_upstream_checkpoint(path) -> CheckpointInfo:
    """Read one of upstream's ``experiments/models/<task>/model_<id>.pkl``."""
    path = Path(path)
    with open(path, "rb") as fh:
        state = _StubUnpickler(fh).load().__dict__["_state"]
    cond = np.asarray(state["condition_mask"])
    theta_dim = int((~cond).sum())
    x_dim = int(cond.sum())
    if not (cond[:theta_dim] == False).all() or not cond[theta_dim:].all():
        raise ValueError(f"unexpected condition mask layout in {path}: {cond}")
    mcfg = dict(state["model_init_params"])
    num_nodes = mcfg.pop("num_nodes")
    if num_nodes != theta_dim + x_dim:
        raise ValueError(f"num_nodes {num_nodes} != {theta_dim} + {x_dim}")
    mcfg.pop("layer_norm", None)  # upstream accepts it but never forwards it
    sde = dict(state["sde_init_params"])
    data = sde.pop("data", None)
    return CheckpointInfo(
        params=jax.tree_util.tree_map(jnp.asarray, state["params"]),
        theta_dim=theta_dim,
        x_dim=x_dim,
        schedule=_schedule_from_sde_params(sde),
        model_cfg=SimformerConfig(**mcfg),
        marginal_end_mean=np.asarray(state["marginal_end_mean"], float),
        marginal_end_std=np.asarray(state["marginal_end_std"], float),
        source="upstream",
        path=str(path),
        meta={
            "sampling_kwargs": dict(state.get("sampling_kwargs") or {}),
            "edge_mask_fn_params": dict(state.get("edge_mask_fn_params") or {}),
            "z_score_params": state.get("z_score_params"),
            "num_training_rows": None if data is None else int(np.asarray(data).shape[0]),
        },
    )


def load_spectra_checkpoint(path) -> CheckpointInfo:
    """Read a checkpoint written by ``scripts/train_model.py``."""
    path = Path(path)
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return CheckpointInfo(
        params=jax.tree_util.tree_map(jnp.asarray, payload["params"]),
        theta_dim=int(payload["theta_dim"]),
        x_dim=int(payload["x_dim"]),
        schedule=VESchedule(**payload["schedule"]),
        model_cfg=SimformerConfig(**payload["model_cfg"]),
        marginal_end_mean=np.asarray(payload["marginal_end_mean"], float),
        marginal_end_std=np.asarray(payload["marginal_end_std"], float),
        source="spectra",
        path=str(path),
        meta=payload.get("meta", {}),
    )


def load_checkpoint(path) -> CheckpointInfo:
    """Dispatch on the payload layout; upstream pickles are class instances."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if b"AllConditionalScoreModel" in head:
        return load_upstream_checkpoint(path)
    return load_spectra_checkpoint(path)


class Backbone:
    """Batched base-score queries against a frozen Simformer checkpoint."""

    def __init__(self, ckpt: CheckpointInfo, mode: str = "conditional", nn=None):
        if mode not in CONDITIONING_MODES:
            raise ValueError(f"mode must be one of {CONDITIONING_MODES}, got {mode!r}")
        self.ckpt = ckpt
        self.mode = mode
        self.schedule = ckpt.schedule
        self.theta_dim = ckpt.theta_dim
        self.x_dim = ckpt.x_dim
        self.num_nodes = ckpt.theta_dim + ckpt.x_dim
        nn = nn if nn is not None else upstream_nn()
        _, self._model_fn = build_model(self.num_nodes, ckpt.schedule, ckpt.model_cfg, nn)
        self._params = ckpt.params
        self._node_id = jnp.arange(self.num_nodes)
        posterior = jnp.array(
            [False] * self.theta_dim + [True] * self.x_dim, dtype=jnp.bool_
        )
        self._cond_network = (
            posterior[None, :]
            if mode == "conditional"
            else jnp.zeros((1, self.num_nodes), dtype=jnp.bool_)
        )

    # ------------------------------------------------------------- queries --

    def score_theta_t(self, z: jnp.ndarray, t, x_o: jnp.ndarray) -> jnp.ndarray:
        """``s_p(z, t, x_o)`` for ``z`` of shape ``(B, theta_dim)``, scalar ``t``."""
        z = jnp.atleast_2d(z)
        batch = z.shape[0]
        xs = jnp.broadcast_to(jnp.asarray(x_o, z.dtype), (batch, self.x_dim))
        data = jnp.concatenate([z, xs], axis=-1)[..., None]
        tt = jnp.broadcast_to(jnp.asarray(t, z.dtype), (batch,))
        out = self._model_fn(
            self._params, tt, data, self._node_id, self._cond_network, None, None
        )
        return out[:, : self.theta_dim, 0]

    def score_theta_tau(self, z: jnp.ndarray, tau, x_o: jnp.ndarray) -> jnp.ndarray:
        """Same query addressed by the noise variance ``tau = sigma(t)^2``."""
        return self.score_theta_t(z, self.schedule.t_of_tau(jnp.asarray(tau)), x_o)

    def denoiser_t(self, z: jnp.ndarray, t, x_o: jnp.ndarray) -> jnp.ndarray:
        """Tweedie mean ``m = z + tau s_p`` (VE, so the marginal mean is 1)."""
        tau = self.schedule.sigma(jnp.asarray(t)) ** 2
        return z + tau * self.score_theta_t(z, t, x_o)

    def denoiser_jacobian_t(self, z: jnp.ndarray, t, x_o: jnp.ndarray) -> jnp.ndarray:
        """``d m / d z`` restricted to the theta block, per batch row.

        PG-FullCov needs only the theta--theta block, so this uses ``jacrev`` over
        ``theta_dim`` outputs rather than upstream's full ``(D, D)`` ``jacfwd``.
        """
        def single(zi):
            return self.denoiser_t(zi[None, :], t, x_o)[0]

        return jax.vmap(jax.jacrev(single))(jnp.atleast_2d(z))

    # --------------------------------------------------------- diagnostics --

    def t_of_tau(self, tau):
        return self.schedule.t_of_tau(jnp.asarray(tau))

    def nominal_t_range(self) -> tuple[float, float]:
        return float(self.schedule.t_min), float(self.schedule.t_max)

    def terminal_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Base Simformer's per-node terminal mean/std, restricted to theta."""
        return (
            self.ckpt.marginal_end_mean[: self.theta_dim],
            self.ckpt.marginal_end_std[: self.theta_dim],
        )
