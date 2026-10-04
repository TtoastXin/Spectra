"""Simformer training: upstream's benchmark recipe on real task data.

The backbone is an ``(theta, x)`` joint over the benchmark tasks, trained with
upstream's ``structured_random`` conditioning so the posterior condition mask
the sampler uses is one the model saw during training.

Everything follows ``priorg/cfg/benchmark.yaml`` and
``methods.run_score_transformer``: VE-SDE, no z-scoring, uniform times on
``[T_min, T_max]``, denoising score matching weighted by ``diffusion(t)^2``,
``adaptive_grad_clip(10) + adam`` on a linear schedule, ``3 x num_rows`` steps
capped at 100k, and early stopping on the validation ratio.  The one
difference is single-device ``jit`` instead of ``pmap`` over devices; with one
GPU the update is the same computation, only the key layout differs.

The random stream is matched to upstream: ``key = PRNGKey(seed)``, one split for
the data (consumed by the simulation job), then one split for training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from spectra.simformer import (
    SimformerConfig,
    TrainConfig,
    VESchedule,
    build_model,
)


@dataclass
class TrainResult:
    params: dict
    history: list = field(default_factory=list)
    steps_run: int = 0
    early_stopped: bool = False
    timing: dict = field(default_factory=dict)
    marginal_end_mean: np.ndarray = None
    marginal_end_std: np.ndarray = None


def terminal_moments(data: jnp.ndarray, schedule: VESchedule):
    """Upstream's ``marginal_end_mean/std``: ``Empirical(data)`` pushed to ``T_max``."""
    mean = np.asarray(jnp.mean(data, axis=0)).reshape(-1)
    var = np.asarray(jnp.var(data, axis=0)).reshape(-1)
    return mean, np.sqrt(var + float(schedule.sigma(schedule.t_max)) ** 2)


def training_key(seed: int):
    """``key_train`` of upstream's ``train.py`` without re-drawing the data."""
    key = jax.random.PRNGKey(seed)
    key, _key_data = jax.random.split(key)
    _key, key_train = jax.random.split(key)
    return key_train


def train_model(
    theta: np.ndarray,
    x: np.ndarray,
    *,
    seed: int,
    schedule: VESchedule,
    model_cfg: SimformerConfig,
    train_cfg: TrainConfig,
    nn: dict,
    condition_mask_fn,
    steps_override: int | None = None,
    print_every_frac: int = 10,
) -> TrainResult:
    theta = jnp.asarray(theta)
    x = jnp.asarray(x)
    theta_dim, x_dim = theta.shape[-1], x.shape[-1]
    num_nodes = theta_dim + x_dim
    data = jnp.hstack([theta, x])[..., None]
    node_id = jnp.arange(num_nodes)

    key_train = training_key(seed)
    key_init, key_loop = jax.random.split(key_train)

    t0 = time.perf_counter()
    init_fn, model_fn = build_model(num_nodes, schedule, model_cfg, nn)
    params = init_fn(
        key_init,
        jnp.ones((10,)),
        data[:10],
        node_id,
        jnp.zeros_like(data[:10]),
        meta_data=None,
    )
    t_setup = time.perf_counter() - t0

    n_val = max(int(train_cfg.validation_fraction * data.shape[0]), 0)
    data_val, data_train = jnp.split(data, [n_val], axis=0)
    data_val = jnp.repeat(data_val, train_cfg.val_repeat, axis=0)

    total_steps = steps_override or train_cfg.total_steps(data.shape[0])
    lr = optax.linear_schedule(
        train_cfg.learning_rate, train_cfg.min_learning_rate,
        total_steps // 2, total_steps // 2,
    )
    optimizer = optax.chain(
        optax.adaptive_grad_clip(train_cfg.clip_max_norm), optax.adam(lr)
    )
    opt_state = optimizer.init(params)

    dsm_loss = nn["dsm_loss"]

    def std_fn(times, x0):
        t = times.reshape((-1,) + (1,) * (x0.ndim - 1))
        return jnp.broadcast_to(schedule.sigma(t), x0.shape)

    def loss_fn(params, key, batch):
        key_times, key_loss, key_cond = jax.random.split(key, 3)
        times = jax.random.uniform(
            key_times, (batch.shape[0],), minval=schedule.t_min, maxval=schedule.t_max
        )
        cmask = condition_mask_fn(key_cond, batch.shape[0], theta_dim, x_dim)
        return dsm_loss(
            params, key_loss, times, batch,
            loss_mask=cmask,
            model_fn=model_fn,
            mean_fn=lambda times, x0: x0,
            std_fn=std_fn,
            weight_fn=lambda t: schedule.weight(t).reshape(-1, 1, 1),
            rebalance_loss=False,
            data_id=node_id,
            condition_mask=cmask,
            meta_data=None,
            edge_mask=None,
        )

    @jax.jit
    def update(params, opt_state, key, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        return loss, optax.apply_updates(params, updates), opt_state

    jit_loss = jax.jit(loss_fn)
    val_every = max(total_steps // train_cfg.val_every, 1)
    print_every = max(total_steps // print_every_frac, 1)

    early_counter, l_train, l_val = 0, None, None
    min_l_val, best_params, history = jnp.inf, None, []
    early_stopped, step = False, 0

    t1 = time.perf_counter()
    t_first = None
    for step in range(total_steps):
        key_loop, k_batch, k_update, k_val = jax.random.split(key_loop, 4)
        idx = jax.random.randint(
            k_batch, (train_cfg.batch_size,), 0, data_train.shape[0]
        )
        loss, params, opt_state = update(params, opt_state, k_update, data_train[idx])
        if step == 0:
            loss.block_until_ready()
            t_first = time.perf_counter() - t1
        l_train = loss if step == 0 else 0.9 * l_train + 0.1 * loss

        if n_val > 0 and (step % val_every) == 0 and step > 50:
            l_val = jit_loss(params, k_val, data_val)
            early_counter = (
                early_counter + 1 if l_val / l_train > train_cfg.val_error_ratio else 0
            )
            if l_val < min_l_val:
                min_l_val = l_val
                best_params = jax.tree_util.tree_map(lambda a: a, params)
            history.append({"step": step, "train": float(l_train),
                            "val": float(l_val), "early_counter": early_counter})

        if early_counter > train_cfg.stop_early_count:
            early_stopped = True
            break

        if (step % print_every) == 0:
            msg = f"  step {step:>6}/{total_steps}  train {float(l_train):.5f}"
            if l_val is not None:
                msg += f"  val {float(l_val):.5f} (ec {early_counter})"
            print(msg, flush=True)

    jax.block_until_ready(params)
    t_train = time.perf_counter() - t1
    out_params = best_params if (early_stopped and best_params is not None) else params
    mean, std = terminal_moments(data, schedule)
    return TrainResult(
        params=out_params, history=history, steps_run=step + 1,
        early_stopped=early_stopped,
        timing={"setup_s": t_setup, "first_step_s": t_first, "train_s": t_train,
                "steps_per_s": (step + 1) / t_train if t_train > 0 else float("nan"),
                "total_steps_planned": total_steps},
        marginal_end_mean=mean, marginal_end_std=std,
    )
