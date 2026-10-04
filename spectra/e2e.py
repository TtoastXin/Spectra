"""One reverse VE-SDE driver shared by every method.

Protocol A (the controlled comparison) fixes everything except the target
score: the same range-preserving power time grid, the same terminal draw taken
from the base Simformer's own ``marginal_end_mean/std``, the same Brownian
increments and the same Langevin noise.  A method is then only a hook

    (z, t) -> (s_q(z, t), per-step diagnostics).

Protocol B keeps upstream's choices instead (``grid="upstream_power"``, its own
terminal, its own conditioning semantics); it is for reproducing PriorGuide,
not for comparing scores within one dynamical system.

Time convention
---------------
``tau = sigma(t)^2`` everywhere.  Integration runs on the reversed clock as
upstream's ``sdeint`` does: with ``dt = t_i - t_{i+1} > 0`` and VE's vanishing
forward drift,

    z <- z + dt * sigma(t)^2 * 2 log(sigma_max/sigma_min) * s_q(z, t)
           + sqrt(dt) * sigma(t) * sqrt(2 log(sigma_max/sigma_min)) * eps.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

GRIDS = ("power", "upstream_power", "uniform")

# Trajectory snapshots: fraction of remaining diffusion time.
DEFAULT_SNAPSHOT_FRACS = (0.8, 0.5, 0.2, 0.05)


# Runtime NFE instrumentation.  ``instrument_hook`` adds
# integer per-call counters to a hook's ``aux`` under this prefix; the driver
# sums them over the Langevin corrector calls of every step, so after a run
# ``nfe_from_aux(result.step_aux)`` is the number of hook / operator calls the
# traced program executed, summed inside ``lax.scan`` rather than by a Python
# counter (which would count traces, not executions).  Hooks without these keys
# leave the driver's computational graph unchanged; the timing runs use such
# hooks and rely on ``nfe_accounting``, which the instrumented runs check.
NFE_PREFIX = "nfe_"


@dataclass(frozen=True)
class SamplerConfig:
    """Reverse-driver settings shared by all methods in one comparison cell."""

    num_steps: int = 100
    grid: str = "power"
    grid_rho: float = 2.0
    langevin_steps: int = 0
    langevin_ratio: float = 0.5

    def as_dict(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "grid": self.grid,
            "grid_rho": self.grid_rho,
            "langevin_steps": self.langevin_steps,
            "langevin_ratio": self.langevin_ratio,
        }


def time_grid(schedule, cfg: SamplerConfig) -> jnp.ndarray:
    """Ascending diffusion-time grid; the sampler walks it backwards.

    ``power`` is the main protocol: it keeps the endpoints ``[T_min, T_max]``
    and only redistributes the steps towards low noise.  ``upstream_power`` is
    PriorGuide's ``linspace(T_min, T_max, N) ** rho``, which also moves the
    lower endpoint to ``T_min ** rho``; it is used only for Protocol B.
    """
    t_min, t_max = float(schedule.t_min), float(schedule.t_max)
    n = cfg.num_steps
    if cfg.grid == "power":
        u = jnp.linspace(0.0, 1.0, n)
        return t_min + (t_max - t_min) * u**cfg.grid_rho
    if cfg.grid == "upstream_power":
        return jnp.linspace(t_min, t_max, n) ** cfg.grid_rho
    if cfg.grid == "uniform":
        return jnp.linspace(t_min, t_max, n)
    raise ValueError(f"grid must be one of {GRIDS}, got {cfg.grid!r}")


def snapshot_indices(num_steps: int, fracs: Sequence[float]) -> list[int]:
    """Step counts after which a trajectory snapshot is taken.

    ``frac`` is the fraction of diffusion time still remaining, so ``0.8`` is an
    early (high-noise) snapshot and ``0.05`` a late one.
    """
    total = num_steps - 1
    idx = sorted({int(round((1.0 - f) * total)) for f in fracs})
    return [i for i in idx if 0 < i < total]


@dataclass
class SampleResult:
    samples: np.ndarray  # (num_samples, theta_dim) clean draws
    snapshots: dict  # {frac: {"t": float, "state": (num_samples, theta_dim)}}
    step_aux: dict  # per-step stacked diagnostics from the hook
    times: np.ndarray  # the ascending time grid used
    nfe: dict = field(default_factory=dict)
    wall_clock_s: float = float("nan")


def draw_terminal(key, num_samples: int, mean, std) -> jnp.ndarray:
    """Base Simformer's terminal law, shared by every method in the cell."""
    mean = jnp.asarray(mean)
    std = jnp.asarray(std)
    return mean + std * jax.random.normal(key, (num_samples, mean.shape[-1]))


def make_noise(key, num_steps: int, num_samples: int, theta_dim: int,
               langevin_steps: int):
    """Pre-draw every random increment so methods are exactly paired."""
    k_diff, k_lang = jax.random.split(key)
    diff = jax.random.normal(k_diff, (num_steps - 1, num_samples, theta_dim))
    lang = (
        jax.random.normal(k_lang, (num_steps - 1, langevin_steps, num_samples, theta_dim))
        if langevin_steps > 0
        else jnp.zeros((num_steps - 1, 0, num_samples, theta_dim))
    )
    return diff, lang


# Compiled sampling kernels, keyed by the hook object.  ``lax.scan`` caches the
# traced body on a weak reference to the function object it is given, so a
# ``step`` closure rebuilt on every call misses that cache and XLA recompiles
# each time, which would put compilation inside every timed run.
# ``make_sampler`` builds the
# pure-JAX kernel once per (hook, schedule, cfg, snapshot_fracs) and hands back
# a jitted callable, so compilation happens on the first call and the timed
# repeats execute an already-compiled program.  Host-side conversion and
# synchronisation stay outside the kernel, in :func:`reverse_sample`.
_SAMPLER_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _sampler_key(schedule, cfg: SamplerConfig, snapshot_fracs) -> tuple:
    try:
        sched = hash(schedule)
    except TypeError:  # pragma: no cover - schedules are frozen dataclasses
        sched = id(schedule)
    return (sched, cfg, tuple(snapshot_fracs))


def make_sampler(hook: Callable, *, schedule, cfg: SamplerConfig,
                 snapshot_fracs: Sequence[float] = DEFAULT_SNAPSHOT_FRACS):
    """The compiled reverse-SDE kernel for one ``(hook, schedule, cfg)``.

    Returns ``kernel(x_init, diffusion_noise, langevin_noise) -> (z, snapshots,
    step_aux)`` with everything on device: no NumPy conversion, no
    ``block_until_ready``.  Repeated calls return the same callable holding
    the same scan body, so JAX reuses its traced jaxpr and compiled executable
    instead of recompiling.  ``kernel.reused`` says whether this call hit the
    cache.
    """
    key = _sampler_key(schedule, cfg, snapshot_fracs)
    try:
        by_key = _SAMPLER_CACHE.setdefault(hook, {})
    except TypeError:  # unhashable / non-weakref-able callable: no caching
        by_key = {}
    if key in by_key:
        kernel = by_key[key]
        kernel.reused = True
        return kernel

    log_ratio = schedule.log_ratio
    c = jnp.sqrt(2.0 * log_ratio)
    n = cfg.num_steps
    ts = time_grid(schedule, cfg)
    t_hi_all = ts[::-1][:-1]
    t_lo_all = ts[::-1][1:]
    cuts = snapshot_indices(n, snapshot_fracs)
    bounds = [0, *cuts, n - 1]

    def step(z, payload):
        t_hi, t_lo, eps, lang_eps = payload
        dt = t_hi - t_lo
        sig = schedule.sigma(t_hi)
        diff2 = sig**2 * (2.0 * log_ratio)

        lang_counts = None
        if cfg.langevin_steps > 0:
            # delta = eta g(t_hi)^2 dt / 2, the released PriorGuide corrector;
            # see ``langevin_step_size`` for the paper-vs-code note.  The
            # corrector acts on the current target noisy state at the base
            # time t_hi, so its step size and noise use g(t_hi).  Methods that
            # query the network at a transformed noise level (Spectra's
            # (m, rho)) do so inside the hook to obtain the target score; this
            # is how the score is computed, not a change of the Langevin
            # kernel, so g(rho) is not used here.
            lstep = cfg.langevin_ratio * diff2 * dt / 2.0

            def lang(zz, e):
                s, aux_l = hook(zz, t_hi)
                return zz + lstep * s + jnp.sqrt(2.0 * lstep + 1e-8) * e, _nfe_part(aux_l)

            z, lang_counts = jax.lax.scan(lang, z, lang_eps)

        score, aux = hook(z, t_hi)
        if lang_counts:
            # corrector calls of this step, folded into the step's own counters
            aux = {**aux, **{k: aux[k] + jnp.sum(v, axis=0)
                             for k, v in lang_counts.items()}}
        z = z + dt * diff2 * score + jnp.sqrt(dt) * sig * c * eps
        return z, aux

    def run(x_init, diffusion_noise, langevin_noise):
        """One reverse pass.

        The segment loop stays in Python and each segment between snapshots is
        its own ``lax.scan``; ``step`` is built once per cached kernel.  The
        loop is not wrapped in a single ``jax.jit``: that would let XLA fuse
        across the snapshot boundaries and change float32 results (up to
        1.5e-3 on the Jacobian-using hooks).
        """
        z = x_init
        snaps, aux_parts = [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            payload = (t_hi_all[lo:hi], t_lo_all[lo:hi], diffusion_noise[lo:hi],
                       langevin_noise[lo:hi])
            z, aux = jax.lax.scan(step, z, payload)
            aux_parts.append(aux)
            if hi < n - 1:
                snaps.append(z)
        return z, tuple(snaps), tuple(aux_parts)

    kernel = run
    kernel.reused = False
    kernel.snapshot_bounds = bounds
    kernel.times = ts
    by_key[key] = kernel
    return kernel


def reverse_sample(
    hook: Callable,
    *,
    schedule,
    cfg: SamplerConfig,
    x_init: jnp.ndarray,
    diffusion_noise: jnp.ndarray,
    langevin_noise: jnp.ndarray,
    snapshot_fracs: Sequence[float] = DEFAULT_SNAPSHOT_FRACS,
) -> SampleResult:
    """Integrate the reverse VE-SDE with ``hook`` supplying the target score.

    ``hook(z, t)`` returns ``(score, aux)``; ``aux`` must be a pytree with the
    same structure at every step (scalars are stacked over steps).

    The numerical work runs inside the cached kernel of :func:`make_sampler`;
    this wrapper only pulls the result back to the host.  ``wall_clock_s``
    therefore still contains compilation on a cache miss, which is why the
    timing probes call the kernel directly and separate the two.
    """
    import time as _time

    ts = time_grid(schedule, cfg)
    n = cfg.num_steps
    kernel = make_sampler(hook, schedule=schedule, cfg=cfg,
                          snapshot_fracs=snapshot_fracs)
    bounds = kernel.snapshot_bounds
    t_lo_all = ts[::-1][1:]

    t0 = _time.perf_counter()
    z, snaps, aux_parts = kernel(x_init, diffusion_noise, langevin_noise)
    z.block_until_ready()
    wall = _time.perf_counter() - t0

    snapshots = {}
    for (lo, hi), state in zip(list(zip(bounds[:-1], bounds[1:]))[:len(snaps)], snaps):
        frac = 1.0 - hi / (n - 1)
        snapshots[round(frac, 4)] = {"t": float(t_lo_all[hi - 1]),
                                     "state": np.asarray(jax.device_get(state))}

    step_aux = _concat_aux(aux_parts)
    return SampleResult(
        samples=np.asarray(jax.device_get(z)),
        snapshots=snapshots,
        step_aux=step_aux,
        times=np.asarray(jax.device_get(ts)),
        wall_clock_s=wall,
    )


def _concat_aux(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return {}
    keys = parts[0].keys()
    return {
        k: np.concatenate([np.asarray(jax.device_get(p[k])) for p in parts])
        for k in keys
    }


def langevin_step_size(schedule, t_hi, dt, langevin_ratio: float):
    """The corrector step ``delta`` used by :func:`reverse_sample`.

    ``delta = eta * g(t_hi)^2 * dt / 2`` with ``g(t)^2 = sigma(t)^2 * 2 log(sigma_max
    / sigma_min) = 2 sigma(t) sigma'(t)`` for the VE schedule.  This is the
    update the released PriorGuide code executes
    (``priorg/sim/methods/guidance_gmm.py``: ``langevin_step_size =
    langevin_ratio * diffusion_coeff**2 * |dt| / 2``).  Under it, ``eta = 1``
    makes the Langevin noise variance ``2 delta`` equal to the Euler-Maruyama
    step variance ``g^2 dt``, which is how the PriorGuide paper describes
    ``eta``.  PriorGuide's Eq. (A1) writes ``delta = eta sigma'(t) sigma(t) dt
    / 2``; under the ``g^2 = 2 sigma sigma'`` convention the two differ by a
    factor of two.  This package uses the executable upstream update.

    Kept as a separate function so the tests can evaluate the same expression
    the driver uses; the driver inlines the same arithmetic.
    """
    sig = schedule.sigma(t_hi)
    diff2 = sig**2 * (2.0 * schedule.log_ratio)
    return langevin_ratio * diff2 * dt / 2.0


def instrument_hook(hook: Callable, ops: dict) -> Callable:
    """Wrap ``hook`` so its ``aux`` carries one integer counter per operator.

    ``ops`` is the hook factory's per-call operator count (``{"base_score": 1,
    "denoiser_jacobian": 1}`` for PG-FullCov).  Every call then reports
    ``nfe_hook_calls = 1`` and ``nfe_<op> = ops[op]``; the driver sums them over
    the corrector calls of each step and ``nfe_from_aux`` over the steps.  The
    score returned is unchanged, so an instrumented run produces identical
    samples (checked in ``tests/test_langevin.py``).
    """
    counts = {f"{NFE_PREFIX}hook_calls": 1,
              **{f"{NFE_PREFIX}{k}": int(v) for k, v in ops.items()}}

    def wrapped(z, t):
        s, aux = hook(z, t)
        aux = dict(aux)
        for k, v in counts.items():
            aux[k] = jnp.asarray(v, dtype=jnp.int32)
        return s, aux

    return wrapped


def _nfe_part(aux: dict):
    """The counter entries of one hook ``aux``; ``None`` when uninstrumented.

    Returning ``None`` (not ``{}``) keeps the corrector scan's carried output
    identical to the uninstrumented driver for hooks that carry no counters.
    """
    part = {k: v for k, v in aux.items() if k.startswith(NFE_PREFIX)}
    return part or None


def nfe_from_aux(step_aux: dict) -> dict:
    """Runtime operator totals of one run, from the stacked per-step counters.

    Keys match :func:`nfe_accounting` (``hook_calls``, ``<op>_calls``) so the
    two can be compared directly; empty when the hook was not instrumented.
    """
    out = {}
    for k, v in step_aux.items():
        if k.startswith(NFE_PREFIX):
            name = k[len(NFE_PREFIX):]
            name = name if name == "hook_calls" else f"{name}_calls"
            out[name] = int(np.sum(np.asarray(v)))
    return out


def nfe_accounting(cfg: SamplerConfig, per_hook: dict) -> dict:
    """Turn per-hook operator counts into per-trajectory totals.

    ``per_hook`` names how many of each operator one ``hook`` call issues, e.g.
    ``{"base_score": 1}`` for exact transport or ``{"base_score": 1,
    "denoiser_jacobian": 1}`` for PG-FullCov.  Score-forward counts alone do
    not describe PG-FullCov, which is why every key is reported separately.
    """
    hooks = (cfg.num_steps - 1) * (1 + cfg.langevin_steps)
    out = {"hook_calls": hooks}
    out.update({f"{k}_calls": int(v) * hooks for k, v in per_hook.items()})
    return out
