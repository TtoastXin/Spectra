"""Steady-state timing harness and jaxpr call-chain statistics.

``CompileCounter`` counts XLA compilation events from JAX's own loggers;
``timed_kernel`` compiles a kernel once, then times repeated executions and
marks the measurement invalid if any timed repeat compiled again.
``jaxpr_stats`` summarises a traced hook, in particular the network
activations that carry an extra ``theta_dim`` axis (the signature of a full
``vmap(jacrev)`` Jacobian, absent from a single vector--Jacobian product).
"""

from __future__ import annotations

import logging
import statistics as st
import time
from collections import Counter

import jax
import numpy as np


class CompileCounter(logging.Handler):
    """Counts XLA compilation events emitted by JAX's own loggers."""

    NAMES = ("jax", "jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.compiler")

    def __init__(self):
        super().__init__()
        self.n = 0

    def emit(self, record):
        msg = record.getMessage()
        if "Compiling" in msg or "Finished XLA compilation" in msg:
            self.n += 1

    def install(self):
        jax.config.update("jax_log_compiles", True)
        for name in self.NAMES:
            lg = logging.getLogger(name)
            lg.addHandler(self)
            lg.setLevel(logging.DEBUG)
        return self


def timed_kernel(kernel, args, repeats: int, counter: "CompileCounter") -> dict:
    """Compile once, then time ``repeats`` executions of the compiled program.

    ``compile_s`` is the first call (tracing + XLA compilation + one execution);
    the timed repeats follow it, and ``timing_valid`` records that none of them
    triggered a new compilation.  Raw repeats are kept.
    """
    n0 = counter.n
    t0 = time.perf_counter()
    out = kernel(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    n_compile_first = counter.n - n0

    n1 = counter.n
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = kernel(*args)
        jax.block_until_ready(out)
        walls.append(time.perf_counter() - t0)
    n_compile_timed = counter.n - n1
    walls.sort()
    q1, q3 = float(np.percentile(walls, 25)), float(np.percentile(walls, 75))
    return {"median_s": float(st.median(walls)), "q25_s": q1, "q75_s": q3, "iqr_s": q3 - q1,
            "min_s": walls[0], "max_s": walls[-1], "repeats": repeats, "all_s": walls,
            "compile_s": compile_s, "compile_events_first_call": n_compile_first,
            "compile_events_during_timing": n_compile_timed,
            # a repeat that recompiles is not a steady-state measurement
            "timing_valid": n_compile_timed == 0}


def _sub_jaxprs(eqn):
    for p in eqn.params.values():
        items = p if isinstance(p, (tuple, list)) else (p,)
        for item in items:
            if isinstance(item, jax.core.ClosedJaxpr):
                yield item.jaxpr
            elif isinstance(item, jax.core.Jaxpr):
                yield item


def is_d_batched(shape, batch: int, dim: int, nodes: int) -> bool:
    """A network activation (sample and node axes) that also carries a ``theta_dim`` axis.

    ``vmap(jacrev(.))`` evaluates the backward pass once per basis cotangent, so
    those activations gain a ``theta_dim`` axis next to the sample axis; a
    single pullback has none.  Requiring the node axis keeps the ``(B, K, d, d)``
    Gaussian-product algebra out.  The two controls (``pg_vjp`` and ``spectra``)
    show whether anything else in the graph still matches.
    """
    return (len(shape) >= 4 and shape[0] == batch and dim in shape[1:]
            and nodes in shape[1:] and shape.index(dim, 1) != shape.index(nodes, 1))


def jaxpr_stats(closed, batch: int, dim: int, nodes: int) -> dict:
    stats = {"n_eqns": 0, "sum_output_elements": 0, "n_d_batched": 0,
             "d_batched_output_elements": 0, "n_jacobian_shaped": 0}
    top = Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            stats["n_eqns"] += 1
            for v in eqn.outvars:
                shape = tuple(getattr(v.aval, "shape", ()))
                n = int(np.prod(shape)) if shape else 1
                stats["sum_output_elements"] += n
                if is_d_batched(shape, batch, dim, nodes):
                    stats["n_d_batched"] += 1
                    stats["d_batched_output_elements"] += n
                if shape == (batch, dim, dim):
                    stats["n_jacobian_shaped"] += 1
                top[(eqn.primitive.name, shape)] += 1
            for sub in _sub_jaxprs(eqn):
                walk(sub)

    walk(closed.jaxpr)
    biggest = sorted(top.items(), key=lambda kv: -int(np.prod(kv[0][1])) if kv[0][1] else 0)[:12]
    stats["largest_outputs"] = [{"primitive": p, "shape": list(s), "count": c}
                                for (p, s), c in biggest]
    return stats
