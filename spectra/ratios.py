"""Test-time prior ratios ``r(theta) = pi_new(theta) / pi_train(theta)``.

``GaussianComponents`` is the exact generalized-Gaussian expansion of ``r``: the
object PriorGuide's closure and PG-FullCov consume, carrying ``log_weights``,
``means`` and ``covs``.  It exposes ``grad_log_r``, the exact ratio score.

The expansions themselves are built in :mod:`spectra.prior_shift`, which reads
the shipped prior JSONs; only ratio families whose exact target is available in
closed form are represented.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GaussianComponents:
    """Exact expansion ``r(z) ∝ sum_k w_k N(z; mu_k, Sigma_k)``."""

    log_weights: jnp.ndarray  # (K,)
    means: jnp.ndarray  # (K, d)
    covs: jnp.ndarray  # (K, d, d)

    def grad_log_r(self, z: jnp.ndarray) -> jnp.ndarray:
        """``grad_z log r(z)`` for a batch ``z`` of shape ``(B, d)``.

        The expansion reproduces ``log r`` only up to a global constant, which
        the gradient removes, so this is the exact clean-time boundary
        ``g_0 = grad log r`` for every shipped test prior (single-component and
        mixture, with a Gaussian or a uniform training prior).

        The normalising ``-d/2 log(2 pi)`` is dropped because it is the same for
        every component and cancels in the responsibility softmax; ``logdet``
        is not, because the components need not share a covariance.
        """
        prec = jnp.linalg.inv(self.covs)  # (K, d, d)
        _, logdet = jnp.linalg.slogdet(self.covs)  # (K,)
        diff = z[:, None, :] - self.means[None, :, :]  # (B, K, d)
        quad = jnp.einsum("bkd,kde,bke->bk", diff, prec, diff)
        log_c = self.log_weights[None, :] - 0.5 * quad - 0.5 * logdet[None, :]
        rho = jax.nn.softmax(log_c, axis=-1)  # (B, K)
        grads = -jnp.einsum("kde,bke->bkd", prec, diff)  # (B, K, d)
        return jnp.einsum("bk,bkd->bd", rho, grads)


