"""The PSD convention shared by every method that needs a covariance.

``J_hat = I + tau grad_z s_hat`` is neither symmetric nor guaranteed PSD,
because a learned score field is not exactly conservative.  PriorGuide uses the
raw ``J_hat^T`` (its closure does not treat it as a covariance), while
PG-FullCov needs ``C_T_hat = tau * PSD(sym(J_hat))``.  The projection uses a
fixed eigenvalue floor.  Points where it is active are kept rather than
dropped, since dropping them would bias the comparison; the asymmetry, the
non-PSD rate and the projection size are reported per step.
"""

from __future__ import annotations

import jax.numpy as jnp

PSD_EPS = 1e-6


# ------------------------------------------------------------ learned side


def psd_project(jac, eps: float = PSD_EPS):
    """``sym(J)`` and its PSD projection with a fixed eigenvalue floor."""
    j_sym = 0.5 * (jac + jnp.swapaxes(jac, -1, -2))
    w, u = jnp.linalg.eigh(j_sym)
    w_clip = jnp.maximum(w, eps)
    j_psd = jnp.einsum("nij,nj,nkj->nik", u, w_clip, u)
    return j_sym, j_psd, w, w_clip
