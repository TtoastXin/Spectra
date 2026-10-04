"""Exact-transport unit tests.

Everything here runs in float64 numpy: :mod:`spectra.transport` is
plain arithmetic on whatever array type it is handed, so the identities can be
checked at machine precision rather than at JAX's default float32.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectra.transport import (
    e2_params,
    e2_target_score,
    guard_ok,
    t0_target_score,
    tq_pieces,
    tq_target_score,
)

DIMS = (2, 10, 20)
TAUS = (1e-6, 1e-3, 0.05, 0.8, 5.0, 225.0)


def _analytic_gaussian_base(m0, s0_sq):
    """``p_0 = N(m0, s0^2 I)``, so ``s_p(z, tau) = -(z - m0) / (s0^2 + tau)``."""

    def score(z, tau):
        return -(z - m0) / (s0_sq + tau)

    return score


def _tilted_posterior_score(m0, s0_sq, a, kappa):
    """Exact target score after ``r ∝ exp(a^T th - kappa/2 ||th||^2)`` on that base."""
    var_q = 1.0 / (1.0 / s0_sq + kappa)
    m_q = var_q * (m0 / s0_sq + a)

    def score(z, tau):
        return -(z - m_q) / (var_q + tau)

    return score


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("tau", TAUS)
def test_kappa_zero_reduces_to_t0(d, tau):
    rng = np.random.default_rng(d)
    a = rng.normal(size=d)
    z = rng.normal(size=(7, d))
    score = _analytic_gaussian_base(rng.normal(size=d), 0.3)
    got, _ = tq_target_score(score, z, tau, a, 0.0)
    want = t0_target_score(score, z, tau, a)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("tau", TAUS)
def test_isotropic_localisation_reduces_to_e2(d, tau):
    rng = np.random.default_rng(100 + d)
    mu = rng.normal(size=d)
    lam = float(np.exp(rng.normal()))
    a, kappa = e2_params(mu, lam)
    z = rng.normal(size=(7, d))
    score = _analytic_gaussian_base(rng.normal(size=d), 0.7)
    got, q = tq_target_score(score, z, tau, a, kappa)
    want = e2_target_score(score, z, tau, mu, lam)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(q.rho, tau * lam / (tau + lam), rtol=1e-13)


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("tau", TAUS)
def test_general_a_and_kappa_are_exact_on_a_gaussian_base(d, tau):
    """The case the single-factor benchmark never covers: both parameters non-zero."""
    rng = np.random.default_rng(200 + d)
    m0 = rng.normal(size=d)
    s0_sq = 0.4
    a = rng.normal(size=d)
    kappa = float(abs(rng.normal()) + 0.1)
    z = m0 + rng.normal(size=(11, d))
    base = _analytic_gaussian_base(m0, s0_sq)
    exact = _tilted_posterior_score(m0, s0_sq, a, kappa)
    got, _ = tq_target_score(base, z, tau, a, kappa)
    np.testing.assert_allclose(got, exact(z, tau), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("d", DIMS)
def test_pieces_satisfy_their_definitions(d):
    rng = np.random.default_rng(300 + d)
    a = rng.normal(size=d)
    kappa, tau = 3.5, 0.2
    z = rng.normal(size=(5, d))
    q = tq_pieces(z, tau, a, kappa)
    assert np.allclose(q.D, 1 + kappa * tau)
    assert np.allclose(q.rho, tau / q.D)
    assert np.allclose(q.m, (z + tau * a) / q.D)
    # rho < tau whenever kappa > 0: the transformed query is always *less* noisy
    assert float(q.rho) < tau


def test_guard_rejects_a_too_negative_kappa():
    assert guard_ok(0.5, 1.0)
    assert guard_ok(0.5, -1.0)  # 1 - 0.5 > 0
    assert not guard_ok(2.0, -1.0)  # 1 - 2 < 0
