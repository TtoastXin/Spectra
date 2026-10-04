"""SLCP: the exact likelihood, its sign-flip symmetry and the tempered SMC reference."""

from __future__ import annotations

import numpy as np
import pytest

from spectra import slcp


def test_slcp_likelihood_matches_a_brute_force_loop():
    rng = np.random.default_rng(0)
    y = rng.uniform(-1, 1, size=(6, 5))
    x_raw = slcp.simulate(y[:1], rng)[0]
    x = slcp.standardise(x_raw)
    fast = slcp.log_likelihood(y, x)
    phys = slcp.to_physical(y)
    S = slcp.covariance(phys)
    pts = slcp.unstandardise(x).reshape(slcp.NUM_DATA, 2)
    slow = []
    for i in range(y.shape[0]):
        cov = S[i]
        inv = np.linalg.inv(cov)
        _, logdet = np.linalg.slogdet(cov)
        s = 0.0
        for pt in pts:
            r = pt - phys[i, :2]
            s += r @ inv @ r
        slow.append(-0.5 * s - 0.5 * slcp.NUM_DATA * logdet)
    assert np.allclose(fast, np.array(slow), atol=1e-8)


def test_slcp_likelihood_is_exactly_sign_flip_invariant():
    """The property the mode-jumping move exploits; it must be exact, not close."""
    rng = np.random.default_rng(1)
    y = rng.uniform(-1, 1, size=(64, 5))
    x = slcp.standardise(slcp.simulate(y[:1], rng)[0])
    base = slcp.log_likelihood(y, x)
    for flip in slcp.SIGN_FLIPS:
        assert np.array_equal(slcp.log_likelihood(y * np.asarray(flip), x), base)


def test_sign_flip_move_is_per_particle():
    """Independent per-particle flips equilibrate a sign imbalance.

    A single flip drawn for the whole ensemble is accepted by every particle
    under the symmetric prior, so the ensemble reflects as a block and the
    imbalance is preserved exactly.  With per-particle flips the occupancy is a
    symmetric two-state chain and relaxes to 1/2, which is what is asserted --
    after one sweep it is 1/3, because two of the three flips move theta_3.
    """
    rng = np.random.default_rng(2)
    theta = np.abs(rng.standard_normal((4000, 5))) * 0.3  # all positive
    ll = np.zeros(4000)
    box = slcp.box_log_prior()
    out, _, acc = slcp.sign_flip_move(theta, ll, 1.0, rng, box)
    assert acc > 0.99  # symmetric prior accepts everything
    assert 0.30 < float((out[:, 2] > 0).mean()) < 0.37  # one sweep -> 1/3
    for _ in range(20):
        out, ll, _ = slcp.sign_flip_move(out, ll, 1.0, rng, box)
    frac = float((out[:, 2] > 0).mean())
    assert 0.45 < frac < 0.55, frac


def test_tempered_smc_reaches_beta_one_on_a_gaussian_target():
    rng = np.random.default_rng(3)
    mu = np.array([0.3, -0.2, 0.1, 0.0, -0.4])

    def lik(theta):
        return -0.5 * np.sum((np.atleast_2d(theta) - mu) ** 2, axis=-1) / 0.01

    res = slcp.tempered_smc(lik, slcp.box_log_prior(),
                            lambda n: rng.uniform(-1, 1, size=(n, 5)),
                            num_particles=2000, rng=rng, num_mcmc=6)
    assert res.diagnostics["final_beta"] == pytest.approx(1.0)
    assert np.allclose(res.samples.mean(axis=0), mu, atol=0.05)
