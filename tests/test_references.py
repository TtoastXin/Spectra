"""Reference posteriors are experimental objects and get their own tests.

The Gaussian Linear reference is checked against a self-normalised importance
sampler built straight from the task definition (prior proposal, Gaussian
likelihood), which shares no code with the conjugate update.  The Two Moons
reference is checked for a valid rejection bound and against the same kind of
independent importance sampler.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectra import benchmark
from spectra.prior_shift import build_prior_shift
from spectra.references import (
    GL_SIMULATOR_SCALE,
    _two_moons_likelihood,
    gaussian_linear_posterior,
    support_matched,
    two_moons_likelihood_bound,
    two_moons_reference,
)


def _importance_moments(theta, log_w):
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    mean = w @ theta
    dev = theta - mean
    cov = (dev * w[:, None]).T @ dev
    ess = 1.0 / np.sum(w**2)
    return mean, cov, ess


@pytest.mark.parametrize("prior_type", ("mild", "strong", "mixture"))
def test_gaussian_linear_posterior_matches_importance_sampling(prior_type):
    shift = build_prior_shift("gaussian_linear", prior_type, 0)
    _, x_o = benchmark.load_observation("gaussian_linear", prior_type, 0, 1000000)
    x_o = np.asarray(x_o, float)

    post = gaussian_linear_posterior(x_o, shift)
    rng = np.random.default_rng(0)
    n = 400_000
    k = rng.choice(len(shift.target_pi), size=n, p=shift.target_pi)
    theta = shift.target_mu[k] + shift.target_sigma[k] * rng.standard_normal(
        (n, shift.theta_dim)
    )
    log_lik = -0.5 * np.sum((x_o - theta) ** 2, axis=-1) / GL_SIMULATOR_SCALE
    mean_is, cov_is, ess = _importance_moments(theta, log_lik)

    assert ess > 1000, f"importance check too noisy (ESS={ess:.0f})"
    tol = 6.0 * np.sqrt(np.diag(post.cov) / ess)
    assert np.all(np.abs(post.mean - mean_is) < tol + 1e-3), (
        f"{prior_type}: analytic mean {post.mean} vs IS {mean_is}"
    )
    np.testing.assert_allclose(np.diag(post.cov), np.diag(cov_is), rtol=0.15, atol=2e-3)


def test_gaussian_linear_flat_prior_limit():
    """A very wide test prior must drive the posterior to ``N(x, sigma_sim I)``."""
    shift = build_prior_shift("gaussian_linear", "mild", 0)
    wide = type(shift)(
        **{**shift.__dict__,
           "target_sigma": np.full_like(shift.target_sigma, 1e4),
           "target_mu": np.zeros_like(shift.target_mu)}
    )
    _, x_o = benchmark.load_observation("gaussian_linear", "mild", 0, 1000000)
    post = gaussian_linear_posterior(np.asarray(x_o, float), wide)
    np.testing.assert_allclose(post.means[0], np.asarray(x_o, float), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(post.variances[0], GL_SIMULATOR_SCALE, rtol=1e-6)


def test_gaussian_linear_mixture_weights_are_a_distribution():
    shift = build_prior_shift("gaussian_linear", "mixture", 3)
    _, x_o = benchmark.load_observation("gaussian_linear", "mixture", 3, 1000000)
    post = gaussian_linear_posterior(np.asarray(x_o, float), shift)
    assert post.weights.shape == (2,)
    assert math.isclose(float(post.weights.sum()), 1.0, rel_tol=1e-12)
    assert np.all(post.weights >= 0)


def test_two_moons_bound_is_an_actual_upper_bound():
    exact, upstream = two_moons_likelihood_bound()
    assert exact > upstream, "the exact bound must exceed upstream's f(mu)/mu"
    grid = np.linspace(1e-4, 0.5, 2_000_00)
    const = (1.0 / math.pi) * (1.0 / (math.sqrt(2 * math.pi) * 0.01))
    dens = const * np.exp(-((grid - 0.1) ** 2) / (2 * 0.01**2)) / grid
    assert dens.max() <= exact * (1 + 1e-9)
    assert dens.max() > upstream, "upstream's value is exceeded by the true density"


@pytest.mark.parametrize("prior_type", ("mild", "strong", "mixture"))
def test_two_moons_reference_matches_importance_sampling(prior_type):
    shift = build_prior_shift("two_moons", prior_type, 0)
    _, x_o = benchmark.load_observation("two_moons", prior_type, 0, 1000000)
    x_o = np.asarray(x_o, float)

    ref = two_moons_reference(x_o, shift, num_samples=4000, seed=0, batch=200_000)
    assert ref.max_acceptance_ratio <= 1.0, "acceptance ratio exceeded 1: bad bound"
    assert 0.0 < ref.acceptance_rate < 1.0

    rng = np.random.default_rng(1)
    n = 2_000_000
    k = rng.choice(len(shift.target_pi), size=n, p=shift.target_pi)
    theta = shift.target_mu[k] + shift.target_sigma[k] * rng.standard_normal((n, 2))
    lik = _two_moons_likelihood(theta, x_o)
    keep = lik > 0
    mean_is, cov_is, ess = _importance_moments(theta[keep], np.log(lik[keep]))

    mean_ref = ref.samples.mean(0)
    se = np.sqrt(np.diag(np.cov(ref.samples.T)) / ref.samples.shape[0])
    se_is = np.sqrt(np.diag(cov_is) / ess)
    assert np.all(np.abs(mean_ref - mean_is) < 5 * (se + se_is) + 1e-3), (
        f"{prior_type}: rejection mean {mean_ref} vs IS mean {mean_is}"
    )


def test_two_moons_reference_records_the_support_caveat():
    shift = build_prior_shift("two_moons", "strong", 0)
    _, x_o = benchmark.load_observation("two_moons", "strong", 0, 1000000)
    ref = two_moons_reference(np.asarray(x_o, float), shift, 2000, seed=0)
    d = ref.diagnostics
    assert 0.0 <= d["posterior_mass_outside_training_box"] < 1.0
    assert d["prior_mass_outside_training_box"] > 0.0
    inside = support_matched(ref.samples, shift.train_box)
    assert inside.shape[0] <= ref.samples.shape[0]
    assert np.all(inside >= shift.train_box[0]) and np.all(inside <= shift.train_box[1])


@pytest.mark.parametrize("prior_type", ("mild", "strong", "mixture"))
def test_training_prior_base_posterior_is_the_backbone_target(prior_type):
    """``N(x/2, 0.05 I)`` for Gaussian Linear: prior 0.1 I, likelihood 0.1 I.

    This is the reference base Simformer should be judged against; scoring it
    against the shifted target only measures a prior it was never given.
    """
    from spectra.references import gaussian_linear_base_posterior

    shift = build_prior_shift("gaussian_linear", prior_type, 0)
    _, x_o = benchmark.load_observation("gaussian_linear", prior_type, 0, 1000000)
    x_o = np.asarray(x_o, float)
    post = gaussian_linear_base_posterior(x_o, shift)
    np.testing.assert_allclose(post.means[0], x_o / 2.0, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(post.variances[0], 0.05, rtol=1e-10)
    assert post.weights.shape == (1,)

    rng = np.random.default_rng(0)
    n = 200_000
    theta = np.sqrt(0.1) * rng.standard_normal((n, shift.theta_dim))
    log_lik = -0.5 * np.sum((x_o - theta) ** 2, axis=-1) / GL_SIMULATOR_SCALE
    mean_is, _, ess = _importance_moments(theta, log_lik)
    assert np.all(np.abs(post.mean - mean_is) < 6 * np.sqrt(0.05 / ess) + 1e-3)


def test_two_moons_mode_mass_is_an_exact_split():
    """The two crescents are separated by the sign of ``theta_1 + theta_2``."""
    from spectra.references import two_moons_mode_mass

    assert two_moons_mode_mass(np.array([[1.0, 1.0], [-1.0, -1.0]])) == 0.5
    assert two_moons_mode_mass(np.array([[0.3, 0.2]])) == 1.0
    assert two_moons_mode_mass(np.array([[-0.3, 0.2]])) == 0.0


@pytest.mark.parametrize("prior_type", ("mild", "strong", "mixture"))
def test_two_moons_training_prior_reference_is_balanced_and_bimodal(prior_type):
    """The uniform-box proposal must recover both modes with equal mass.

    Under the training prior the two solutions are exchanged by a symmetry of the
    forward map, so their masses have to agree; the shifted target priors break
    that symmetry, so the base row must not be scored against them.
    """
    from spectra.references import two_moons_mode_mass

    shift = build_prior_shift("two_moons", prior_type, 0)
    _, x_o = benchmark.load_observation("two_moons", prior_type, 0, 1000000)
    ref = two_moons_reference(np.asarray(x_o, float), shift, 3000, seed=0,
                              proposal="training")
    assert ref.max_acceptance_ratio <= 1.0
    assert ref.diagnostics["proposal"] == "training"
    mass = two_moons_mode_mass(ref.samples)
    se = math.sqrt(0.25 / ref.samples.shape[0])
    assert abs(mass - 0.5) < 5 * se, f"{prior_type}: mode mass {mass:.4f}"
