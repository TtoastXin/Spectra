"""The exact ratio representations must reproduce ``log pi_new - log pi_tr``.

If the exact Gaussian components or the exponential--quadratic parameters
differed from the log ratio by anything other than one global constant, every
PriorGuide, PG-FullCov and TQ result would inherit the error.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectra import benchmark
from spectra.prior_shift import build_prior_shift

TASKS = ("gaussian_linear", "gaussian_linear_high", "two_moons", "oup", "slcp", "bav")
PRIOR_IDS = (0, 3, 7)


def _points(shift, rng, num=64):
    """Random points on the scale of the ratio, inside a uniform box if there is one."""
    if shift.train_box is not None:
        low, high = shift.train_box
        return rng.uniform(low, high, size=(num, shift.theta_dim))
    scale = float(np.max(shift.train_sigma))
    return shift.train_mu + 2.0 * scale * rng.standard_normal((num, shift.theta_dim))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("prior_type", benchmark.PRIOR_TYPES)
@pytest.mark.parametrize("prior_id", PRIOR_IDS)
def test_components_match_log_ratio_up_to_a_constant(task, prior_type, prior_id):
    shift = build_prior_shift(task, prior_type, prior_id)
    rng = np.random.default_rng(0)
    theta = _points(shift, rng)
    diff = shift.log_components(theta) - shift.log_r(theta)
    assert np.max(np.abs(diff - diff.mean())) < 1e-8, (
        f"{task}/{prior_type}_{prior_id}: expansion is not a constant offset, "
        f"spread {np.ptp(diff):.3e}"
    )


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("prior_type", ("mild", "strong"))
@pytest.mark.parametrize("prior_id", PRIOR_IDS)
def test_expquad_matches_log_ratio_up_to_a_constant(task, prior_type, prior_id):
    shift = build_prior_shift(task, prior_type, prior_id)
    assert shift.expquad is not None
    rng = np.random.default_rng(1)
    theta = _points(shift, rng)
    diff = np.asarray(shift.expquad.log_r_unnorm(theta)) - shift.log_r(theta)
    assert np.max(np.abs(diff - diff.mean())) < 1e-8, (
        f"{task}/{prior_type}_{prior_id}: exp-quad params are not a constant offset, "
        f"spread {np.ptp(diff):.3e}"
    )


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("prior_type", ("mild", "strong"))
def test_transport_guard_holds_on_the_whole_ve_range(task, prior_type):
    """``1 + kappa tau > 0`` must hold up to ``sigma_max^2``; kappa > 0 makes it free."""
    shift = build_prior_shift(task, prior_type, 0)
    assert shift.expquad.kappa > 0
    assert shift.expquad.guard(15.0**2)


def test_mixture_has_two_positive_components():
    shift = build_prior_shift("gaussian_linear", "mixture", 0)
    assert shift.is_mixture
    assert shift.components.log_weights.shape == (2,)
    assert shift.expquad is None


@pytest.mark.parametrize("task", ("two_moons", "oup", "slcp"))
@pytest.mark.parametrize("prior_type", benchmark.PRIOR_TYPES)
def test_uniform_tasks_have_small_but_nonzero_outside_box_mass(task, prior_type):
    """Upstream keeps the mean >= 3 sigma inside the box: small, nonzero outside mass."""
    shift = build_prior_shift(task, prior_type, 0)
    p_out = shift.target_mass_outside_training_box()
    assert p_out is not None
    assert 0.0 < p_out < 0.05


def test_gaussian_tasks_have_no_box():
    shift = build_prior_shift("gaussian_linear", "strong", 0)
    assert shift.target_mass_outside_training_box() is None


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("prior_type", benchmark.PRIOR_TYPES)
def test_jnp_view_matches_the_exact_float64_parameters(task, prior_type):
    """The arrays the sampler queries are the float32 image of the exact ones."""
    shift = build_prior_shift(task, prior_type, 0)
    comps = shift.components
    for got, want in (
        (comps.log_weights, shift.comp_log_weights),
        (comps.means, shift.comp_means),
        (comps.covs, shift.comp_covs),
    ):
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6, atol=1e-6)
    if shift.expquad is not None:
        np.testing.assert_allclose(
            np.asarray(shift.expquad.a_jnp), shift.expquad.a, rtol=1e-6, atol=1e-6
        )
