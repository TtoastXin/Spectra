"""The exact-transport guard must describe the interval the sampler will use.

``D = 1 + kappa tau`` is affine in ``tau``, so its minimum over the sampler's
noise interval is at ``t_min`` for a positive ``kappa`` and at ``t_max`` for a
negative one.  Reporting the ``t_min`` value unconditionally calls an interval
safe precisely where the identity breaks down, and the hook then runs and
returns NaN instead of refusing.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from spectra import hooks
from spectra.transport import guard_ok, min_guard_D, require_guard


class _Schedule:
    """sigma(t) = sigma_min (sigma_max/sigma_min)^t, as the VE schedule is."""

    sigma_min, sigma_max = 1e-2, 15.0

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** np.asarray(t, float)


SCH = _Schedule()
RANGE = (0.0, 1.0)
TAU_MIN = float(SCH.sigma(RANGE[0]) ** 2)
TAU_MAX = float(SCH.sigma(RANGE[1]) ** 2)


def test_positive_kappa_minimum_is_at_t_min():
    kappa = 2.0
    assert min_guard_D(SCH, RANGE, kappa) == pytest.approx(1.0 + kappa * TAU_MIN)


def test_negative_kappa_minimum_is_at_t_max():
    """For a negative ``kappa`` the minimum is at ``t_max``, not ``t_min``."""
    kappa = -1e-4
    got = min_guard_D(SCH, RANGE, kappa)
    assert got == pytest.approx(1.0 + kappa * TAU_MAX)
    assert got < 1.0 + kappa * TAU_MIN


def test_zero_kappa_is_one_everywhere():
    assert min_guard_D(SCH, RANGE, 0.0) == pytest.approx(1.0)


def test_min_over_several_atoms():
    kappa = np.array([5.0, 0.5, -1e-5])
    assert min_guard_D(SCH, RANGE, kappa) == pytest.approx(1.0 + (-1e-5) * TAU_MAX)


@pytest.mark.parametrize("kappa", [3.0, 0.0, -1e-6])
def test_valid_intervals_are_accepted(kappa):
    d = require_guard(SCH, RANGE, kappa, "test")
    assert d > 0.0
    assert guard_ok(np.array([TAU_MIN, TAU_MAX]), kappa)


@pytest.mark.parametrize("kappa", [-1.0 / TAU_MAX, -0.01, -1.0])
def test_invalid_intervals_are_refused(kappa):
    """D <= 0 anywhere in the interval: rho and t_rho are undefined there."""
    with pytest.raises(ValueError, match="guard"):
        require_guard(SCH, RANGE, kappa, "test")


@pytest.fixture(scope="module")
def oracle():
    """The analytic stand-in backbone; no checkpoint needed."""
    from spectra.backbone import VESchedule
    from spectra.oracles import GaussianOracleBackbone, single_gaussian

    d = 2
    base = single_gaussian(jnp.zeros(d), jnp.eye(d))
    return GaussianOracleBackbone(base, VESchedule())


def test_the_hook_refuses_an_invalid_kappa(oracle):
    x_o = jnp.zeros(1)
    a = jnp.zeros(oracle.theta_dim)
    _, _, meta = hooks.make_tq_hook(oracle, x_o, a, 0.5)
    assert meta["guard_min_D"] > 0.0
    with pytest.raises(ValueError, match="guard"):
        hooks.make_tq_hook(oracle, x_o, a, -0.01)


def test_the_hook_reports_the_true_minimum(oracle):
    t_min, t_max = oracle.nominal_t_range()
    kappa = -1e-7
    _, _, meta = hooks.make_tq_hook(oracle, jnp.zeros(1),
                                    jnp.zeros(oracle.theta_dim), kappa)
    at_t_max = 1.0 + kappa * float(oracle.schedule.sigma(t_max) ** 2)
    at_t_min = 1.0 + kappa * float(oracle.schedule.sigma(t_min) ** 2)
    assert meta["guard_min_D"] == pytest.approx(at_t_max)
    assert meta["guard_min_D"] < at_t_min
