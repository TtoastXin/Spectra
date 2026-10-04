"""Working coordinates: the ratio identity, isotropy and the affine maps."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectra import coordinates
from spectra.prior_shift import build_prior_shift


ALL_CELLS = [("gaussian_linear", 5, 1000002),
             ("gaussian_linear_high", 4, 1000000),
             ("two_moons", 2, 1000003), ("oup", 7, 1000009)]


@pytest.mark.parametrize("task,pid,_obs", ALL_CELLS)
def test_ratio_identity_holds_on_the_training_support(task, pid, _obs):
    shift = build_prior_shift(task, "mixture", pid)
    res = coordinates.ratio_identity_check(shift, num=2048, seed=0)
    assert res["status"] == "ok"
    assert res["native_exact"], res
    assert res["max_abs_dev"] < 1e-8


def test_isotropy_is_a_coordinate_dependent_property():
    """OUP rescales its two axes by 1 and 2, so isotropy is not transform-free.

    The shipped priors are isotropic atoms only because they are written in
    model coordinates; a physical-space isotropic Gaussian would be an
    anisotropic model-space atom with no exact single-query transport.
    """
    assert coordinates.get_coordinates("two_moons").isotropy_preserved
    assert coordinates.get_coordinates("slcp").isotropy_preserved
    assert not coordinates.get_coordinates("oup").isotropy_preserved
    assert not coordinates.get_coordinates("bav").isotropy_preserved


def test_unknown_task_has_no_coordinate_entry():
    with pytest.raises(ValueError):
        coordinates.get_coordinates("not_a_task")


def test_affine_transform_round_trips():
    c = coordinates.get_coordinates("oup")
    y = np.array([[0.3, -0.7], [-1.0, 1.0]])
    assert np.allclose(c.to_model(c.to_physical(y)), y)
    assert math.isclose(c.log_jacobian(), math.log(2.0))
