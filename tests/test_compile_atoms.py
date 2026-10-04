"""Prior compilation: the correlated Gaussian as a positive common-bandwidth atom dictionary."""

from __future__ import annotations

import math

import numpy as np

from spectra import compile_atoms


def test_compiled_dictionary_is_a_positive_common_bandwidth_family():
    box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    prior = compile_atoms.design_prior(box)
    for name, sf in compile_atoms.CAPACITIES.items():
        d = compile_atoms.compile_dictionary(prior, spread_factor=sf)
        b = np.exp(d.log_b)
        assert np.all(b > 0), name
        assert math.isclose(float(b.sum()), 1.0, rel_tol=1e-10)
        assert 0.0 < d.lam < float(np.linalg.eigvalsh(prior.cov).min())
        js = d.as_prior_json("two_moons")
        sig = np.asarray(js["sigma"], float)
        assert np.allclose(sig, math.sqrt(d.lam))


def test_compilation_error_is_prior_only_and_improves_with_capacity():
    box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    prior = compile_atoms.design_prior(box)
    small = compile_atoms.approximation_error(
        compile_atoms.compile_dictionary(prior, spread_factor=3.0), num=50_000)
    medium = compile_atoms.approximation_error(
        compile_atoms.compile_dictionary(prior, spread_factor=6.0), num=50_000)
    assert medium["eps_r_over_Z_bound_uniform"] < small["eps_r_over_Z_bound_uniform"]
    assert medium["eps_r_over_Z_bound_uniform"] < 0.05
    assert prior.mass_inside_box() > 0.99


def test_design_prior_is_deterministic():
    box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    a = compile_atoms.design_prior(box)
    b = compile_atoms.design_prior(box)
    assert np.array_equal(a.mu, b.mu) and np.array_equal(a.cov, b.cov)
