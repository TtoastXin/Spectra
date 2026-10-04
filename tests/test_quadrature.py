"""Grid quadrature references: atom masses, the OUP likelihood, sample responsibilities."""

from __future__ import annotations

import numpy as np
import pytest

from spectra import atoms as atoms_mod
from spectra import benchmark, coordinates, quadrature
from spectra.prior_shift import build_prior_shift


GRID_CELLS = [("two_moons", 2, 1000003), ("oup", 7, 1000009)]


@pytest.mark.parametrize("task,pid,obs", GRID_CELLS)
def test_two_routes_to_the_atom_masses_agree(task, pid, obs):
    """``pi_k propto b_k E_{p_0}[phi_k]`` and ``pi_k = E_q[gamma_k]`` are the
    same number computed two ways; disagreement means the bookkeeping is wrong.
    """
    shift = build_prior_shift(task, "mixture", pid)
    atoms = atoms_mod.atoms_from_shift(shift)
    _, x_o = benchmark.load_observation(task, "mixture", pid, obs)
    x_o = np.asarray(x_o, float)
    box = quadrature.default_box(task, shift, pad_sd=6.0)
    n = 512
    centers, area = quadrature._grid(box, n)
    ll = quadrature.log_likelihood(task, centers, x_o)
    base = quadrature.grid_posterior(task, x_o, shift, kind="base", box=box, n=n,
                                     log_lik=ll, centers=centers, cell_area=area)
    tgt = quadrature.grid_posterior(task, x_o, shift, kind="target_restricted",
                                    box=box, n=n, log_lik=ll, centers=centers,
                                    cell_area=area)
    lz = quadrature.atom_evidence_on_grid(base, atoms)
    resp = quadrature.responsibility_masses(tgt, atoms)
    assert abs(lz["log_odds"] - resp["log_odds"]) < 1e-9
    assert base.diagnostics["edge_mass"] < 1e-6


def test_oup_likelihood_matches_the_implemented_transition():
    """The sufficient-statistic form must equal the literal Euler chain.

    The exact-discretisation OU transition would be a different likelihood from
    the Euler chain that upstream simulates, so the test spells out the Euler
    step.
    """
    _, x = benchmark.load_observation("oup", "mixture", 0, 1000000)
    x = np.asarray(x, float)
    theta = np.random.default_rng(0).uniform(-1, 1, size=(5, 2))
    fast = quadrature.oup_log_likelihood(theta, x)
    coords = coordinates.get_coordinates("oup")
    phys = coords.to_physical(theta)
    xr = quadrature.oup_unstandardise(x)
    var = quadrature.OUP_DIFFUSION**2 * quadrature.OUP_DT
    slow = []
    for p in phys:
        s = 0.0
        for t in range(quadrature.OUP_NUM_POINTS - 1):
            mu = xr[t] + p[0] * (np.exp(p[1]) - xr[t]) * quadrature.OUP_DT
            s += (xr[t + 1] - mu) ** 2
        slow.append(-0.5 * s / var)
    assert np.allclose(fast, np.array(slow), atol=1e-9)


def test_sample_responsibility_reference_is_bounded_and_unbiased():
    shift = build_prior_shift("two_moons", "mixture", 2)
    atoms = atoms_mod.atoms_from_shift(shift)
    _, x_o = benchmark.load_observation("two_moons", "mixture", 2, 1000003)
    x_o = np.asarray(x_o, float)
    box = quadrature.default_box("two_moons", shift, pad_sd=6.0)
    n = 512
    centers, area = quadrature._grid(box, n)
    ll = quadrature.log_likelihood("two_moons", centers, x_o)
    tgt = quadrature.grid_posterior("two_moons", x_o, shift,
                                    kind="target_restricted", box=box, n=n,
                                    log_lik=ll, centers=centers, cell_area=area)
    exact = quadrature.responsibility_masses(tgt, atoms)
    drawn = tgt.sample(np.random.default_rng(0), 40000)
    est = quadrature.responsibility_masses_from_samples(drawn, atoms,
                                                        num_shards=4)
    assert np.all(est["gamma_min"] >= 0.0) and np.all(est["gamma_max"] <= 1.0)
    assert abs(est["log_odds"] - exact["log_odds"]) < 6.0 * max(
        est["log_odds_se"], 1e-3)
