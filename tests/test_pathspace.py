"""Path-space normaliser estimator: the transported oracle score, exactness, and refusals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectra import atoms as atoms_mod
from spectra import e2e, pathspace
from spectra.simformer import VESchedule


def _toy_atoms():
    return atoms_mod.AtomSet(log_b=np.log(np.array([0.4, 0.6])),
                             nu=np.array([[0.2, -0.1], [-0.3, 0.25]]),
                             kappa=np.array([40.0, 40.0]))


def test_transported_oracle_score_equals_the_tilted_gaussian_score():
    sch = VESchedule()
    atoms = _toy_atoms()
    res = pathspace.atom_score_parity(
        sch, atoms, base_mean=np.array([0.05, -0.02]), base_var=0.05,
        taus=[float(sch.sigma(t) ** 2) for t in (0.05, 0.3, 0.6, 1.0)],
        rng=np.random.default_rng(0), num=32)
    assert res["max_abs_score_gap"] < 1e-4


def test_path_weights_recover_a_known_log_z_with_an_exact_score():
    """With an exact score the integrand is constant in the continuum limit.

    So this both checks the sign of the change of measure and pins the claim the
    oracle control rests on: any spread in the weights is discretisation.
    """
    sch = VESchedule()
    atoms = _toy_atoms()
    m_b, v_b = np.array([0.05, -0.02]), 0.05
    score_p, make_q = pathspace.oracle_scores(sch, atoms, m_b, v_b)
    d = m_b.shape[0]
    term_mean = np.zeros(d)
    term_std = np.full(d, float(sch.sigma(sch.t_max)))
    truth = np.array([
        -0.5 * np.sum((atoms.nu[k] - m_b) ** 2) / (v_b + 1.0 / atoms.kappa[k])
        - 0.5 * d * math.log(2 * math.pi * (v_b + 1.0 / atoms.kappa[k]))
        for k in range(2)])
    errs = {}
    for steps in (100, 800):
        cfg = e2e.SamplerConfig(num_steps=steps, grid="power", langevin_steps=0)
        res = pathspace.pathspace_evidence(
            score_p, make_q, schedule=sch, cfg=cfg, atoms=atoms, atom=0,
            terminal_mean=term_mean, terminal_std=term_std,
            num_trajectories=1024, seed=0)
        errs[steps] = abs(res.log_z - truth[0])
        assert res.diagnostics["integrand"]["status"] == "ok"
    assert errs[800] < errs[100]
    assert errs[800] < 0.15


def test_path_weights_refuse_a_langevin_corrector():
    sch = VESchedule()
    atoms = _toy_atoms()
    score_p, make_q = pathspace.oracle_scores(sch, atoms, np.zeros(2), 0.05)
    cfg = e2e.SamplerConfig(num_steps=10, langevin_steps=3)
    with pytest.raises(ValueError):
        pathspace.run_atom_chain(
            score_p, make_q(0), schedule=sch, cfg=cfg,
            x_init=np.zeros((4, 2)), diffusion_noise=np.zeros((9, 4, 2)))


def test_weight_diagnostics_report_a_dead_estimator_rather_than_crashing():
    d = pathspace.weight_diagnostics(np.full(64, -np.inf))
    assert d["status"] == "nonfinite" and d["ess"] == 0.0
    d2 = pathspace.weight_diagnostics(np.zeros(64))
    assert math.isclose(d2["ess_normalized"], 1.0)
