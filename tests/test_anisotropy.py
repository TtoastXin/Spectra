"""Anisotropy cells: the tie-up compiler and the shipped compiled dictionaries.

The compiler used for the anisotropic targets differs from the shared one only in
how an integer node-count criterion is rounded.  It must reproduce the correlated
prior's shipped dictionaries exactly, the opposite rounding must fail that
check, and the shipped anisotropic dictionaries must be exactly what it compiles
from the recorded targets.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import anisotropy_compiler as RC  # noqa: E402
from scripts import anisotropy_config as G  # noqa: E402
from spectra import benchmark, compile_atoms  # noqa: E402


def test_tie_up_rule_reproduces_the_correlated_prior_dictionaries():
    assert RC.correlated_prior_check("tie_up") == []


def test_tie_down_rule_is_rejected_by_the_same_check():
    reasons = RC.correlated_prior_check("tie_down")
    assert reasons and all("tie_down" in r for r in reasons)


def test_node_order_rounds_integer_ties_up_and_is_ceil_plus_one_elsewhere():
    assert RC.node_order(2.0000000000000018, "tie_up", 31) == 4
    assert RC.node_order(1.9999999999999982, "tie_up", 31) == 4
    assert RC.node_order(2.0000000000000018, "tie_down", 31) == 3
    for raw in (0.3, 1.5, 2.7, 12.01):
        assert RC.node_order(raw, "tie_up", 31) == int(np.ceil(raw)) + 1


def test_shipped_anisotropic_dictionaries_are_the_compiled_targets():
    cells = G.load_cells()
    box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    expected_k = {61: 16, 62: 52, 63: 52, 64: 104, 65: 104}
    for cid in G.COMPILED_CELLS:
        c = cells[cid]
        prior = compile_atoms.CorrelatedGaussianPrior(
            mu=np.asarray(c["mu"], float), cov=np.asarray(c["cov"], float), box=box, rule="")
        d, rec = RC.compile_dictionary(prior, spread_factor=compile_atoms.CAPACITIES[G.CAPACITY])
        got = json.loads(json.dumps(d.as_prior_json(G.TASK)))
        want = benchmark.load_prior_json(G.TASK, "mixture", cid)
        for key in ("mu", "sigma", "pi"):
            assert got[key] == want[key], (cid, key)
        assert rec["num_atoms"] == c["compiled"]["num_atoms"] == expected_k[cid]
        assert want["compilation"]["node_orders"] == rec["node_orders"]


def test_cells_fix_centre_and_determinant_and_vary_only_anisotropy():
    cells = G.load_cells()
    src = benchmark.load_prior_json(G.TASK, G.SOURCE_PRIOR_TYPE, G.SOURCE_PRIOR_ID)
    s = float(src["sigma"][0])
    for cid, c in cells.items():
        cov = np.asarray(c["cov"], float)
        assert c["mu"] == src["mu"]
        assert np.isclose(np.linalg.det(cov), s ** 4, rtol=1e-12)
        assert np.isclose(np.linalg.cond(cov), c["gamma"] ** 2, rtol=1e-12)
        assert np.allclose(np.diag(cov), G.cov_diag(cid, s), rtol=0, atol=0)
