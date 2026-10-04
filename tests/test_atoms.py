"""Spectra: the decomposition, the evidence, the hook.

Spectra replaces an online marginal-field estimate with two quantities, and
they can go wrong in different ways:

1. the decomposition ``r ∝ sum_k b_k phi_k`` with every atom isotropic, so
   each atom has an exact transformed-query transport, and ``pi_k ∝ b_k Z_k``
   is the posterior atom weight.  This is algebra and is checked at machine
   precision;
2. the evidence ``Z_k = E_{p_0}[phi_k]``, one global scalar per atom and the
   only empirical quantity Spectra depends on.  The plug-in estimator is unbiased
   but its variance is governed by the overlap between the base posterior and the
   atom, which is very poor on the shipped Gaussian Linear shifts.  The tests
   cover both the easy regime (the estimate is accurate) and the hard one (the
   ESS diagnostic fires), so a bad benchmark number is not mistaken for a coding
   error.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectra import atoms as atoms_mod
from spectra import benchmark, hooks
from spectra.backbone import Backbone, CheckpointInfo
from spectra.oracles import GaussianMixture, single_gaussian, tilt_mixture
from spectra.prior_shift import build_prior_shift
from spectra.references import (
    GL_SIMULATOR_SCALE,
    gaussian_linear_base_posterior,
    gaussian_linear_posterior,
)
from spectra.simformer import (
    SimformerConfig,
    VESchedule,
    build_model,
    upstream_nn,
)

MIXTURE_CELLS = [
    ("gaussian_linear", "mixture", 0),
    ("gaussian_linear_high", "mixture", 0),
    ("gaussian_linear_high", "mixture", 4),
    ("two_moons", "mixture", 0),
]
ALL_CELLS = MIXTURE_CELLS + [
    ("gaussian_linear", "strong", 0),
    ("two_moons", "mild", 0),
]


# ------------------------------------------------------- the decomposition --


@pytest.mark.parametrize("task,prior_type,prior_id", ALL_CELLS)
def test_atoms_reproduce_log_r_up_to_one_global_constant(task, prior_type, prior_id):
    """``log sum_k b_k phi_k`` and ``log r`` may differ by a constant, nothing else.

    The guidance does not see that constant; any other difference would make
    the atom weights, which the categorical draw uses, wrong.
    """
    shift = build_prior_shift(task, prior_type, prior_id)
    atoms = atoms_mod.atoms_from_shift(shift)
    rng = np.random.default_rng(0)
    scale = float(np.mean(shift.target_sigma))
    theta = shift.target_mu[0] + 3.0 * scale * rng.normal(
        size=(64, shift.theta_dim))

    lp = atoms_mod.log_phi(theta, atoms)
    lhs = atoms_mod._logsumexp(atoms.log_b[None, :] + lp, axis=1)
    diff = lhs - shift.log_r(theta)
    assert np.max(np.abs(diff - diff.mean())) < 1e-8


@pytest.mark.parametrize("task,prior_type,prior_id", ALL_CELLS)
def test_every_atom_is_isotropic_and_normalisable(task, prior_type, prior_id):
    shift = build_prior_shift(task, prior_type, prior_id)
    atoms = atoms_mod.atoms_from_shift(shift)
    assert atoms.num_atoms == shift.target_pi.shape[0]
    assert np.all(atoms.kappa > 0)
    assert np.all(np.isfinite(atoms.log_b))


@pytest.mark.parametrize("prior_type", ["mild", "strong"])
def test_single_atom_matches_the_expquad_transport_parameters(prior_type):
    """For ``K = 1`` Spectra's atom must be the very ``(a, kappa)`` TQ already uses."""
    shift = build_prior_shift("gaussian_linear", prior_type, 0)
    atoms = atoms_mod.atoms_from_shift(shift)
    assert atoms.num_atoms == 1
    np.testing.assert_allclose(atoms.a[0], shift.expquad.a, rtol=1e-10, atol=1e-12)
    assert math.isclose(float(atoms.kappa[0]), shift.expquad.kappa, rel_tol=1e-10)


def test_isotropy_guard_rejects_an_anisotropic_atom():
    """The guard exists because ``r``'s atoms become transport atoms.

    An anisotropic component has no exact isotropic transformed-query form, so it
    must raise at construction instead of being replaced by its diagonal.
    """
    class _Shift:
        task = "synthetic"
        comp_log_weights = np.zeros(1)
        comp_means = np.zeros((1, 2))
        comp_covs = np.array([[[0.2, 0.0], [0.0, 0.5]]])

    with pytest.raises(ValueError, match="not isotropic"):
        atoms_mod.atoms_from_shift(_Shift())


# ------------------------------------------------------------ the evidence --


def _exact_log_evidence(atoms, mean, var, d):
    """``log E_{N(mean, var I)}[phi_k]`` in closed form."""
    out = np.empty(atoms.num_atoms)
    for k in range(atoms.num_atoms):
        v = var + 1.0 / atoms.kappa[k]
        out[k] = (-0.5 * np.sum((atoms.nu[k] - mean) ** 2) / v
                  - 0.5 * d * math.log(2 * math.pi * v))
    return out


@pytest.mark.parametrize("task,prior_id", [("gaussian_linear", 0),
                                           ("gaussian_linear_high", 0),
                                           ("gaussian_linear_high", 4)])
def test_analytic_atom_weights_equal_the_evidence_decomposition(task, prior_id):
    """``pi*`` two ways: conjugate update, and ``b_k Z_k`` on the base posterior.

    ``gaussian_linear_posterior`` reweights the test prior components by their
    predictive evidence; Spectra reweights the ratio atoms by their expectation
    under the base posterior.  The two must agree, which justifies using the
    closed-form posterior weights as ``pi*``.
    """
    shift = build_prior_shift(task, "mixture", prior_id)
    _, x_o = benchmark.load_observation(task, "mixture", prior_id, 1000000)
    x_o = np.asarray(x_o, float)
    atoms = atoms_mod.atoms_from_shift(shift)

    base = gaussian_linear_base_posterior(x_o, shift, GL_SIMULATOR_SCALE)
    log_z = _exact_log_evidence(atoms, np.asarray(base.means[0], float),
                                float(base.variances[0]), shift.theta_dim)
    from_atoms = np.exp(atoms_mod.atom_log_weights(atoms, log_z))
    from_posterior = gaussian_linear_posterior(x_o, shift, GL_SIMULATOR_SCALE).weights
    np.testing.assert_allclose(from_atoms, from_posterior, rtol=1e-8, atol=1e-10)


def test_plugin_evidence_is_accurate_when_the_atom_overlaps_the_base():
    """The regime the plug-in is designed for: wide atom, low dimension."""
    d, n = 2, 200_000
    rng = np.random.default_rng(4)
    mean, var = np.zeros(d), 0.25
    atoms = atoms_mod.AtomSet(
        log_b=np.log([0.4, 0.6]),
        nu=np.array([[0.2, -0.1], [-0.3, 0.15]]),
        kappa=np.array([4.0, 4.0]),
    )
    bank = mean + math.sqrt(var) * rng.standard_normal((n, d))
    got = atoms_mod.evidence_plugin(bank, atoms)
    exact = _exact_log_evidence(atoms, mean, var, d)
    assert np.all(got["ess"] > 0.05 * n)
    assert abs(atoms_mod.log_odds(atoms, got["log_z"])
               - atoms_mod.log_odds(atoms, exact)) < 0.05


@pytest.mark.parametrize("task,prior_id", [("gaussian_linear", 0),
                                           ("gaussian_linear_high", 4)])
def test_plugin_ess_diagnostic_fires_on_the_shipped_gaussian_linear_shifts(
        task, prior_id):
    """The plug-in ESS diagnostic flags the direct-evidence estimate on these shifts.

    On an exact base-posterior bank of the size the benchmark uses, the atom
    ESS collapses to O(1): the test prior is 5x narrower (in standard
    deviation) than the training prior, so almost no base sample lands inside
    an atom.  This is a property of the shift, not of the learned model or of
    the implementation.
    """
    shift = build_prior_shift(task, "mixture", prior_id)
    _, x_o = benchmark.load_observation(task, "mixture", prior_id, 1000000)
    atoms = atoms_mod.atoms_from_shift(shift)
    base = gaussian_linear_base_posterior(np.asarray(x_o, float), shift,
                                          GL_SIMULATOR_SCALE)
    rng = np.random.default_rng(5)
    bank = (np.asarray(base.means[0], float)
            + math.sqrt(float(base.variances[0]))
            * rng.standard_normal((5000, shift.theta_dim)))
    got = atoms_mod.evidence_plugin(bank, atoms, num_bootstrap=50, seed=0)
    assert np.all(got["ess"] < 50.0)
    assert np.all(got["rse"] > 0.1)
    assert got["bootstrap_log_odds_sd"] > 0.5


# ----------------------------------------------------------------- the hook --


class _AnalyticBackbone:
    """Backbone-shaped view of an analytic Gaussian-mixture base posterior.

    Only what the hooks call is provided.  ``score_theta_t`` accepts a per-row
    time vector, which Spectra needs to keep one batched forward pass per step
    when walkers carry different atoms.
    """

    def __init__(self, mix: GaussianMixture, schedule: VESchedule):
        self.mix = mix
        self.schedule = schedule
        self.theta_dim = mix.dim

    def nominal_t_range(self):
        return float(self.schedule.t_min), float(self.schedule.t_max)

    def score_theta_t(self, z, t, x_o):
        z = jnp.atleast_2d(z)
        tau = self.schedule.sigma(jnp.asarray(t, z.dtype)) ** 2
        tau = jnp.broadcast_to(jnp.atleast_1d(tau), (z.shape[0],))
        return jax.vmap(self.mix.score)(z, tau)


@pytest.mark.parametrize("tau_t", [0.15, 0.45, 0.8])
def test_eamt_hook_is_the_exact_target_score_of_its_own_atom(tau_t):
    """Per atom, the Spectra hook returns the exact target score.

    ``q_k ∝ p_0 phi_k`` is available in closed form for a Gaussian-mixture base,
    so the hook's output is compared against the analytic score of that tilted
    mixture at the same noise level, without sampling.
    """
    d = 4
    rng = np.random.default_rng(6)
    mix = GaussianMixture(
        log_weights=jnp.asarray(np.log([0.35, 0.65])),
        means=jnp.asarray(rng.normal(size=(2, d)) * 0.3),
        covs=jnp.asarray(np.stack([np.eye(d) * 0.08, np.eye(d) * 0.05])),
    )
    sch = VESchedule()
    backbone = _AnalyticBackbone(mix, sch)
    atoms = atoms_mod.AtomSet(
        log_b=np.log([0.4, 0.6]),
        nu=rng.normal(size=(2, d)) * 0.2,
        kappa=np.array([9.0, 25.0]),
    )
    z = jnp.asarray(rng.normal(size=(12, d)) * 0.5)
    tau = float(sch.sigma(tau_t) ** 2)

    for k in range(atoms.num_atoms):
        labels = jnp.full((z.shape[0],), k, dtype=jnp.int32)
        hook, ops, meta = hooks.make_spectra_hook(backbone, jnp.zeros(1), atoms, labels)
        got, _ = hook(z, tau_t)
        q_k = tilt_mixture(mix, jnp.asarray(atoms.a[k]),
                           jnp.asarray(atoms.kappa[k] * np.eye(d)))
        want = jax.vmap(q_k.score, in_axes=(0, None))(z, tau)
        rel = np.linalg.norm(np.asarray(got - want), axis=-1) / np.maximum(
            np.linalg.norm(np.asarray(want), axis=-1), 1e-9)
        assert np.max(rel) < 2e-4
        assert ops == {"base_score": 1}
        assert meta["num_atoms"] == 2


def test_mixed_atom_batch_matches_the_single_atom_rows():
    """Batching walkers with different atoms must not change any single walker.

    Different atoms mean different transformed states and different transformed
    times in the same forward pass; this test fails if the per-row time is
    broadcast from row 0.
    """
    d = 3
    rng = np.random.default_rng(7)
    mix = single_gaussian(jnp.asarray(rng.normal(size=d) * 0.2),
                          jnp.asarray(np.eye(d) * 0.06))
    sch = VESchedule()
    backbone = _AnalyticBackbone(mix, sch)
    atoms = atoms_mod.AtomSet(log_b=np.log([0.5, 0.5]),
                              nu=rng.normal(size=(2, d)) * 0.3,
                              kappa=np.array([6.0, 40.0]))
    z = jnp.asarray(rng.normal(size=(8, d)) * 0.4)
    labels = jnp.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=jnp.int32)

    hook, _, _ = hooks.make_spectra_hook(backbone, jnp.zeros(1), atoms, labels)
    mixed, _ = hook(z, 0.5)
    for k in (0, 1):
        hk, _, _ = hooks.make_spectra_hook(
            backbone, jnp.zeros(1), atoms,
            jnp.full((z.shape[0],), k, dtype=jnp.int32))
        pure, _ = hk(z, 0.5)
        sel = np.asarray(labels) == k
        np.testing.assert_allclose(np.asarray(mixed)[sel], np.asarray(pure)[sel],
                                   rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def learned_ctx():
    """A real (randomly initialised) Simformer, to pin Spectra against TQ."""
    theta_dim, x_dim = 3, 2
    nn = upstream_nn()
    sch = VESchedule()
    cfg = SimformerConfig(token_dim=8, time_embedding_dim=8, num_layers=1,
                          num_heads=2)
    num_nodes = theta_dim + x_dim
    init_fn, _ = build_model(num_nodes, sch, cfg, nn)
    params = init_fn(
        jax.random.PRNGKey(11), jnp.ones((2,)), jnp.zeros((2, num_nodes, 1)),
        jnp.arange(num_nodes), jnp.zeros((2, num_nodes, 1)), meta_data=None)
    ckpt = CheckpointInfo(
        params=params, theta_dim=theta_dim, x_dim=x_dim, schedule=sch,
        model_cfg=cfg, marginal_end_mean=np.zeros(num_nodes),
        marginal_end_std=np.full(num_nodes, 15.0), source="spectra",
        path="<test>", meta={})
    return Backbone(ckpt, mode="conditional", nn=nn), theta_dim


@pytest.mark.parametrize("t", [0.1, 0.5, 0.95])
def test_single_atom_spectra_matches_the_tq_hook(learned_ctx, t):
    """With one atom Spectra must reduce to the single-factor transport of the TQ hook."""
    backbone, d = learned_ctx
    rng = np.random.default_rng(12)
    a = rng.normal(size=d)
    kappa = 7.5
    atoms = atoms_mod.AtomSet(log_b=np.zeros(1), nu=(a / kappa)[None, :],
                              kappa=np.array([kappa]))
    z = jnp.asarray(rng.normal(size=(6, d)))
    x_o = jnp.asarray(rng.normal(size=2))
    labels = jnp.zeros((6,), dtype=jnp.int32)

    spectra, _, _ = hooks.make_spectra_hook(backbone, x_o, atoms, labels)
    tq, _, _ = hooks.make_tq_hook(backbone, x_o, jnp.asarray(a), kappa)
    got, _ = spectra(z, t)
    want, _ = tq(z, t)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=1e-5, atol=1e-6)


def test_atom_labels_follow_the_weights_and_use_their_own_key():
    log_pi = np.log([0.25, 0.75])
    labels = hooks.draw_atom_labels(jax.random.PRNGKey(0), log_pi, 20000)
    frac = np.bincount(np.asarray(labels), minlength=2) / 20000
    np.testing.assert_allclose(frac, [0.25, 0.75], atol=0.02)
    again = hooks.draw_atom_labels(jax.random.PRNGKey(0), log_pi, 20000)
    np.testing.assert_array_equal(np.asarray(labels), np.asarray(again))
