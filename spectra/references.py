"""Reference posteriors, written independently of the guidance code.

* Gaussian Linear 10D / 20D: closed form.  The conjugate update is derived
  here from the sbibm task definition (prior scale 0.1, simulator scale 0.1,
  identity design) rather than by calling upstream's
  ``GaussianLinear._get_reference_posterior``; ``tests/test_references.py``
  checks it against an independent importance sampler.
* Two Moons: rejection sampling with the test prior as proposal, following
  the upstream recipe with two differences.  The mixture proposal keeps the
  prior's own weights (upstream uses ``[0.5, 0.5]``), and the likelihood bound
  is the maximum of ``f(rho)/rho`` instead of ``f(mu_r)/mu_r``, which is about
  0.5% smaller than the maximum.  Both bounds are recorded with the samples.

Support caveat: for the uniform-training tasks the target prior has mass
outside ``supp(pi_train)``, which no frozen-base-posterior adaptation can
reach.  Every reference bank therefore also reports the outside-box fraction,
and Two Moons additionally gets a support-matched reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from spectra.benchmark import outside_box_fraction
from spectra.prior_shift import PriorShift

# sbibm Gaussian Linear (and High): theta ~ N(0, 0.1 I), x | theta ~ N(theta, 0.1 I)
GL_PRIOR_SCALE = 0.1
GL_SIMULATOR_SCALE = 0.1

# sbibm Two Moons simulator constants
TM_A_LOW, TM_A_HIGH = -math.pi / 2.0, math.pi / 2.0
TM_BASE_OFFSET = 0.25
TM_R_LOC, TM_R_SCALE = 0.1, 0.01


# ------------------------------------------------------- Gaussian Linear ----


@dataclass(frozen=True)
class GaussianMixture:
    """``sum_k w_k N(mu_k, Sigma_k)`` with isotropic components."""

    weights: np.ndarray  # (K,)
    means: np.ndarray  # (K, d)
    variances: np.ndarray  # (K,) isotropic

    def sample(self, rng, num: int) -> np.ndarray:
        k = rng.choice(len(self.weights), size=num, p=self.weights)
        sd = np.sqrt(self.variances)[k][:, None]
        return self.means[k] + sd * rng.standard_normal((num, self.means.shape[1]))

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.means

    @property
    def cov(self) -> np.ndarray:
        d = self.means.shape[1]
        m = self.mean
        out = np.zeros((d, d))
        for w, mu, v in zip(self.weights, self.means, self.variances):
            dm = (mu - m)[:, None]
            out += w * (v * np.eye(d) + dm @ dm.T)
        return out

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        d = theta.shape[-1]
        logs = []
        for w, mu, v in zip(self.weights, self.means, self.variances):
            r2 = np.sum((theta - mu) ** 2, axis=-1)
            logs.append(
                math.log(w) - 0.5 * r2 / v - 0.5 * d * math.log(2 * math.pi * v)
            )
        logs = np.stack(logs, 0)
        m = logs.max(0)
        return m + np.log(np.exp(logs - m).sum(0))


def gaussian_linear_posterior(x_o: np.ndarray, shift: PriorShift,
                              simulator_scale: float = GL_SIMULATOR_SCALE
                              ) -> GaussianMixture:
    """Analytic target posterior under the shifted (isotropic) test prior.

    Component ``k`` of the prior gives precision ``1/s_k^2 + 1/sigma_sim^2`` and
    mean ``cov (mu_k / s_k^2 + x / sigma_sim^2)``; the mixture weights are
    reweighted by the evidence ``N(x; mu_k, (s_k^2 + sigma_sim^2) I)``.
    """
    x_o = np.asarray(x_o, float).reshape(-1)
    d = x_o.shape[0]
    var_q = shift.target_sigma[:, 0] ** 2  # isotropic, checked at build time
    post_var = 1.0 / (1.0 / var_q + 1.0 / simulator_scale)
    post_mean = post_var[:, None] * (
        shift.target_mu / var_q[:, None] + x_o[None, :] / simulator_scale
    )
    pred_var = var_q + simulator_scale
    log_ev = -0.5 * np.sum((x_o[None, :] - shift.target_mu) ** 2, axis=-1) / pred_var
    log_ev = log_ev - 0.5 * d * np.log(2 * math.pi * pred_var)
    log_w = np.log(shift.target_pi) + log_ev
    log_w -= log_w.max()
    w = np.exp(log_w)
    return GaussianMixture(w / w.sum(), post_mean, post_var)


def gaussian_linear_base_posterior(x_o: np.ndarray, shift: PriorShift,
                                   simulator_scale: float = GL_SIMULATOR_SCALE
                                   ) -> GaussianMixture:
    """Analytic posterior under the training prior, which the backbone learned.

    Scoring base Simformer against the shifted target would mostly measure the
    prior shift it does not see.  Against this reference it measures the
    backbone itself, which is the score floor every method inherits.
    """
    x_o = np.asarray(x_o, float).reshape(-1)
    var_p = float(shift.train_sigma[0] ** 2)
    post_var = 1.0 / (1.0 / var_p + 1.0 / simulator_scale)
    post_mean = post_var * (shift.train_mu / var_p + x_o / simulator_scale)
    return GaussianMixture(np.ones(1), post_mean[None, :], np.array([post_var]))


# ------------------------------------------------------------- Two Moons ----


def _two_moons_likelihood(theta: np.ndarray, x_o: np.ndarray) -> np.ndarray:
    """``p(x_o | theta)`` in Cartesian coordinates, zero outside the half-disc.

    The ``1 / (pi rho)`` factor is the polar Jacobian of the simulator's
    ``(a, r) -> p`` map; the sbibm ``_likelihood`` helper omits it, so the
    rejection sampler needs this form rather than that one.
    """
    theta = np.atleast_2d(theta)
    rot = 1.0 / math.sqrt(2.0)
    m_x = -np.abs(theta[:, 0] + theta[:, 1]) * rot
    m_y = (-theta[:, 0] + theta[:, 1]) * rot
    ux = x_o[0] - (m_x + TM_BASE_OFFSET)
    uy = x_o[1] - m_y
    rho = np.hypot(ux, uy)
    phi = np.arctan2(uy, ux)
    ok = (rho > 0) & (phi >= -math.pi / 2) & (phi <= math.pi / 2)
    dens = np.zeros_like(rho)
    valid = ok & (rho > 0)
    r = rho[valid]
    dens[valid] = (
        (1.0 / math.pi)
        * (1.0 / (math.sqrt(2 * math.pi) * TM_R_SCALE))
        * np.exp(-((r - TM_R_LOC) ** 2) / (2 * TM_R_SCALE**2))
        / r
    )
    return dens


def two_moons_mode_mass(samples: np.ndarray) -> float:
    """Fraction of mass in the ``theta_1 + theta_2 > 0`` crescent.

    The forward map uses ``m_x = -|theta_1 + theta_2| / sqrt(2)``, so a given
    observation admits exactly two parameter solutions and they are separated by
    the sign of ``theta_1 + theta_2``.  The split is therefore exact rather than a
    clustering heuristic.  A method can match every marginal and still put the
    wrong mass on a mode.
    """
    samples = np.atleast_2d(samples)
    return float((samples[:, 0] + samples[:, 1] > 0).mean())


def two_moons_likelihood_bound() -> tuple[float, float]:
    """Exact ``max_rho f(rho)/rho`` and upstream's value.

    Maximising ``exp(-(rho-mu)^2 / 2 sigma^2) / rho`` gives
    ``rho* = (mu + sqrt(mu^2 - 4 sigma^2)) / 2``; upstream evaluates at
    ``rho = mu`` instead, which under-states the bound by about 0.5% and makes
    the acceptance probability exceed 1 in a thin shell.
    """
    mu, sd = TM_R_LOC, TM_R_SCALE
    rho_star = (mu + math.sqrt(mu**2 - 4 * sd**2)) / 2.0
    const = (1.0 / math.pi) * (1.0 / (math.sqrt(2 * math.pi) * sd))
    exact = const * math.exp(-((rho_star - mu) ** 2) / (2 * sd**2)) / rho_star
    upstream = const / mu
    return exact, upstream


@dataclass
class RejectionReference:
    samples: np.ndarray
    proposals: int
    accepted: int
    seed: int
    bound: float
    upstream_bound: float
    max_acceptance_ratio: float
    diagnostics: dict = field(default_factory=dict)
    # Which test-prior mixture component proposed each accepted draw.  Because
    # the proposal is the test prior and acceptance is by likelihood only, the
    # accepted ``(theta, k)`` pairs are distributed as the joint posterior
    # ``p(theta, k | x_o)``, so the label frequencies estimate the target
    # component weights ``pi*`` without using any of the methods under test
    # (Two Moons has no closed form).  ``None`` for the training-prior proposal,
    # which has no components.  Recording the labels does not consume random
    # numbers, so the samples are the same as without labels.
    component_labels: np.ndarray | None = None

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposals if self.proposals else float("nan")


def two_moons_reference(x_o, shift: PriorShift, num_samples: int, seed: int,
                        batch: int = 200_000, max_proposals: int = 200_000_000,
                        proposal: str = "target") -> RejectionReference:
    """Rejection sampler for the Two Moons posterior.

    ``proposal="target"`` draws from the shifted test prior and gives the target
    posterior, the reference every prior-adapted method is scored against.
    ``proposal="training"`` draws from the uniform training box instead and gives
    the posterior the frozen backbone was trained for, which is the reference
    for the unguided base Simformer row.
    """
    x_o = np.asarray(x_o, float).reshape(-1)
    rng = np.random.default_rng(seed)
    bound, upstream_bound = two_moons_likelihood_bound()
    if proposal == "training" and shift.train_box is None:
        raise ValueError("a training-prior proposal needs a bounded training prior")
    out, labels, proposals, worst = [], [], 0, 0.0
    total = 0
    while total < num_samples:
        if proposals >= max_proposals:
            raise RuntimeError(
                f"two moons rejection did not reach {num_samples} samples in "
                f"{proposals} proposals (accepted {total})"
            )
        k = None
        if proposal == "training":
            low, high = shift.train_box
            theta = rng.uniform(low, high, size=(batch, shift.theta_dim))
        else:
            pi = np.asarray(shift.target_pi, float)
            # shipped mixture weights are float32 round-trips; renormalise
            k = rng.choice(len(pi), size=batch, p=pi / pi.sum())
            theta = shift.target_mu[k] + shift.target_sigma[k] * rng.standard_normal(
                (batch, shift.theta_dim)
            )
        ratio = _two_moons_likelihood(theta, x_o) / bound
        worst = max(worst, float(ratio.max()))
        # the accept mask is stored so the proposing component can be kept
        # alongside the draw
        accept = rng.random(batch) < ratio
        keep = theta[accept]
        out.append(keep)
        if k is not None:
            labels.append(k[accept])
        total += keep.shape[0]
        proposals += batch
    samples = np.concatenate(out)[:num_samples]
    ref = RejectionReference(
        samples=samples, proposals=proposals, accepted=total, seed=seed,
        bound=bound, upstream_bound=upstream_bound, max_acceptance_ratio=worst,
        component_labels=(np.concatenate(labels)[:num_samples] if labels else None),
    )
    ref.diagnostics["proposal"] = proposal
    if shift.train_box is not None and proposal == "target":
        ref.diagnostics["posterior_mass_outside_training_box"] = outside_box_fraction(
            samples, shift.train_box
        )
        ref.diagnostics["prior_mass_outside_training_box"] = (
            shift.target_mass_outside_training_box()
        )
    return ref


def support_matched(samples: np.ndarray, box) -> np.ndarray:
    """Restrict a reference bank to ``supp(pi_train)`` and renormalise by dropping.

    A frozen base posterior puts no clean mass outside the training box, so this
    is the target against which structured transport is exact; the untruncated
    bank keeps the coverage gap visible.
    """
    low, high = box
    keep = np.all((samples >= low) & (samples <= high), axis=-1)
    return samples[keep]


def seed_stability(per_seed_samples: list) -> dict:
    """Cross-seed reproducibility of a reference, independent of the atom count.

    For K=2 the grade is based on the spread of the component log-odds, which
    is undefined when K=1 (``log_odds`` is NaN).  Agreement between
    independently seeded runs on the posterior itself is always defined, so this
    reports, per parameter axis, the largest seed-to-seed difference in the mean
    in units of that axis' posterior standard deviation, plus the same for the
    standard deviations.

    No pass/fail threshold is applied: a discrepancy is only meaningful next to
    the method gap it has to resolve, so the numbers are carried into the
    metrics stage and compared there against a reference-vs-reference C2ST and
    its split-half null.
    """
    if len(per_seed_samples) < 2:
        return {"num_seeds": len(per_seed_samples)}
    means = np.stack([s.mean(0) for s in per_seed_samples])
    sds = np.stack([s.std(0, ddof=1) for s in per_seed_samples])
    pooled_sd = np.concatenate(per_seed_samples).std(0, ddof=1)
    mean_gap = (means.max(0) - means.min(0)) / pooled_sd
    sd_gap = (sds.max(0) - sds.min(0)) / pooled_sd
    return {
        "num_seeds": len(per_seed_samples),
        "per_seed_mean": means.tolist(),
        "per_seed_sd": sds.tolist(),
        "pooled_sd": pooled_sd.tolist(),
        "max_standardised_mean_gap": float(mean_gap.max()),
        "max_standardised_sd_gap": float(sd_gap.max()),
        "standardised_mean_gap_per_axis": mean_gap.tolist(),
    }
