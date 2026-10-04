# Benchmark priors and observations

Layout: `priors/<task>/<prior>.json` and
`observations/<task>/<prior>/obs_<seed>.json`, where `<prior>` is `training` or
`<type>_<id>` with `<type>` one of `mild`, `strong`, `mixture`.
`anisotropy/cells.json` holds the anisotropy study's target records (below).
Task names: `gaussian_linear` (GL-10D), `gaussian_linear_high` (GL-20D),
`two_moons`, `slcp`, `oup`, `bav` (BCI). Parameters are in the coordinates used
by the diffusion model.

## Sources

**PriorGuide benchmark files.** All files for `gaussian_linear`,
`gaussian_linear_high` and `bav`, and the `training`, `mild_0`–`mild_9`,
`strong_0`–`strong_9` and `mixture_0`–`mixture_9` files for `two_moons` and
`oup`, are copied unchanged from PriorGuide
(<https://github.com/acerbilab/prior-guide>, commit `b4852fc`,
`experiments/data/`), distributed under the MIT License in
`third_party/priorguide/LICENSE`.

**SLCP.** PriorGuide does not ship SLCP. All `slcp` priors and observations
(except the shift-severity cells below) were generated for this work by
`scripts/prepare_slcp.py`, following the same construction as the uniform-prior
PriorGuide tasks.

**Shift-severity sweep** (`mild_20`, `mild_25`, `mild_30`, `mild_40`,
`mild_50` for `two_moons`, `oup` and `slcp`). Each keeps the center of
`mild_0` and scales its width to the stated fraction of the training-prior
standard deviation (for example `mild_20` is 0.20). Its observations are the
`mild_0` observations. Written by `scripts/prepare_severity_priors.py` (cell
definition in `scripts/severity_config.py`).

**Component-count study** (`two_moons` `mixture_101`, `mixture_102`,
`mixture_104`, `mixture_108`, `mixture_116`, `mixture_132`, `mixture_164`).
`mixture_<100+K>` has K equal-weight components centered on every (64/K)-th
point of one fixed 64-point circle around the `mild_0` center, with the width of
`strong_0`. Observations: three per prior, copied from `mild_0`. Written by
`scripts/prepare_k_mixtures.py` (constants in `scripts/k_scaling_config.py`).

**Correlated-Gaussian approximation** (`two_moons` `mixture_20`, `mixture_21`).
Positive-quadrature approximations of one correlated Gaussian target prior with
27 and 64 isotropic components. The `compiled_from` field records the target
and the construction. Observations are copied from `mixture_2`. Written by
`scripts/compile_correlated_prior.py` (`slurm/correlated_prior.sh`).

**Anisotropy boundary** (`two_moons` `mixture_61`–`mixture_65`, and
`anisotropy/cells.json`). `cells.json` records the exact anisotropic Gaussian
targets: the `mild_20` center and determinant with the per-axis variances
`s^2/gamma` and `s^2 gamma` in either orientation (gamma = 1, 2, 4). These targets
are not single factors, so they are stored as records rather than prior files.
PriorGuide reads its exact ratio from them. `mixture_62`–`mixture_65` are their
positive-quadrature approximations with 52 (gamma = 2) and 104 (gamma = 4)
isotropic components, and `mixture_61` is the compiled gamma = 1 target, a
compiler sanity check that no method uses. Observations are copied from
`mild_20`. Written by `scripts/prepare_anisotropy_priors.py` (cell definition in
`scripts/anisotropy_config.py`, compiler in `scripts/anisotropy_compiler.py`).

Rerunning these scripts reproduces the shipped files exactly.

## Training sets

The Simformer training sets (10,000 simulations per task and model seed) are not
shipped. `slurm/training_data.sh` regenerates them exactly: the five PriorGuide
tasks with PriorGuide's own task code (`scripts/simulate_training_data.py`) and
SLCP with `scripts/prepare_slcp.py --step data`.
