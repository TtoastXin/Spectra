# Job scripts

Each script here is a SLURM array job with the grid, arguments and XLA setting
of the paper's runs.  Add your cluster's partition and account options to the
`#SBATCH` lines, then submit from the repository root:

```sh
mkdir -p logs
sbatch slurm/references_single_factor.sh
```

To run one array element without SLURM:

```sh
SLURM_ARRAY_TASK_ID=0 bash slurm/references_single_factor.sh
```

Results go to `results/<study>/`.  `SPECTRA_RESULTS` sets another output root,
and `SPECTRA_CHECKPOINTS` other checkpoints to sample from.  `common.sh` has the
details.

## Order

| Step | Job scripts | Before running |
|---|---|---|
| Training data (optional) | `training_data.sh` | Elements 0–4 (the five PriorGuide tasks) need a PriorGuide checkout at commit `b4852fc` in `$PRIORGUIDE_UPSTREAM`, with `sbibm`, `pyro-ppl` and `numpyro` installed. |
| Training (optional) | `train.sh` | Needs `training_data.sh`.  Writes to `results/checkpoints`, which the other jobs use when `SPECTRA_CHECKPOINTS` points there. |
| References | `references_single_factor.sh`, `references_bci.sh`, `references_mixture.sh`, `references_severity.sh`, `references_k_scaling.sh` | Element 5 of `references_mixture.sh` (BCI component weights) needs `references_bci.sh`. |
| Single factor | `single_factor_controlled.sh`, `single_factor_oup.sh`, `single_factor_practical.sh`, `single_factor_dev.sh`, `single_factor_heldout.sh`, `single_factor_vjp.sh`, `severity.sh` | |
| Mixtures | `mixture_controlled.sh`, then `mixture_practical.sh` | Needs the mixture references and reference weights. |
| Component count | `k_scaling.sh` | Needs `references_k_scaling.sh`.  Element 3 also needs `mixture_controlled.sh`, because it checks against a stored K = 2 cell read from `results/mixture_controlled`. |
| Correlated prior | `correlated_prior.sh`, then `correlated_k64.sh` | |
| Anisotropy | `references_anisotropy.sh`, then `anisotropy.sh`.  Also `anisotropy_sir.sh`. | `anisotropy_sir.sh` reweights the SIR bank of `single_factor_controlled.sh` (mild_0). |
| Timing | `timing.sh` | Run on an otherwise idle node. |
| Metrics | `metrics.sh` | Needs the references and samples of each study. |

The two training steps are not needed to reproduce the paper.  All other jobs
sample from the shipped checkpoints.

## Output

The metric files `results/<study>/metrics/<task>.csv` have one row per sample
file.  The tasks are `two_moons` (Two Moons), `slcp` (SLCP), `oup` (OUP), `bav`
(BCI), `gaussian_linear` (GL-10D) and `gaussian_linear_high` (GL-20D).  Each
method is named `<row>_ms<seed>` for sampling seed `<seed>`, or `sir_n<bank>`
for SIR with a bank of that size.  The row names are the code's identifiers:

| Row | Method |
|---|---|
| `base` | Base (the pretrained sampler, no adaptation) |
| `pg_exact`, `pg` | PriorGuide with the exact prior ratio |
| `pg_vjp` | PriorGuide (VJP) |
| `a_full` | PG-FullCov |
| `spectra` | Spectra (single factor) |
| `eamt_ref_reference`, `eamt_ref_direct`, `eamt_ref_pathspace` | Spectra-Ref, Spectra-Direct and Spectra-PS |
| `spectra_ref`, `spectra_ps` | Spectra-Ref and Spectra-PS (component-count and anisotropy studies) |
| `pg_truecov`, `a_full_truecov`, `eamt_compiled_reference` | PriorGuide and PG-FullCov with the true correlated covariance, and Spectra on the compiled prior (correlated-prior study) |

Spectra-Ref takes the component weights from the reference posterior.  It is
not a practical method and is reported only to separate weight-estimation error
from the rest of the sampler.

## Reproducibility

- **Reruns.**  With the versions pinned in `requirements.txt` and
  `requirements-gpu.txt`, rerunning a sampling, weight-estimation, reference or
  metric job reproduces the paper's results exactly on an NVIDIA A100.  Other
  GPUs or drivers may change the last digits.  Without the CUDA library pins,
  pip installs newer libraries.  The jobs still run and the results agree
  statistically, but some GPU computations in the jobs that disable the Triton
  GEMM emitter change in the last digits.
- **XLA flag.**  Most GPU jobs run with
  `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`.  The exceptions ran with the
  default flags: the BCI controlled single-factor rows and BCI references, the
  mixture jobs of every task except Two Moons, and the training of every task
  except Two Moons.  The flag changes which GPU kernels XLA selects, and with
  them the last digits of floating-point results.
- **Training.**  The Two Moons checkpoints were trained with the Triton GEMM
  emitter disabled, because in our environment (JAX 0.4.28 on an NVIDIA A100)
  the default XLA:GPU stack segfaults in their backward pass.  Retraining them
  that way reproduces the shipped checkpoints exactly.  The other tasks were
  trained with the default flags, under which XLA autotunes its GPU kernels on
  the device.  Retraining them on other hardware gives an equally trained model
  that is not an exact copy of the shipped one.  Every result in the paper comes
  from the shipped checkpoints.
- **Timing.**  Timings are medians of synchronized repeats with compilation
  excluded, measured in one session on one A100.  Absolute values depend on the
  hardware.
- **Seeds.**  All random seeds are fixed.  The observation seeds are listed in
  `common.sh` and the job scripts.  The sampling, SMC and estimator seeds are
  set in the job scripts, in `scripts/*_config.py` or as the scripts' argument
  defaults.
