# Spectra

Code and trained models for *Spectra: Exact Component Transport for Test-Time
Prior Adaptation in Simulation-Based Inference* by Xin Zhao, Nico Scherf, Robert
Trampel, Kerrin J. Pine and Nikolaus Weiskopf.

Spectra adapts a pretrained diffusion posterior sampler (a Simformer) to a new
prior at test time, without retraining.  When the ratio of the new prior to the
training prior is a single isotropic exponential–quadratic factor, the adapted
score is available in closed form and Spectra transports the sampler exactly.
For a mixture of such factors, each trajectory draws one component once.  The
component weights are estimated from the pretrained sampler itself, either by
path-space estimation (Spectra-PS, the method in the paper's main text) or
directly (Spectra-Direct).

## Installation

With Python 3.10:

```sh
pip install -r requirements.txt
```

This installs the CPU build of JAX.  For the experiments on an NVIDIA GPU,
install instead:

```sh
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-gpu.txt
```

## Quickstart

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) adapts the shipped Two
Moons model to a new prior and plots the samples before and after.  It runs on a
CPU in a few minutes and needs no reference posterior.  Open it with Jupyter
after installing `requirements.txt`.

## Running the experiments

The job scripts in `slurm/` rerun every experiment in the paper with the trained
models in `checkpoints/`.  [`slurm/README.md`](slurm/README.md) lists the jobs
in the order to run them and describes their output.

## License

MIT License (`LICENSE`).  `third_party/priorguide/` and the PriorGuide benchmark
files in `data/` are distributed under PriorGuide's MIT License, reproduced in
`third_party/priorguide/LICENSE`.
