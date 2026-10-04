#!/bin/bash
#SBATCH --job-name=spectra-anisotropy-sir
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
# SIR rows of the anisotropy boundary (CPU): the 20,000-point (100, 0) Base bank
# of the mild_0 single-factor controlled run (slurm/single_factor_controlled.sh),
# reweighted with each exact target and resampled.  The self-test first requires
# the stored mild_0 resampling indices back exactly.
# Output: $RESULTS/anisotropy/samples/two_moons/sir_n20000
source slurm/common.sh
export JAX_PLATFORMS=cpu

python scripts/sample_anisotropy_sir.py --self-test --root "$RESULTS/anisotropy" \
    --bank-root "$RESULTS/single_factor_controlled_mild"
