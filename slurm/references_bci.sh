#!/bin/bash
#SBATCH --job-name=spectra-refs-bci
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
# BCI reference posteriors (GPU; Gauss-Hermite likelihood, tempered SMC), with the
# default XLA flags as in the paper's runs.
#   0-19   single factor: {strong, mild}_0 x ten observations
#   20-49  mixture: mixture_0..9 x the three development observations; their
#          component weights are then written by slurm/references_mixture.sh
# Output: $RESULTS/single_factor_controlled_{strong,mild}/references, $RESULTS/mixture_controlled/references
source slurm/common.sh
triton_default

if [ "$IDX" -lt 20 ]; then
    SHIFTS=(strong mild)
    SHIFT=${SHIFTS[$((IDX / 10))]}
    IFS=, read -ra OBS_ALL <<< "$OBS10"
    python scripts/reference_bci.py --prior-type "$SHIFT" --prior-id 0 \
        --obs-seed "${OBS_ALL[$((IDX % 10))]}" --gh-deg 301 --theta-chunk 48 --backend jax \
        --particles 8000 --smc-seeds 0,1,2,3 --smc-mcmc 12 --num-samples 10000 \
        --out "$RESULTS/single_factor_controlled_$SHIFT"
else
    J=$((IDX - 20))
    IFS=, read -ra OBS_ALL <<< "$DEV3"
    python scripts/reference_bci.py --prior-type mixture --prior-id $((J % 10)) \
        --obs-seed "${OBS_ALL[$((J / 10))]}" --gh-deg 301 --theta-chunk 16 \
        --particles 8000 --smc-seeds 0,1,2,3 --smc-mcmc 12 --num-samples 10000 \
        --out "$RESULTS/mixture_controlled"
fi
