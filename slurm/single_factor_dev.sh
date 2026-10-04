#!/bin/bash
#SBATCH --job-name=spectra-k1-dev
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Development sweep of the accuracy-compute figure: PriorGuide, PG-FullCov and
# Spectra at six (N, L) configurations, model 0, strong_0 x the three development
# observations.  One array element per task.
# Output: $RESULTS/single_factor_dev/samples
source slurm/common.sh
triton_off

TASKS=(two_moons slcp gaussian_linear_high)
TASK=${TASKS[$IDX]}

link_references single_factor_dev single_factor_controlled_strong "$TASK" strong_0
python scripts/sample_single_factor.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" 0)" --model-id 0 \
    --prior-type strong --prior-id 0 --obs-seeds "$DEV3" --sampler-seeds 0,1,2 \
    --configs 25:0,25:8,50:2,100:0,100:2,250:0 --methods pg_exact,a_full,spectra --num-samples 1000 \
    --out "$RESULTS/single_factor_dev"
