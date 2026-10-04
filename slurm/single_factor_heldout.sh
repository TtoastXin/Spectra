#!/bin/bash
#SBATCH --job-name=spectra-k1-heldout
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Held-out check of the two configurations chosen on the development
# observations, (25, 8) and (100, 0): PriorGuide, PG-FullCov and Spectra on
# strong_0 x the seven held-out observations.  One array element per (task, model).
# Output: $RESULTS/single_factor_heldout/samples
source slurm/common.sh
triton_off

TASKS=(two_moons slcp gaussian_linear_high)
TASK=${TASKS[$((IDX / 3))]}
MID=$((IDX % 3))

link_references single_factor_heldout single_factor_controlled_strong "$TASK" strong_0
python scripts/sample_single_factor.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" "$MID")" \
    --model-id "$MID" --prior-type strong --prior-id 0 --obs-seeds "$HELD7" --sampler-seeds 0,1,2 \
    --configs 25:8,100:0 --methods pg_exact,a_full,spectra --num-samples 1000 --skip-existing \
    --out "$RESULTS/single_factor_heldout"
