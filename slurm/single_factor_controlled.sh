#!/bin/bash
#SBATCH --job-name=spectra-k1-controlled
#SBATCH --array=0-29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Single-factor controlled comparison at (N, L) = (100, 0): Base, PriorGuide,
# PG-FullCov and Spectra (sampler seeds 0-2) and SIR/SNIS with 1k, 5k and 20k
# banks, on {strong, mild}_0 x ten observations.  One array element per
# (shift, task, model).  BCI ran with the default XLA flags.
# Output: $RESULTS/single_factor_controlled_{strong,mild}/samples
source slurm/common.sh

SHIFTS=(strong mild)
TASKS=(slcp two_moons gaussian_linear gaussian_linear_high bav)
SHIFT=${SHIFTS[$((IDX / 15))]}
TASK=${TASKS[$(((IDX % 15) / 3))]}
MID=$((IDX % 3))

if [ "$TASK" = bav ]; then triton_default; else triton_off; fi
python scripts/sample_single_factor_baselines.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" "$MID")" \
    --model-id "$MID" --prior-type "$SHIFT" --prior-id 0 --obs-seeds "$OBS10" --sampler-seeds 0,1,2 \
    --num-samples 1000 --num-steps 100 --grid power --sir-budgets 1000,5000,20000 --timing-repeat 2 \
    --out "$RESULTS/single_factor_controlled_$SHIFT"
