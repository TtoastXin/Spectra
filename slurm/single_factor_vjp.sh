#!/bin/bash
#SBATCH --job-name=spectra-k1-vjp
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# PriorGuide with the VJP implementation on strong_0.
#   0-8   (task, model): (25, 8) x ten observations
#   9-11  task: model 0, the other five development configurations x the three
#         development observations ((25, 8) is covered by 0-8)
# Output: $RESULTS/single_factor_vjp/samples
source slurm/common.sh
triton_off

TASKS=(two_moons slcp gaussian_linear_high)
if [ "$IDX" -lt 9 ]; then
    TASK=${TASKS[$((IDX / 3))]}; MID=$((IDX % 3)); OBS=$OBS10; CONFIGS=25:8
else
    TASK=${TASKS[$((IDX - 9))]}; MID=0; OBS=$DEV3; CONFIGS=25:0,50:2,100:0,100:2,250:0
fi

link_references single_factor_vjp single_factor_controlled_strong "$TASK" strong_0
python scripts/sample_single_factor.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" "$MID")" \
    --model-id "$MID" --prior-type strong --prior-id 0 --obs-seeds "$OBS" --sampler-seeds 0,1,2 \
    --configs "$CONFIGS" --methods pg_vjp --num-samples 1000 --skip-existing --out "$RESULTS/single_factor_vjp"
