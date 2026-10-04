#!/bin/bash
#SBATCH --job-name=spectra-train
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Train the 18 Simformer checkpoints (three model seeds per task) on the sets
# from slurm/training_data.sh.  Optional: the checkpoints behind the paper ship in
# checkpoints/; these go to $RESULTS/checkpoints so they never overwrite them.
#
# Two Moons trains with the Triton GEMM emitter disabled (in our environment the
# default XLA:GPU stack segfaults in its backward pass); the other tasks trained
# with the default flags.  See scripts/train_model.py on reproducing the shipped checkpoints.
source slurm/common.sh

TASKS=(two_moons slcp oup gaussian_linear gaussian_linear_high bav)
TASK=${TASKS[$((IDX / 3))]}
SEED=$((IDX % 3))

if [ "$TASK" = two_moons ]; then triton_off; else triton_default; fi
python scripts/train_model.py --data "$RESULTS/training_data/$TASK/sims10000_seed$SEED.npz" \
    --task "$TASK" --seed "$SEED" --out "$RESULTS/checkpoints"
