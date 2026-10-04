#!/bin/bash
#SBATCH --job-name=spectra-k2-controlled
#SBATCH --array=0-44
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Two-component mixture cells mixture_0..9 x the three development observations at
# (100, 0): estimates the Direct and Path-Space component weights, and samples
# Base, PriorGuide, PG-FullCov, SIR/SNIS (20k bank) and Spectra with the
# reference, Direct and Path-Space weights (sampler seeds 0-2).
#   0-14   (task, model) for Two Moons, OUP, SLCP, GL-10D, GL-20D
#   15-44  BCI (model, prior id)
# Only Two Moons ran with the Triton GEMM emitter disabled.  Needs the references
# and reference weights of slurm/references_mixture.sh and slurm/references_bci.sh.
# Output: $RESULTS/mixture_controlled/{samples,weights,status}
source slurm/common.sh

OUT="$RESULTS/mixture_controlled"
if [ "$IDX" -lt 15 ]; then
    TASKS=(two_moons oup slcp gaussian_linear gaussian_linear_high)
    TASK=${TASKS[$((IDX / 3))]}
    MID=$((IDX % 3))
    if [ "$TASK" = two_moons ]; then triton_off; else triton_default; fi
    python scripts/mixture_weights_and_controlled.py --task "$TASK" --model-id "$MID" \
        --checkpoint "$(checkpoint "$TASK" "$MID")" --prior-ids 0-9 --obs-seeds "$DEV3" \
        --ref-roots "$OUT" --skip-existing --out "$OUT"
else
    J=$((IDX - 15))
    MID=$((J / 10))
    triton_default
    python scripts/mixture_weights_and_controlled.py --task bav --model-id "$MID" \
        --checkpoint "$(checkpoint bav "$MID")" --prior-ids $((J % 10)) --obs-seeds "$DEV3" \
        --ref-roots "$OUT" --num-steps 100 --skip-existing --out "$OUT"
fi
