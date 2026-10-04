#!/bin/bash
#SBATCH --job-name=spectra-k2-practical
#SBATCH --array=0-44
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Table 1, mixture: PriorGuide and Spectra with the Direct and Path-Space weights
# at the practical configuration (25, 8), reusing the weights and PriorGuide
# posterior-covariance banks estimated by slurm/mixture_controlled.sh.
#   0-14   (task, model) for Two Moons, OUP, SLCP, GL-10D, GL-20D
#   15-44  BCI (model, prior id)
# Only Two Moons ran with the Triton GEMM emitter disabled.
# Output: $RESULTS/mixture_practical/samples
source slurm/common.sh

WEIGHTS="$RESULTS/mixture_controlled"
OUT="$RESULTS/mixture_practical"
mkdir -p "$OUT"
ln -sfn "$WEIGHTS/references" "$OUT/references" 2>/dev/null \
    || [ "$(readlink "$OUT/references")" = "$WEIGHTS/references" ]
if [ "$IDX" -lt 15 ]; then
    TASKS=(two_moons oup slcp gaussian_linear gaussian_linear_high)
    TASK=${TASKS[$((IDX / 3))]}
    MID=$((IDX % 3))
    if [ "$TASK" = two_moons ]; then triton_off; else triton_default; fi
    python scripts/sample_mixture.py --task "$TASK" --model-id "$MID" \
        --checkpoint "$(checkpoint "$TASK" "$MID")" --weights-root "$WEIGHTS" --prior-ids 0-9 \
        --obs-seeds "$DEV3" --rows pg,eamt_ref_direct,eamt_ref_pathspace --sampler-seeds 0,1,2 \
        --config 25:8 --num-samples 1000 --skip-existing --out "$OUT"
else
    J=$((IDX - 15))
    MID=$((J / 10))
    triton_default
    python scripts/sample_mixture.py --task bav --model-id "$MID" \
        --checkpoint "$(checkpoint bav "$MID")" --weights-root "$WEIGHTS" --config 25:8 \
        --prior-ids $((J % 10)) --obs-seeds "$DEV3" --skip-existing --out "$OUT"
fi
