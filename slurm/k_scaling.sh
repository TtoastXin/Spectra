#!/bin/bash
#SBATCH --job-name=spectra-kscaling
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Component-count study on Two Moons (scripts/k_scaling_config.py), K = 2, 4, 8, 16
# at (25, 8) x the three development observations.
#   0-2  model: PriorGuide and Spectra with the reference weights
#   3    model 0: Path-Space weights (after a regression check against a stored
#        K = 2 cell of slurm/mixture_controlled.sh), then Spectra with them
# References: slurm/references_k_scaling.sh.  Timing: slurm/timing.sh.
# Output: $RESULTS/k_scaling/{samples,weights}
source slurm/common.sh
triton_off

OUT="$RESULTS/k_scaling"
if [ "$IDX" -lt 3 ]; then
    python scripts/sample_k_scaling.py --ks 2,4,8,16 --obs-seeds "$DEV3" --model-id "$IDX" \
        --checkpoint "$(checkpoint two_moons "$IDX")" --sampler-seeds 0,1,2 --weights reference \
        --methods pg_exact,spectra_ref --num-samples 1000 --skip-existing --out "$OUT"
else
    CKPT=$(checkpoint two_moons 0)
    python scripts/k_scaling_weights.py --regression --out "$OUT" \
        --checkpoint "$CKPT" \
        --regression-weights "$RESULTS/mixture_controlled/weights/pathspace_two_moons_p0_o1000000_model0.json"
    python scripts/k_scaling_weights.py --ks 2,4,8,16 --obs-seeds "$DEV3" --model-id 0 \
        --checkpoint "$CKPT" --seeds 0,1,2 --trajectories 4096 --steps 400 --grid power \
        --skip-existing --out "$OUT"
    python scripts/sample_k_scaling.py --ks 2,4,8,16 --obs-seeds "$DEV3" --model-id 0 \
        --checkpoint "$CKPT" --sampler-seeds 0,1,2 --weights pathspace --methods spectra_ps \
        --num-samples 1000 --skip-existing --out "$OUT"
fi
