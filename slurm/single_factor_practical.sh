#!/bin/bash
#SBATCH --job-name=spectra-k1-practical
#SBATCH --array=0-29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Table 1, single factor: PriorGuide and Spectra at the practical configuration
# (25, 8), BCI at (250, 0), on {mild, strong}_0 x ten observations.  One array
# element per (task, shift, model).  Scored against the references of the
# controlled study (slurm/references_single_factor.sh, slurm/references_bci.sh).
# Output: $RESULTS/single_factor_practical/samples
source slurm/common.sh
triton_off

TASKS=(bav gaussian_linear gaussian_linear_high slcp two_moons)
SHIFTS=(mild strong)
TASK=${TASKS[$((IDX / 6))]}
SHIFT=${SHIFTS[$(((IDX / 3) % 2))]}
MID=$((IDX % 3))
if [ "$TASK" = bav ]; then CONFIG=250:0; else CONFIG=25:8; fi

link_references single_factor_practical "single_factor_controlled_$SHIFT" "$TASK" "${SHIFT}_0"
python scripts/sample_single_factor.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" "$MID")" \
    --model-id "$MID" --prior-type "$SHIFT" --prior-id 0 --obs-seeds "$OBS10" --sampler-seeds 0,1,2 \
    --configs "$CONFIG" --methods pg_exact,spectra --num-samples 1000 --skip-existing \
    --out "$RESULTS/single_factor_practical"
