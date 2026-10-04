#!/bin/bash
#SBATCH --job-name=spectra-k1-oup
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%A_%a.out
# OUP single-factor rows on {mild, strong}_0 x ten observations, one array element
# per (shift, model): the (100, 0) controlled rows with SIR/SNIS, then PriorGuide
# and Spectra at the practical configuration (25, 8).
# Output: $RESULTS/single_factor_oup/samples
source slurm/common.sh
triton_off

SHIFTS=(mild strong)
SHIFT=${SHIFTS[$((IDX / 3))]}
MID=$((IDX % 3))
CKPT=$(checkpoint oup "$MID")
OUT="$RESULTS/single_factor_oup"

python scripts/sample_single_factor_baselines.py --task oup --checkpoint "$CKPT" --model-id "$MID" \
    --prior-type "$SHIFT" --prior-id 0 --obs-seeds "$OBS10" --sampler-seeds 0,1,2 --num-samples 1000 \
    --num-steps 100 --grid power --sir-budgets 1000,5000,20000 --timing-repeat 2 --out "$OUT"
python scripts/sample_single_factor.py --task oup --checkpoint "$CKPT" --model-id "$MID" \
    --prior-type "$SHIFT" --prior-id 0 --obs-seeds "$OBS10" --sampler-seeds 0,1,2 --configs 25:8 \
    --methods pg_exact,spectra --num-samples 1000 --skip-existing --out "$OUT"
