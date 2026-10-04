#!/bin/bash
#SBATCH --job-name=spectra-anisotropy
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Anisotropy boundary on Two Moons (scripts/anisotropy_config.py) at (25, 8), all
# ten observations, sampler seeds 0-2.  One array element per (model, unit):
#   unit 0     gamma = 1 control on mild_20: PriorGuide and Spectra (K = 1 exact);
#   units 1-4  one anisotropic cell 62..65: path-space weights over every compiled
#              component, then PriorGuide with the exact anisotropic ratio and
#              Spectra with the reference and the path-space weights.
# References: slurm/references_anisotropy.sh.  SIR: slurm/anisotropy_sir.sh.
# Output: $RESULTS/anisotropy/{samples,weights}
source slurm/common.sh
triton_off

OUT="$RESULTS/anisotropy"
UNITS=(k1 62 63 64 65)
MID=$((IDX / 5))
UNIT=${UNITS[$((IDX % 5))]}
CKPT=$(checkpoint two_moons "$MID")

if [ "$UNIT" = k1 ]; then
    python scripts/sample_single_factor.py --task two_moons --checkpoint "$CKPT" \
        --model-id "$MID" --prior-type mild --prior-id 20 --obs-seeds "$OBS10" \
        --sampler-seeds 0,1,2 --configs 25:8 --methods pg_exact,spectra \
        --num-samples 1000 --skip-existing --out "$OUT"
else
    python scripts/anisotropy_weights.py --cells "$UNIT" --model-id "$MID" --checkpoint "$CKPT" \
        --obs-seeds "$OBS10" --skip-existing --out "$OUT"
    python scripts/sample_anisotropy.py --cells "$UNIT" --model-id "$MID" --checkpoint "$CKPT" \
        --obs-seeds "$OBS10" --sampler-seeds 0,1,2 --skip-existing --out "$OUT"
fi
