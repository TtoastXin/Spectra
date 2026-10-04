#!/bin/bash
#SBATCH --job-name=spectra-k64
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Correlated-Gaussian prior, K = 64 dictionary (mixture_21), one array element per
# model: Path-Space component weights on the three observations, then Spectra
# (100, 0), sampler seeds 0-2, with the reference weights and with the Path-Space
# weights (scripts/correlated_k64_config.py).  Needs slurm/correlated_prior.sh.
# Output: $RESULTS/correlated_prior/{weights,reference,pathspace}
source slurm/common.sh
triton_off

MID=$IDX
ROOT="$RESULTS/correlated_prior"
CKPT=$(checkpoint two_moons "$MID")

python scripts/correlated_k64_weights.py --model-id "$MID" --checkpoint "$CKPT" \
    --obs-seeds 1000003,1000005,1000010 --skip-existing --out "$ROOT"
for OBS in 1000003 1000005 1000010; do
    python scripts/sample_correlated.py --prior-id 21 --obs-seed "$OBS" --checkpoint "$CKPT" \
        --model-id "$MID" --seeds 0,1,2 \
        --weights "$ROOT/references/two_moons/mixture_21/obs_${OBS}_weights.json" \
        --out "$ROOT/reference"
    python scripts/sample_correlated.py --prior-id 21 --obs-seed "$OBS" --checkpoint "$CKPT" \
        --model-id "$MID" --seeds 0,1,2 \
        --weights "$ROOT/weights/pathspace_two_moons_p21_o${OBS}_model${MID}.json" \
        --out "$ROOT/pathspace"
done
