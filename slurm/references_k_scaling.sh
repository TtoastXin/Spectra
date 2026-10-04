#!/bin/bash
#SBATCH --job-name=spectra-refs-kscaling
#SBATCH --array=0-2
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Reference posteriors and component weights of the Two Moons K-component cells
# mixture_{102,104,108,116} (K = 2, 4, 8, 16; scripts/k_scaling_config.py), one
# array element per development observation (CPU).
# Output: $RESULTS/k_scaling/references
source slurm/common.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

IFS=, read -ra OBS_ALL <<< "$DEV3"
for PID in 102 104 108 116; do
    python scripts/reference_posterior.py --task two_moons --prior-type mixture --prior-id "$PID" \
        --obs-seed "${OBS_ALL[$IDX]}" --method quadrature --levels 1024,2048,4096 \
        --num-samples 10000 --out "$RESULTS/k_scaling"
done
