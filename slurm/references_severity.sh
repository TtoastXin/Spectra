#!/bin/bash
#SBATCH --job-name=spectra-refs-severity
#SBATCH --array=0-9
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Reference posteriors of the shift-severity cells mild_{50,40,30,25,20}
# (scripts/severity_config.py), one array element per observation (CPU).  Two
# Moons and OUP use all ten observations; SLCP uses 1000003-1000005 (elements 0-2).
# Output: $RESULTS/severity/references
source slurm/common.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

OUT="$RESULTS/severity"
IFS=, read -ra OBS_ALL <<< "$OBS10"
OBS=${OBS_ALL[$IDX]}
for PID in 50 40 30 25 20; do
    CHECK=()
    if [ "$PID" = 50 ]; then CHECK=(--compare-rejection); fi
    python scripts/reference_posterior.py --task two_moons --prior-type mild --prior-id "$PID" \
        --obs-seed "$OBS" --method quadrature --levels 1024,2048,4096 --num-samples 10000 \
        ${CHECK[@]+"${CHECK[@]}"} --out "$OUT"
    python scripts/reference_posterior.py --task oup --prior-type mild --prior-id "$PID" \
        --obs-seed "$OBS" --method quadrature --levels 1024,2048,4096 --num-samples 10000 \
        --out "$OUT"
done
if [ "$IDX" -lt 3 ]; then
    SLCP_OBS=(1000003 1000004 1000005)
    for PID in 50 40 30 25 20; do
        python scripts/reference_posterior.py --task slcp --prior-type mild --prior-id "$PID" \
            --obs-seed "${SLCP_OBS[$IDX]}" --method smc --particles 40000 --smc-seeds 0,1,2,3 \
            --smc-mcmc 48 --num-samples 10000 --out "$OUT"
    done
fi
