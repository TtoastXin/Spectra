#!/bin/bash
#SBATCH --job-name=spectra-refs-anisotropy
#SBATCH --array=0-9
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Anisotropy boundary on Two Moons (scripts/anisotropy_config.py), one array
# element per observation (CPU):
#  0. element 0 only: rewrite the compiled cells mixture_{61..65} and
#     data/anisotropy/cells.json into a fresh directory; they must be
#     identical to the shipped ones;
#  1. exact vs compiled target on the grid (prior and posterior TV, refinement
#     2048 -> 4096) and the exact anisotropic reference samples aniso_{62..65};
#  2. reference posteriors of the gamma = 1 control mild_20 and of every compiled
#     anisotropic dictionary mixture_{62..65}, which also writes the reference
#     component weights of Spectra-Ref.
# Output: $RESULTS/anisotropy/{representation,references}
source slurm/common.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

OUT="$RESULTS/anisotropy"
IFS=, read -ra OBS_ALL <<< "$OBS10"
OBS=${OBS_ALL[$IDX]}

if [ "$IDX" = 0 ]; then
    rm -rf "$OUT/prepare"
    python scripts/prepare_anisotropy_priors.py --out "$OUT/prepare"
    cmp "$OUT/prepare/anisotropy/cells.json" data/anisotropy/cells.json
    for PID in 61 62 63 64 65; do
        cmp "$OUT/prepare/priors/two_moons/mixture_$PID.json" "data/priors/two_moons/mixture_$PID.json"
        for O in "${OBS_ALL[@]}"; do
            cmp "$OUT/prepare/observations/two_moons/mixture_$PID/obs_$O.json" \
                "data/observations/two_moons/mixture_$PID/obs_$O.json"
        done
    done
fi

python scripts/anisotropy_representation.py --obs-seed "$OBS" --root "$OUT"
python scripts/reference_posterior.py --task two_moons --prior-type mild --prior-id 20 \
    --obs-seed "$OBS" --method quadrature --levels 1024,2048,4096 --num-samples 10000 --out "$OUT"
for PID in 62 63 64 65; do
    python scripts/reference_posterior.py --task two_moons --prior-type mixture --prior-id "$PID" \
        --obs-seed "$OBS" --method quadrature --levels 1024,2048,4096 --num-samples 10000 --out "$OUT"
done
