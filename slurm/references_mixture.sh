#!/bin/bash
#SBATCH --job-name=spectra-refs-k2
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Reference posteriors and reference component weights of the two-component
# mixture cells mixture_0..9 x the three development observations (CPU).
#   0-4  Two Moons, OUP, SLCP, GL-10D, GL-20D
#   5    BCI component weights, from the references of slurm/references_bci.sh
# Output: $RESULTS/mixture_controlled/references
source slurm/common.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

TASKS=(two_moons oup slcp gaussian_linear gaussian_linear_high bav)
TASK=${TASKS[$IDX]}
OUT="$RESULTS/mixture_controlled"

if [ "$TASK" = bav ]; then
    python scripts/reference_weights_bci.py --references "$OUT" --out "$OUT"
    exit 0
fi
IFS=, read -ra OBS_ALL <<< "$DEV3"
for PID in $(seq 0 9); do
    case "$TASK" in
        two_moons|oup)
            for OBS in "${OBS_ALL[@]}"; do
                python scripts/reference_posterior.py --task "$TASK" --prior-type mixture \
                    --prior-id "$PID" --obs-seed "$OBS" --method quadrature \
                    --levels 1024,2048,4096 --num-samples 10000 --out "$OUT"
            done ;;
        slcp)
            for OBS in "${OBS_ALL[@]}"; do
                python scripts/reference_posterior.py --task slcp --prior-type mixture \
                    --prior-id "$PID" --obs-seed "$OBS" --method smc --particles 40000 \
                    --smc-seeds 0,1,2,3 --smc-mcmc 48 --num-samples 10000 --out "$OUT"
            done ;;
        gaussian_linear|gaussian_linear_high)
            python scripts/reference_closed_form.py --task "$TASK" --prior-type mixture \
                --prior-id "$PID" --obs-seeds "$DEV3" --num-samples 10000 --seed 0 --out "$OUT"
            python scripts/reference_weights_gl.py --task "$TASK" --prior-id "$PID" \
                --obs-seeds "$DEV3" --out "$OUT" ;;
    esac
done
