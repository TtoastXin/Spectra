#!/bin/bash
#SBATCH --job-name=spectra-refs-k1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Reference posteriors of the single-factor cells strong_0 and mild_0, one array
# element per observation (CPU).  Two Moons and OUP by grid quadrature, SLCP by
# tempered SMC against the exact likelihood, the Gaussian Linear tasks in closed
# form.  BCI is in slurm/references_bci.sh.
# Output: $RESULTS/single_factor_controlled_{strong,mild}/references (OUP: $RESULTS/single_factor_oup)
source slurm/common.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

IFS=, read -ra OBS_ALL <<< "$OBS10"
OBS=${OBS_ALL[$IDX]}

for SHIFT in strong mild; do
    OUT="$RESULTS/single_factor_controlled_$SHIFT"
    for TASK in gaussian_linear gaussian_linear_high; do
        python scripts/reference_closed_form.py --task "$TASK" --prior-type "$SHIFT" --prior-id 0 \
            --obs-seeds "$OBS" --num-samples 10000 --out "$OUT"
    done
    python scripts/reference_posterior.py --task two_moons --prior-type "$SHIFT" --prior-id 0 \
        --obs-seed "$OBS" --method quadrature --levels 1024,2048,4096 --num-samples 10000 \
        --compare-rejection --out "$OUT"
    python scripts/reference_posterior.py --task slcp --prior-type "$SHIFT" --prior-id 0 \
        --obs-seed "$OBS" --method smc --particles 40000 --smc-seeds 0,1,2,3 --smc-mcmc 48 \
        --num-samples 10000 --out "$OUT"
    python scripts/reference_posterior.py --task oup --prior-type "$SHIFT" --prior-id 0 \
        --obs-seed "$OBS" --method quadrature --levels 1024,2048,4096 --num-samples 10000 \
        --out "$RESULTS/single_factor_oup"
done
