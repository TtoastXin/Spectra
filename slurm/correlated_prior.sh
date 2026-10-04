#!/bin/bash
#SBATCH --job-name=spectra-correlated
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Correlated-Gaussian prior on Two Moons, observations 1000003, 1000005, 1000010,
# one after another (every observation rewrites the same compiled prior files).
#  1. Compile the correlated target into isotropic mixtures with K = 27 and 64
#     components (mixture_20, mixture_21) and compute the compiled-target
#     references and weights (CPU).  The compiled priors ship in data/; the ones
#     written here must be identical to them.
#  2. Base, PriorGuide and PG-FullCov with the true covariance, and Spectra with
#     the reference weights, model 0, (100, 0), sampler seeds 0-2.
# Then slurm/correlated_k64.sh.
# Output: $RESULTS/correlated_prior/{compilation,references,benchmark_e2e}
source slurm/common.sh

ROOT="$RESULTS/correlated_prior"
for OBS in 1000003 1000005 1000010; do
    JAX_PLATFORMS=cpu python scripts/compile_correlated_prior.py --task two_moons --obs-seed "$OBS" \
        --donor-prior-id 2 --grid-n 4096 --local-data "$ROOT/data" --out "$ROOT/compilation"
    for PID in 20 21; do
        cmp "$ROOT/data/priors/two_moons/mixture_$PID.json" "data/priors/two_moons/mixture_$PID.json"
        cmp "$ROOT/data/observations/two_moons/mixture_$PID/obs_$OBS.json" \
            "data/observations/two_moons/mixture_$PID/obs_$OBS.json"
    done

    triton_off
    for PID in 20 21; do
        python scripts/sample_correlated.py --task two_moons --prior-id "$PID" --obs-seed "$OBS" \
            --checkpoint "$(checkpoint two_moons 0)" --model-id 0 --num-samples 1000 --num-steps 100 \
            --seeds 0,1,2 --out "$ROOT/benchmark_e2e"
    done
    triton_default
done
