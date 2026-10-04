#!/bin/bash
#SBATCH --job-name=spectra-timing
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Steady-state online timing on one GPU, model 0 (compilation excluded, median of
# synchronized repeats).  Run on an otherwise idle node.
#   0-4  single factor, strong_0, observation 1000000: Two Moons, SLCP, GL-20D at
#        the six development configurations, GL-10D at (25, 8) and (100, 0), BCI
#        at (250, 0) and (100, 0)
#   5    mixture first-use cost stages, two prior-observation pairs per task
#        (Two Moons with the Triton GEMM emitter disabled, the others default)
#   6    mixture online stage of PriorGuide-VJP, same pairs and flags
#   7    component-count study: online time against K
# Output: $RESULTS/timing/{single_factor,mixture,mixture_vjp}, $RESULTS/k_scaling
source slurm/common.sh

MIXTURE_TASKS=(two_moons oup slcp gaussian_linear gaussian_linear_high)
case "$IDX" in
    0|1|2|3|4)
        TASKS=(two_moons slcp gaussian_linear_high gaussian_linear bav)
        ALL=25:0,25:8,50:2,100:0,100:2,250:0
        CONFIGS=("$ALL" "$ALL" "$ALL" 25:8,100:0 250:0,100:0)
        TASK=${TASKS[$IDX]}
        triton_off
        python scripts/timing_single_factor.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" 0)" \
            --model-id 0 --prior-type strong --prior-id 0 --obs-seed 1000000 --num-samples 1000 \
            --configs "${CONFIGS[$IDX]}" --repeats 5 --out "$RESULTS/timing/single_factor" ;;
    5)
        for TASK in "${MIXTURE_TASKS[@]}"; do
            if [ "$TASK" = two_moons ]; then triton_off; else triton_default; fi
            python scripts/timing_mixture.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" 0)" \
                --model-id 0 --pairs 0:1000000,1:1000001 --repeats 5 --ps-repeats 3 \
                --out "$RESULTS/timing/mixture"
        done ;;
    6)
        for TASK in "${MIXTURE_TASKS[@]}"; do
            if [ "$TASK" = two_moons ]; then triton_off; else triton_default; fi
            python scripts/timing_mixture_vjp.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" 0)" \
                --model-id 0 --pairs 0:1000000,1:1000001 --repeats 5 \
                --out "$RESULTS/timing/mixture_vjp"
        done ;;
    7)
        triton_off
        python scripts/timing_k_scaling.py --checkpoint "$(checkpoint two_moons 0)" --model-id 0 \
            --obs-seed 1000000 --repeats 5 --out "$RESULTS/k_scaling" ;;
esac
