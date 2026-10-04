#!/bin/bash
#SBATCH --job-name=spectra-training-data
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Simformer training sets, 10,000 simulations per model seed 0, 1, 2 (CPU).
# Optional: the trained checkpoints ship in checkpoints/.
#
# Elements 0-4 draw the PriorGuide tasks with PriorGuide's own task code and need
# a PriorGuide checkout at commit b4852fc in $PRIORGUIDE_UPSTREAM.  Element 5 draws
# SLCP with this repository's simulator.
# Output: $RESULTS/training_data/<task>/sims10000_seed<m>.npz
source slurm/common.sh

TASKS=(two_moons gaussian_linear gaussian_linear_high oup bav slcp)
TASK=${TASKS[$IDX]}

if [ "$TASK" = slcp ]; then
    python scripts/prepare_slcp.py --step data --model-seeds 0,1,2 --num-sims 10000 \
        --out "$RESULTS"
else
    : "${PRIORGUIDE_UPSTREAM:?set PRIORGUIDE_UPSTREAM to a prior-guide checkout at commit b4852fc}"
    for SEED in 0 1 2; do
        python scripts/simulate_training_data.py --task "$TASK" --seed "$SEED" \
            --num-sims 10000 --out "$RESULTS/training_data"
    done
fi
