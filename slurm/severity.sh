#!/bin/bash
#SBATCH --job-name=spectra-severity
#SBATCH --array=0-44
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Fixed-centre shift-severity sweep (scripts/severity_config.py): PriorGuide and
# Spectra at (25, 8) on mild_{50,40,30,25,20}, one array element per
# (task, width, model).  Two Moons and OUP use all ten observations, SLCP
# 1000003-1000005.  References: slurm/references_severity.sh.
# Output: $RESULTS/severity/samples
source slurm/common.sh
triton_off

TASKS=(two_moons oup slcp)
PRIOR_IDS=(50 40 30 25 20)
TASK=${TASKS[$((IDX / 15))]}
PID=${PRIOR_IDS[$(((IDX / 3) % 5))]}
MID=$((IDX % 3))
if [ "$TASK" = slcp ]; then OBS=1000003,1000004,1000005; else OBS=$OBS10; fi

python scripts/sample_single_factor.py --task "$TASK" --checkpoint "$(checkpoint "$TASK" "$MID")" \
    --model-id "$MID" --prior-type mild --prior-id "$PID" --obs-seeds "$OBS" --sampler-seeds 0,1,2 \
    --configs 25:8 --methods pg_exact,spectra --num-samples 1000 --skip-existing \
    --out "$RESULTS/severity"
