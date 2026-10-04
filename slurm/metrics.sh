#!/bin/bash
#SBATCH --job-name=spectra-metrics
#SBATCH --array=0-43
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Per-file metrics (C2ST, MMTV, moment errors) of every sampled study against its
# references (CPU), one array element per (study, task), written to
# <study>/metrics/<task>.csv.  --support-matched adds the rows
# scored against the reference restricted to the training support, where the
# reference has one; as in the paper's runs it is off for the BCI controlled rows.
# Output: $RESULTS/<study>/metrics/<task>.csv
source slurm/common.sh
export JAX_PLATFORMS=cpu

#        study                            task                  flags
ENTRIES=(
    "single_factor_controlled_strong  slcp                  --support-matched"
    "single_factor_controlled_strong  two_moons             --support-matched"
    "single_factor_controlled_strong  gaussian_linear       --support-matched"
    "single_factor_controlled_strong  gaussian_linear_high  --support-matched"
    "single_factor_controlled_strong  bav"
    "single_factor_controlled_mild    slcp                  --support-matched"
    "single_factor_controlled_mild    two_moons             --support-matched"
    "single_factor_controlled_mild    gaussian_linear       --support-matched"
    "single_factor_controlled_mild    gaussian_linear_high  --support-matched"
    "single_factor_controlled_mild    bav"
    "single_factor_oup                oup                   --support-matched"
    "single_factor_practical          two_moons             --support-matched"
    "single_factor_practical          slcp                  --support-matched"
    "single_factor_practical          bav                   --support-matched"
    "single_factor_practical          gaussian_linear       --support-matched"
    "single_factor_practical          gaussian_linear_high  --support-matched"
    "single_factor_dev                two_moons             --support-matched"
    "single_factor_dev                slcp                  --support-matched"
    "single_factor_dev                gaussian_linear_high  --support-matched"
    "single_factor_heldout            two_moons             --support-matched"
    "single_factor_heldout            slcp                  --support-matched"
    "single_factor_heldout            gaussian_linear_high  --support-matched"
    "single_factor_vjp                two_moons             --support-matched"
    "single_factor_vjp                slcp                  --support-matched"
    "single_factor_vjp                gaussian_linear_high  --support-matched"
    "mixture_controlled               two_moons             --support-matched"
    "mixture_controlled               oup                   --support-matched"
    "mixture_controlled               slcp                  --support-matched"
    "mixture_controlled               gaussian_linear       --support-matched"
    "mixture_controlled               gaussian_linear_high  --support-matched"
    "mixture_controlled               bav                   --support-matched"
    "mixture_practical                two_moons             --support-matched"
    "mixture_practical                oup                   --support-matched"
    "mixture_practical                slcp                  --support-matched"
    "mixture_practical                gaussian_linear       --support-matched"
    "mixture_practical                gaussian_linear_high  --support-matched"
    "mixture_practical                bav                   --support-matched"
    "severity                         two_moons             --support-matched"
    "severity                         oup                   --support-matched"
    "severity                         slcp                  --support-matched"
    "k_scaling                        two_moons             --support-matched"
    "correlated_prior                 two_moons"
    "anisotropy                       two_moons             --support-matched"
    "anisotropy_compiled_target       two_moons             --support-matched"
)
read -r STUDY TASK FLAGS <<< "${ENTRIES[$IDX]}"
ROOT="$RESULTS/$STUDY"
WORKERS=${SLURM_CPUS_PER_TASK:-1}

arviz_stamp
if [ "$STUDY" = correlated_prior ]; then
    for OBS in 1000003 1000005 1000010; do
        python scripts/score_correlated.py --root "$ROOT" --task two_moons --obs-seed "$OBS" \
            --prior-ids 20,21 --out "$ROOT/metrics/compiled_obs$OBS.csv"
    done
    python scripts/score_correlated_k64.py --root "$ROOT" --workers "$WORKERS" --self-test
    exit 0
fi

if [ "$STUDY" = anisotropy_compiled_target ]; then
    # Spectra's anisotropy rows scored against the compiled target instead of the
    # exact one: a scoring root that links the Spectra samples and puts the
    # compiled-target reference (mixture_<cell>) where the exact one (aniso_<cell>)
    # would be.
    SRC="$RESULTS/anisotropy"
    mkdir -p "$ROOT/samples/two_moons" "$ROOT/references/two_moons"
    for D in "$SRC"/samples/two_moons/spectra_ref_ms* "$SRC"/samples/two_moons/spectra_ps_ms*; do
        ln -sfn "$D" "$ROOT/samples/two_moons/$(basename "$D")"
    done
    for CELL in 62 63 64 65; do
        ln -sfn "$SRC/references/two_moons/mixture_$CELL" "$ROOT/references/two_moons/aniso_$CELL"
    done
fi

python scripts/compute_metrics.py --root "$ROOT" --task "$TASK" --out "$ROOT/metrics/$TASK.csv" \
    --workers "$WORKERS" --c2st-jobs 1 $FLAGS
