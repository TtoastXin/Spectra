# Shared settings for the job scripts in slurm/.
#
# Every job script is a SLURM array job.  Submit it from the repository root with
# the Python environment of requirements.txt activated (sbatch passes the
# environment on to the job), after creating the log directory:
#
#     mkdir -p logs && sbatch slurm/<script>.sh
#
# Clusters that do not pass the environment by default (SBATCH_EXPORT=NONE) need
# `sbatch --export=ALL`.
#
# Outside SLURM, run one array element at a time from the repository root:
#
#     SLURM_ARRAY_TASK_ID=0 bash slurm/<script>.sh
#
# The #SBATCH lines state the resources of the paper's runs (one NVIDIA A100 per
# GPU job); add the partition and account options of your cluster.
#
# Environment variables:
#   SPECTRA_RESULTS      output root (default: results/)
#   SPECTRA_CHECKPOINTS  checkpoints to sample from (default: the shipped checkpoints/)

set -euo pipefail

# sbatch starts the job in the directory it was submitted from
REPO=$PWD
if [ ! -f spectra/__init__.py ]; then
    echo "run the job scripts from the repository root" >&2
    exit 1
fi
RESULTS=${SPECTRA_RESULTS:-$REPO/results}
CHECKPOINTS=${SPECTRA_CHECKPOINTS:-$REPO/checkpoints}
IDX=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID to the array element to run}

# Observation seeds: all ten, the three development ones, and the seven held-out ones.
OBS10=1000000,1000001,1000002,1000003,1000004,1000005,1000008,1000009,1000010,1000012
DEV3=1000000,1000001,1000002
HELD7=1000003,1000004,1000005,1000008,1000009,1000010,1000012

checkpoint() {  # checkpoint <task> <model id>
    echo "$CHECKPOINTS/$1/model_$2.pkl"
}

# The paper's GPU jobs disabled XLA's Triton GEMM emitter, except where a script
# says otherwise (triton_default).  Kernel selection changes floating-point
# results, so reproducing a run exactly needs the same setting.
triton_off() {
    export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
}
triton_default() {
    unset XLA_FLAGS
}

# arviz, imported through sbi by the metrics, writes a once-a-day warning stamp
# into its cache directory on import; parallel metric workers race on creating it.
# Writing the stamp first avoids the race.
arviz_stamp() {
    python - <<'PY'
import datetime
import pathlib

from platformdirs import user_cache_dir

d = pathlib.Path(user_cache_dir("arviz", "arviz"))
d.mkdir(parents=True, exist_ok=True)
(d / "daily_warning").write_text(datetime.date.today().isoformat())
PY
}

# link_references <study> <source study> <task> <prior cell>: score a study
# against the references another study already computed for the same cell.
link_references() {
    local dst="$RESULTS/$1/references/$3"
    local src="$RESULTS/$2/references/$3/$4"
    mkdir -p "$dst"
    ln -sfn "$src" "$dst/$4" 2>/dev/null || [ "$(readlink "$dst/$4")" = "$src" ]
}
