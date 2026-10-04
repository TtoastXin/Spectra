"""The plumbing a re-run depends on: which model, which results root.

These pin behaviour that only shows up when a user follows the README and
redirects ``SPECTRA_CHECKPOINTS`` or ``SPECTRA_RESULTS`` at a directory of their
own.  Nothing here samples: the loaders are stubbed so the test captures the
path a runner would have opened.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _script(name):
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(f"_test_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Loaded(Exception):
    """Carries the path a stubbed loader was asked for."""

    def __init__(self, path):
        super().__init__(str(path))
        self.path = str(path)


def _capture_checkpoint(mod, monkeypatch, argv):
    def stub(path, *a, **kw):
        raise _Loaded(path)

    monkeypatch.setattr(mod, "load_checkpoint", stub)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_Loaded) as exc:
        mod.main()
    return exc.value.path


def test_k64_weights_default_checkpoint_is_the_shipped_one(monkeypatch):
    mod = _script("correlated_k64_weights")
    got = _capture_checkpoint(mod, monkeypatch,
                              ["correlated_k64_weights.py", "--model-id", "0"])
    assert got == str(REPO / "checkpoints" / "two_moons" / "model_0.pkl")


def test_k64_weights_honours_an_explicit_checkpoint(monkeypatch, tmp_path):
    """Without this the weights belong to one model and the samples to another."""
    mod = _script("correlated_k64_weights")
    other = tmp_path / "retrained" / "two_moons" / "model_0.pkl"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"")
    got = _capture_checkpoint(mod, monkeypatch,
                              ["correlated_k64_weights.py", "--model-id", "0",
                               "--checkpoint", str(other)])
    assert got == str(other)
    assert got != str(REPO / "checkpoints" / "two_moons" / "model_0.pkl")


def test_k_scaling_regression_honours_an_explicit_checkpoint(monkeypatch, tmp_path):
    mod = _script("k_scaling_weights")
    other = tmp_path / "retrained" / "model_0.pkl"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"")
    seen = []

    stored = tmp_path / "stored.json"
    stored.write_text(json.dumps({"status": "ok", "per_seed": [], "pi": [0.5, 0.5],
                                  "budget": {}}))

    def stub(path, *a, **kw):
        seen.append(str(path))
        raise _Loaded(path)

    monkeypatch.setattr(mod, "load_checkpoint", stub)
    monkeypatch.setattr(sys, "argv",
                        ["k_scaling_weights.py", "--regression",
                         "--checkpoint", str(other),
                         "--regression-weights", str(stored),
                         "--out", str(tmp_path / "out")])
    with pytest.raises(Exception):
        mod.main()
    assert seen and seen[0] == str(other)


def test_slurm_checkpoint_helper_follows_the_environment():
    """``SPECTRA_CHECKPOINTS`` is documented in README; the helper must honour it."""
    import subprocess
    script = ('cd "%s" && export SLURM_ARRAY_TASK_ID=0 SPECTRA_CHECKPOINTS=/tmp/retrained && '
              'source slurm/common.sh >/dev/null 2>&1 && checkpoint two_moons 0' % REPO)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.stdout.strip() == "/tmp/retrained/two_moons/model_0.pkl", out


def test_correlated_k64_job_passes_the_resolved_checkpoint_to_the_weights_step():
    job = (REPO / "slurm" / "correlated_k64.sh").read_text()
    weights_call = job.split("correlated_k64_weights.py", 1)[1].split("python", 1)[0]
    assert "--checkpoint" in weights_call, job
