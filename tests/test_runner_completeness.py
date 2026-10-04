"""A pair is finished only when its samples exist, not when its manifests do.

A tree of manifests with no ``.npz`` has nothing to score.  These tests check
the completeness rule in the functions the runners use: a pair with no output,
or with only its ``.json`` manifest, counts as missing.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _script(name):
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(f"_cmpl_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")


# ------------------------------------------------------ expected_outputs ----


def test_expected_outputs_requires_samples_and_manifest(tmp_path):
    sm = _script("sample_mixture")
    want = sm.expected_outputs(tmp_path, "two_moons", "0", 0, 1000000,
                               ["pg"], [0], "steps4_L0_power")
    files = want[("pg", 0)]
    assert len(files) == 2
    assert {f.suffix for f in files} == {".npz", ".json"}


def test_a_manifest_without_samples_is_not_complete(tmp_path):
    """A manifest without its ``.npz`` counts as missing until the samples exist."""
    sm = _script("sample_mixture")
    tag = "steps4_L0_power"
    want = sm.expected_outputs(tmp_path, "two_moons", "0", 0, 1000000,
                               ["pg"], [0], tag)
    npz, js = want[("pg", 0)]
    _touch(js)                                   # manifest only
    missing = [k for k, fs in want.items() if not all(f.is_file() for f in fs)]
    assert missing == [("pg", 0)], "a JSON-only row must count as missing"
    npz.write_bytes(b"")                         # now the samples too
    missing = [k for k, fs in want.items() if not all(f.is_file() for f in fs)]
    assert missing == []


def test_controlled_runner_counts_the_sir_row_once_per_pair(tmp_path):
    """SIR is written once per pair, the sampler rows once per (row, seed)."""
    src = (REPO / "scripts" / "mixture_weights_and_controlled.py").read_text()
    assert 'want[f"sir_n{args.sir_bank}"] = files(f"sir_n{args.sir_bank}")' in src
    assert "if args.sir_bank > 0:" in src
    assert "for name in rows_want for seed in sampler_seeds" in src


# ----------------------------------------------------------- load_frozen ---


def _frozen_tree(root, *, direct, pathspace=None, reference=None, pg_var=0.35):
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "status" / "two_moons" / "model_0").mkdir(parents=True, exist_ok=True)
    good = {"status": "ok", "log_pi": [-math.log(2.0)] * 2, "pi": [0.5, 0.5]}
    (root / "weights" / "direct_two_moons_p0_o1000000_model0.json").write_text(
        json.dumps(direct))
    (root / "weights" / "pathspace_two_moons_p0_o1000000_model0.json").write_text(
        json.dumps(pathspace if pathspace is not None else good))
    status = {"complete": True, "pg_total_variance": pg_var}
    if reference is not None:
        status["reference_weights"] = reference
    (root / "status" / "two_moons" / "model_0" / "mixture_0_obs_1000000.json").write_text(
        json.dumps(status))


GOOD = {"status": "ok", "log_pi": [-math.log(2.0)] * 2, "pi": [0.5, 0.5]}


def test_two_component_weights_are_accepted(tmp_path):
    sm = _script("sample_mixture")
    _frozen_tree(tmp_path, direct=GOOD, reference={**GOOD, "path": "x", "sha256_16": "y"})
    got = sm.load_frozen(tmp_path, "two_moons", 0, 1000000, "0")
    assert got["problems"] == []
    assert set(got["weights"]) == {"eamt_ref_direct", "eamt_ref_pathspace",
                                   "eamt_ref_reference"}


def test_a_single_component_record_is_refused(tmp_path):
    """A K=1 weight record must not be read into a K=2 run."""
    sm = _script("sample_mixture")
    _frozen_tree(tmp_path, direct={"status": "ok", "log_pi": [0.0], "pi": [1.0]})
    got = sm.load_frozen(tmp_path, "two_moons", 0, 1000000, "0")
    assert "eamt_ref_direct" not in got["weights"]
    assert any("expected 2" in p for p in got["problems"]), got["problems"]


def test_a_malformed_reference_record_is_refused(tmp_path):
    sm = _script("sample_mixture")
    _frozen_tree(tmp_path, direct=GOOD,
                 reference={"log_pi": [0.0], "pi": [1.0], "path": "x", "sha256_16": "y"})
    got = sm.load_frozen(tmp_path, "two_moons", 0, 1000000, "0")
    assert "eamt_ref_reference" not in got["weights"]
    assert any("eamt_ref_reference" in p for p in got["problems"]), got["problems"]


@pytest.mark.parametrize("bad,why", [
    ({"status": "ok", "log_pi": [[0.0], [0.0]]}, "1-D"),
    ({"status": "ok", "log_pi": 0.0}, "1-D"),
    ({"status": "ok", "log_pi": []}, "empty"),
    ({"status": "ok", "log_pi": [-math.log(2.0)] * 2, "pi": [0.9, 0.1]}, "exp"),
])
def test_malformed_shapes_give_a_readable_refusal(tmp_path, bad, why):
    """Each of these yields a readable problem instead of an IndexError or a pass."""
    from spectra.weights import weight_record_problems
    problems = weight_record_problems(bad, expected_k=2)
    assert problems, bad
    assert any(why in p for p in problems), problems
