"""Stored component weights must not reach a sampler unless they are usable.

A path-space estimate that fails writes a record whose ``log_pi`` entries are
finite and equal.  Read without checking, that is an ordinary-looking uniform
component assignment, so the run produces normal-shaped samples from a failed
estimate.  These tests pin the guard at the real load entry points, through the
real JSON serialisation, not against a hand-built dict.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from spectra.weights import load_weight_record, weight_record_problems

REPO = Path(__file__).resolve().parents[1]


def _script(name):
    """Import a scripts/ module the way the job scripts run it."""
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(log_pi):
    a = np.asarray(log_pi, float)
    m = a[np.isfinite(a)].max()
    return list(a - (m + math.log(float(np.exp(a - m).sum()))))


USABLE = {
    "reference record, no status field": {"log_pi": _norm([0.0, 0.0])},
    "status ok": {"status": "ok", "log_pi": _norm([0.0, 0.0])},
    "one component with no mass": {"status": "ok", "log_pi": [0.0, -math.inf]},
    "extreme but valid weights": {"status": "ok", "log_pi": _norm([0.0, -27.6])},
}

UNUSABLE = {
    "failed estimate, placeholder logits": {"status": "nonfinite", "log_pi": [-700.0, -700.0]},
    "placeholder logits without a status": {"log_pi": [-700.0, -700.0]},
    "NaN": {"status": "ok", "log_pi": [float("nan"), 0.0]},
    "positive infinity": {"status": "ok", "log_pi": [float("inf"), -1.0]},
    "no finite mass anywhere": {"status": "ok", "log_pi": [-math.inf, -math.inf]},
    "not normalised": {"status": "ok", "log_pi": [0.0, 0.0]},
    "no log_pi at all": {"status": "ok"},
}


@pytest.mark.parametrize("name", sorted(USABLE))
def test_usable_records_are_accepted(name):
    assert weight_record_problems(USABLE[name]) == []


@pytest.mark.parametrize("name", sorted(UNUSABLE))
def test_unusable_records_are_refused(name):
    assert weight_record_problems(UNUSABLE[name]) != []


def test_component_count_mismatch_is_refused():
    ok = {"status": "ok", "log_pi": _norm([0.0, 0.0])}
    assert weight_record_problems(ok, expected_k=2) == []
    assert weight_record_problems(ok, expected_k=4) != []


def test_load_weight_record_exits_on_a_failed_estimate(tmp_path):
    f = tmp_path / "pathspace.json"
    f.write_text(json.dumps({"status": "nonfinite", "log_pi": [-700.0, -700.0],
                             "pi": [float("nan"), float("nan")]}))
    with pytest.raises(SystemExit):
        load_weight_record(f, expected_k=2)


def test_k_scaling_sampler_refuses_a_failed_estimate(tmp_path):
    """The entry point the K-scaling runner calls."""
    sk = _script("sample_k_scaling")
    k, obs, model = 2, 1000000, "0"
    pid = sk.KG.prior_id(k)
    d = tmp_path / "weights"
    d.mkdir()
    name = f"pathspace_{sk.KG.TASK}_K{k}_p{pid}_o{obs}_model{model}.json"
    (d / name).write_text(json.dumps({"status": "nonfinite",
                                      "log_pi": [-700.0, -700.0],
                                      "pi": [float("nan"), float("nan")]}))
    with pytest.raises(SystemExit):
        sk.load_weights("pathspace", tmp_path, k, obs, model)


def test_k_scaling_sampler_accepts_a_good_estimate(tmp_path):
    sk = _script("sample_k_scaling")
    k, obs, model = 2, 1000000, "0"
    pid = sk.KG.prior_id(k)
    d = tmp_path / "weights"
    d.mkdir()
    name = f"pathspace_{sk.KG.TASK}_K{k}_p{pid}_o{obs}_model{model}.json"
    log_pi = _norm([0.0, -1.0])
    (d / name).write_text(json.dumps({"status": "ok", "log_pi": log_pi,
                                      "pi": list(np.exp(log_pi))}))
    got, prov = sk.load_weights("pathspace", tmp_path, k, obs, model)
    assert np.allclose(got, log_pi)
    assert prov["weight_source"] == "pathspace"
