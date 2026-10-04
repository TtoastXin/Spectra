"""The K=2 regression command must fail when the numbers do not reproduce.

Given its seed set the path-space estimator is deterministic: re-running it in
one process reproduces every ``log Z`` exactly.  The regression command is
therefore an exact-reproduction gate, and a run whose numbers differ must exit
non-zero rather than print the difference and continue.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _mod():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("_test_kw", REPO / "scripts" / "k_scaling_weights.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Args:
    regression_rtol = 1e-6
    regression_weights = None
    checkpoint = None


BUDGET = {"seeds": [0], "trajectories": 8, "steps": 4, "grid": "power", "atoms": 2}


def _stored(log_z, pi):
    return {"budget": dict(BUDGET),
            "per_seed": [{"seed": 0, "delta": float(log_z[0][0] - log_z[0][1]),
                          "atoms": [{"log_z": v} for v in row]} for row in log_z],
            "pi": list(pi)}


def _fresh(log_z, pi):
    return {"budget": dict(BUDGET), "log_z_per_seed": log_z, "pi": list(pi),
            "wall_clock_s": 0.0}


def _run(monkeypatch, tmp_path, stored, fresh):
    mod = _mod()
    w = tmp_path / "stored.json"
    w.write_text(json.dumps(stored))
    monkeypatch.setattr(mod, "pathspace_weights", lambda *a, **kw: fresh)
    monkeypatch.setattr(mod, "build_prior_shift", lambda *a, **kw: object())
    monkeypatch.setattr(mod.atoms_mod, "atoms_from_shift", lambda *a, **kw: object())
    monkeypatch.setattr(mod.benchmark, "load_observation", lambda *a, **kw: (None, None))
    args = _Args()
    args.regression_weights = str(w)
    return mod.regression_check(lambda p: object(), args)


def test_identical_numbers_pass(monkeypatch, tmp_path):
    lz = [[1000.0, 900.0]]
    res = _run(monkeypatch, tmp_path, _stored(lz, [0.7, 0.3]), _fresh(lz, [0.7, 0.3]))
    assert res["budget_matches"] and res["passed"]


def test_wrong_log_z_fails(monkeypatch, tmp_path):
    """Same budget, log Z off by 100 nats."""
    res = _run(monkeypatch, tmp_path,
               _stored([[0.0, 0.0]], [0.5, 0.5]),
               _fresh([[100.0, 0.0]], [1.0, 0.0]))
    assert res["budget_matches"], "both records use the same budget"
    assert not res["passed"]
    assert not res["log_z_matches"]
    assert not res["pi_matches"]


def test_wrong_budget_fails(monkeypatch, tmp_path):
    lz = [[1000.0, 900.0]]
    fresh = _fresh(lz, [0.7, 0.3])
    fresh["budget"] = {**BUDGET, "steps": 999}
    res = _run(monkeypatch, tmp_path, _stored(lz, [0.7, 0.3]), fresh)
    assert not res["budget_matches"]
    assert not res["passed"]


def test_tolerance_is_relative_to_the_log_z_scale(monkeypatch, tmp_path):
    """log Z is order 1e5 on the real cells, so the window must scale with it."""
    lz = [[-240949.8573, -90273.0333]]
    near = [[-240949.8574, -90273.0333]]          # 1e-4 absolute on a 2.4e5 value
    res = _run(monkeypatch, tmp_path, _stored(lz, [0.6, 0.4]), _fresh(near, [0.6, 0.4]))
    assert res["passed"], res
    far = [[-240949.8573 + 10.0, -90273.0333]]    # 10 nats: a real regression
    res = _run(monkeypatch, tmp_path, _stored(lz, [0.6, 0.4]), _fresh(far, [0.6, 0.4]))
    assert not res["passed"]


def test_pi_deviation_alone_fails(monkeypatch, tmp_path):
    lz = [[1000.0, 900.0]]
    res = _run(monkeypatch, tmp_path, _stored(lz, [0.7, 0.3]), _fresh(lz, [0.71, 0.29]))
    assert res["log_z_matches"] and not res["pi_matches"] and not res["passed"]

# ---------------------------------------------------------------- the CLI --
#
# The tests above exercise ``regression_check``.  They would still pass if the
# exit branch in ``main`` were removed, so that the command printed a difference
# and still succeeded.  These go through ``main`` so the parser, the JSON write
# and the exit code are covered.


def _run_main(monkeypatch, tmp_path, stored, fresh, extra=()):
    mod = _mod()
    w = tmp_path / "stored.json"
    w.write_text(json.dumps(stored))
    monkeypatch.setattr(mod, "pathspace_weights", lambda *a, **kw: fresh)
    monkeypatch.setattr(mod, "build_prior_shift", lambda *a, **kw: object())
    monkeypatch.setattr(mod.atoms_mod, "atoms_from_shift", lambda *a, **kw: object())
    monkeypatch.setattr(mod.benchmark, "load_observation", lambda *a, **kw: (None, None))
    monkeypatch.setattr(mod, "load_checkpoint", lambda *a, **kw: object())
    monkeypatch.setattr(mod, "Backbone", lambda *a, **kw: object())
    monkeypatch.setattr(mod, "upstream_nn", lambda *a, **kw: None)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv",
                        ["k_scaling_weights.py", "--regression",
                         "--regression-weights", str(w), "--out", str(out), *extra])
    return mod, out


def test_cli_exits_zero_when_the_numbers_reproduce(monkeypatch, tmp_path):
    lz = [[1000.0, 900.0]]
    mod, out = _run_main(monkeypatch, tmp_path,
                         _stored(lz, [0.7, 0.3]), _fresh(lz, [0.7, 0.3]))
    assert mod.main() == 0
    rec = json.loads((out / "manifests" / "regression.json").read_text())
    assert rec["passed"] is True


def test_cli_exits_non_zero_when_log_z_is_wrong(monkeypatch, tmp_path):
    """A log Z difference must make the command exit non-zero, not only be reported."""
    mod, out = _run_main(monkeypatch, tmp_path,
                         _stored([[0.0, 0.0]], [0.5, 0.5]),
                         _fresh([[100.0, 0.0]], [1.0, 0.0]))
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code != 0
    rec = json.loads((out / "manifests" / "regression.json").read_text())
    assert rec["budget_matches"] and rec["passed"] is False


def test_cli_exits_non_zero_on_a_budget_mismatch(monkeypatch, tmp_path):
    lz = [[1000.0, 900.0]]
    fresh = _fresh(lz, [0.7, 0.3])
    fresh["budget"] = {**BUDGET, "steps": 999}
    mod, _ = _run_main(monkeypatch, tmp_path, _stored(lz, [0.7, 0.3]), fresh)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code != 0


def test_tolerance_is_element_wise(monkeypatch, tmp_path):
    """A near-zero component must not inherit the window of a large one.

    With one tolerance scaled by the largest |log Z| in the array, a 0.5 nat
    error on a component whose own log Z is 0 would pass.
    """
    stored = _stored([[-1e6, 0.0]], [0.0, 1.0])
    fresh = _fresh([[-1e6, 0.5]], [0.0, 1.0])
    res = _run(monkeypatch, tmp_path, stored, fresh)
    assert res["budget_matches"]
    assert not res["log_z_matches"], res
    assert not res["passed"]
