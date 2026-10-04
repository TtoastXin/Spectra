"""Small helpers shared by the experiment scripts: argument parsing and run provenance."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import jax


def parse_configs(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        n, l = part.split(":")
        out.append((int(n), int(l)))
    return out


def parse_ids(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def peak_gpu_memory_mb():
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    return None if peak is None else round(peak / 2**20, 1)


def git_state(repo: Path) -> dict:
    def run(cmd):
        return subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": run(["git", "rev-parse", "HEAD"]) or "no-commit",
            "dirty": run(["git", "status", "--short"])}
