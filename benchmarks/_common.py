"""Shared benchmark plumbing: environment metadata and robust timing."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).parent / "results"


def metadata() -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    import eigentune

    try:
        import triton

        triton_v = triton.__version__
    except ImportError:
        triton_v = None
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "git_sha": sh(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"]),
        "eigentune": eigentune.__version__,
        "torch": torch.__version__,
        "triton": triton_v,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "gpu": dev,
        "python": platform.python_version(),
        "date": time.strftime("%Y-%m-%d"),
    }


def time_gpu(fn, warmup: int = 10, reps: int = 7, inner: int = 50):
    """Median / p10 / p90 of the per-call time in microseconds, over ``reps`` timed loops of ``inner`` calls.

    Inclusive of CPU launch overhead (an eager loop), which is what an eager user pays.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000 / inner)
    times.sort()
    return {"median_us": statistics.median(times), "p10_us": times[0], "p90_us": times[-1]}


def time_graph(fn, warmup: int = 5, reps: int = 7, inner: int = 50):
    """Per-call GPU time with launch overhead removed: ``inner`` calls captured in one CUDA graph."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000 / inner)
    times.sort()
    return {"median_us": statistics.median(times), "p10_us": times[0], "p90_us": times[-1]}


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
