"""Basis initialization: time, peak memory and accuracy of each SVD strategy on real weights.

    python benchmarks/initialization.py [--out NAME]

The reference is the exact rank-r reconstruction from ``torch.linalg.svd``. The error is
``||W_r(approx) - W_r(exact)||_F / ||W_r(exact)||_F`` plus the largest relative singular-value error.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import metadata, save  # noqa: E402
from _weights import remote_tensor  # noqa: E402

from eigentune import EigenTuneConfig  # noqa: E402
from eigentune.svd import compute_bases  # noqa: E402

REPO = "NousResearch/Llama-2-7b-hf"
WEIGHTS = [
    ("model.layers.16.self_attn.q_proj.weight", "q_proj 4096x4096"),
    ("model.layers.16.mlp.up_proj.weight", "up_proj 11008x4096"),
]
# (backend, settings, label): the converged randomized solver (the default) against the naive settings it replaces
STRATEGIES = [
    ("exact", {}, "exact"),
    ("randomized", {}, "randomized (converged, default)"),
    ("randomized", {"svd_oversampling": 8, "svd_niter": 4, "svd_tol": 0.0}, "randomized (niter=4, p=8)"),
    ("lowrank", {"svd_oversampling": 8, "svd_niter": 2}, "torch.svd_lowrank (niter=2)"),
]


def timed(fn, device, reps=3):
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="initialization")
    ap.add_argument("--ranks", type=int, nargs="*", default=[8, 16, 64])
    a = ap.parse_args()
    rows = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for key, label in WEIGHTS:
        W16 = remote_tensor(REPO, key)
        for r in a.ranks:
            ref = None
            for device in devices:
                W = W16.to(device)
                for backend, extra, slabel in STRATEGIES:
                    cfg = EigenTuneConfig(rank=r, svd_backend=backend, **extra)
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                    secs, b = timed(lambda: compute_bases(W, cfg), device, reps=3 if device == "cuda" else 2)
                    peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device == "cuda" else None
                    if ref is None and backend == "exact":
                        ref = ((b.U * b.S) @ b.Vh).cpu(), b.S.cpu()
                    approx = ((b.U * b.S) @ b.Vh).cpu()
                    err = ((approx - ref[0]).norm() / ref[0].norm()).item()
                    sv = ((b.S.cpu() - ref[1]).abs() / ref[1]).max().item()
                    row = {
                        "weight": label,
                        "rank": r,
                        "device": device,
                        "backend": backend,
                        "strategy": slabel,
                        "seconds": secs,
                        "peak_mib": peak,
                        "recon_rel_error": err,
                        "sv_rel_error": sv,
                    }
                    rows.append(row)
                    print(
                        f"{label:20s} r={r:3d} {device:4s} {slabel:32s} "
                        f"{secs:7.3f}s peak={'-' if peak is None else round(peak):>6} MiB  "
                        f"recon_err={err:.2e}  sv_err={sv:.2e}",
                        flush=True,
                    )
                W = None
                if device == "cuda":
                    torch.cuda.empty_cache()
    print("saved", save(a.out, {"meta": metadata(), "rows": rows}))


if __name__ == "__main__":
    main()
