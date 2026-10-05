"""Kernel latency: forward update and forward+backward, per backend, eager and CUDA-graph.

    python benchmarks/kernels.py [--out NAME] [--quick]

Bounded on purpose: representative transformer shapes only, one process, no profiler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import metadata, save, time_gpu, time_graph  # noqa: E402

from eigentune.kernels import available_backends, get_backend  # noqa: E402

DIMS = [(2048, 2048), (4096, 4096), (4096, 11008), (11008, 4096), (8192, 8192)]
TOKENS = [1, 4, 16, 64, 128, 512, 2048]
RANKS = [8, 16, 64]


def run(dtype, quick):
    dev = "cuda"
    rows = []
    names = available_backends()
    dims = [(4096, 4096), (4096, 11008)] if quick else DIMS
    tokens = [1, 16, 128, 2048] if quick else TOKENS
    ranks = [16] if quick else RANKS
    for (inn, out) in dims:
        for r in ranks:
            Vh = (torch.randn(r, inn, device=dev) / inn**0.5).to(dtype)
            U = (torch.randn(out, r, device=dev) / r**0.5).to(dtype)
            w = torch.randn(r, device=dev)
            for N in tokens:
                x = torch.randn(N, inn, device=dev).to(dtype)
                base = torch.randn(N, out, device=dev).to(dtype)
                g = torch.randn(N, out, device=dev).to(dtype)
                for name in names:
                    b = get_backend(name)
                    if not b.supports(x, r, "diag"):
                        continue
                    wt = w.to(dtype)
                    _, Q = b.forward(x, Vh, U, wt, "diag", base)
                    fwd = lambda: b.forward(x, Vh, U, wt, "diag", base)  # noqa: E731
                    bwd = lambda: b.backward(g, Q, Vh, U, wt, "diag", True)  # noqa: E731
                    row = {"in": inn, "out": out, "rank": r, "tokens": N, "backend": name, "dtype": str(dtype)}
                    for tag, fn in (("fwd", fwd), ("bwd", bwd)):
                        row[f"{tag}_eager"] = time_gpu(fn)
                        row[f"{tag}_graph"] = time_graph(fn)
                    rows.append(row)
                    print(f"{inn}x{out} r={r} N={N:5d} {name:7s} fwd eager {row['fwd_eager']['median_us']:7.1f}us "
                          f"graph {row['fwd_graph']['median_us']:7.1f}us | bwd eager {row['bwd_eager']['median_us']:7.1f}us "
                          f"graph {row['bwd_graph']['median_us']:7.1f}us", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kernels")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "this benchmark needs a GPU"
    rows = run(getattr(torch, a.dtype), a.quick)
    print("saved", save(a.out, {"meta": metadata(), "rows": rows}))


if __name__ == "__main__":
    main()
