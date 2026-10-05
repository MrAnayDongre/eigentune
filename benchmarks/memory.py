"""Activation memory: bytes autograd keeps alive for the backward pass, per adapter.

Counted with ``saved_tensors_hooks``, deduplicated by storage and excluding parameters/buffers, so it is
device-independent. The input requires grad (as it does inside a model), the base layer is frozen.

    python benchmarks/memory.py [--out NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from _common import metadata, save  # noqa: E402

from eigentune import EigenTuneConfig, get_eigentune_model  # noqa: E402


KINDS = ("frozen", "eigentune_diag", "eigentune_core", "lora", "lora_bf16", "dora")


def saved_bytes(module: nn.Module, x: torch.Tensor) -> int:
    persistent = {p.untyped_storage().data_ptr() for p in list(module.parameters()) + list(module.buffers())}
    seen: dict[int, int] = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in persistent:
            seen[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        module(x)
    return sum(seen.values())


class Holder(nn.Module):
    def __init__(self, d_in, d_out, dtype):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False).to(dtype).requires_grad_(False)

    def forward(self, x):
        return self.proj(x)


def build(kind, rank, d_in, d_out, dtype):
    m = Holder(d_in, d_out, dtype)
    if kind == "frozen":
        return m
    if kind.startswith("eigentune"):
        cfg = EigenTuneConfig(rank=rank, method="spectral_core" if "core" in kind else "diagonal", backend="torch")
        return get_eigentune_model(m, cfg)
    from peft import LoraConfig, get_peft_model

    kw = {"use_dora": True} if kind == "dora" else {}
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=["proj"], lora_dropout=0.0, **kw)
    if kind == "lora_bf16":  # adapters kept in the base dtype: PEFT then does not cast the input to fp32
        return get_peft_model(m, cfg, autocast_adapter_dtype=False)
    return get_peft_model(m, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="memory")
    a = ap.parse_args()
    d_in, d_out, dtype = 4096, 4096, torch.bfloat16
    rows = []
    print(f"layer {d_in}x{d_out}, bf16; MiB of activations kept for backward (adapter + base)")
    print(f"{'tokens':>7} {'rank':>5}  " + "  ".join(f"{k:>16}" for k in KINDS))
    for n in (256, 2048, 8192):
        for r in (8, 16):
            x = torch.randn(n, d_in, dtype=dtype, requires_grad=True)
            row = {"tokens": n, "rank": r, "input_mib": n * d_in * 2 / 2**20}
            for kind in KINDS:
                row[kind] = saved_bytes(build(kind, r, d_in, d_out, dtype), x) / 2**20
            rows.append(row)
            print(f"{n:7d} {r:5d}  " + "  ".join(f"{row[k]:16.3f}" for k in KINDS))
    print("saved", save(a.out, {"meta": metadata(), "rows": rows}))


if __name__ == "__main__":
    main()
