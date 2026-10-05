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


def stack_peak(kind, rank, tokens, layers=4):
    """Peak GPU memory (above the resident weights) of one forward+backward through a Qwen3-shaped stack."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=layers,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=16384,
    )
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16)
    model.config.use_cache = False
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model.requires_grad_(False)
    if kind == "frozen":
        model.enable_input_require_grads()  # a frozen stack still has to backpropagate through its layers
    elif kind.startswith("eigentune"):
        model = get_eigentune_model(
            model,
            EigenTuneConfig(
                rank=rank,
                method="spectral_core" if "core" in kind else "diagonal",
                target_modules=targets,
                backend="torch",
            ),
        )
    else:
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=rank, lora_alpha=2 * rank, target_modules=targets, lora_dropout=0.0, use_dora=(kind == "dora")
        )
        model = get_peft_model(model, lc, autocast_adapter_dtype=(kind != "lora_bf16"))
    ids = torch.randint(0, 2048, (1, tokens), device="cuda")
    for _ in range(2):
        model(input_ids=ids, labels=ids).loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model(input_ids=ids, labels=ids).loss.backward()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    del model
    torch.cuda.empty_cache()
    return peak


def main_model(out):
    from _thermal import governor

    torch.cuda.set_per_process_memory_fraction(0.8)
    rows = []
    kinds = ("frozen", "eigentune_diag", "eigentune_core", "lora", "lora_bf16", "dora")
    print("peak MiB above resident weights, 4-layer Qwen3-shaped stack, bf16, rank 16 (frozen: input-grad only)")
    print(f"{'tokens':>7}  " + "  ".join(f"{k:>15}" for k in kinds))
    for tokens in (2048, 8192, 12288):
        row = {"tokens": tokens, "rank": 16}
        for k in kinds:
            governor()
            try:
                row[k] = stack_peak(k, 16, tokens)
            except torch.OutOfMemoryError:  # over the 6 GiB cap: that is a result, not a failure
                row[k] = None
                torch.cuda.empty_cache()
        rows.append(row)
        print(
            f"{tokens:7d}  " + "  ".join(f"{'OOM' if row[k] is None else f'{row[k]:.1f}':>15}" for k in kinds),
            flush=True,
        )
    print("saved", save(out, {"meta": metadata(), "rows": rows}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="memory")
    ap.add_argument("--model", action="store_true", help="whole-stack peak GPU memory instead of per-layer saved bytes")
    a = ap.parse_args()
    if a.model:
        return main_model("memory_stack")
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
