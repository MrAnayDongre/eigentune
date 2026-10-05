"""Whole-model training-step latency per backend: do the kernel wins survive inside a real transformer stack?

    python benchmarks/training.py [--out NAME]

A Qwen3-0.6B-shaped stack (1024 hidden, 3072 MLP, 16 heads) truncated to a few layers, randomly initialised (no
weights needed), all seven projections adapted, bf16. Time is one forward + backward of the language-model loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import metadata, save, time_gpu  # noqa: E402
from _thermal import governor  # noqa: E402

from eigentune import EigenTuneConfig, get_eigentune_model  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build(backend, method, rank, layers):
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
        max_position_embeddings=8192,
    )
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16)
    model.config.use_cache = False
    ecfg = EigenTuneConfig(rank=rank, method=method, target_modules=TARGETS, backend=backend)
    model = get_eigentune_model(model, ecfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.01)  # nonzero, so the backward does real work
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    rows = []
    for method in ("diagonal", "spectral_core"):
        for tokens in (1, 16, 768, 3072, 8192):
            row = {"method": method, "tokens": tokens, "rank": a.rank, "layers": a.layers}
            for backend in ("torch", "triton", "native", "auto"):
                governor()
                model = build(backend, method, a.rank, a.layers)
                ids = torch.randint(0, 2048, (1, tokens), device="cuda")

                def step(model=model, ids=ids):
                    model(input_ids=ids, labels=ids).loss.backward()

                torch.cuda.reset_peak_memory_stats()
                row[backend] = time_gpu(step, warmup=5, reps=5, inner=10)
                row[backend]["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
                del model
                torch.cuda.empty_cache()
            base = row["torch"]["median_us"]
            print(
                f"{method:14s} N={tokens:5d}: "
                + "  ".join(
                    f"{b} {row[b]['median_us'] / 1000:7.2f}ms (x{base / row[b]['median_us']:.2f})"
                    for b in ("torch", "triton", "native", "auto")
                ),
                flush=True,
            )
            rows.append(row)
    print("saved", save(a.out, {"meta": metadata(), "rows": rows}))


if __name__ == "__main__":
    main()
