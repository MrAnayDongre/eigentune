# EigenTune

Parameter-efficient fine-tuning inside a model's pretrained singular subspaces.

[![CI](https://github.com/MrAnayDongre/eigentune/actions/workflows/ci.yml/badge.svg)](https://github.com/MrAnayDongre/eigentune/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

EigenTune freezes a layer's weight `W = U Σ Vᵀ` and trains a tiny update that only moves along `W`'s own top singular
directions: one scalar per direction (`diagonal`), or an `r × r` matrix mixing them (`spectral_core`). It adds a few thousand
trainable parameters per model instead of millions.

**It is not a new idea.** Training singular-value scales is SVFit, an `r × r` core between frozen SVD bases is LoRA-XS, and sparse patterns over
singular-vector products are SVFT ([related work](docs/research/RELATED_WORK.md)). This is a careful implementation of them with
honest accounting and optional fast kernels. **It is also not the highest-quality PEFT method**: on the one task measured it trails LoRA,
DoRA and PiSSA clearly (below). What you get is a very small trainable footprint, a tiny adapter file, and very low adapter activation memory.

## Install

```bash
pip install eigentune                  # CPU, any PyTorch device; needs only torch, numpy, safetensors
pip install "eigentune[triton]"        # + Triton kernels (NVIDIA CUDA; AMD untested)
```

The native CUDA kernels are built on first use if `nvcc` is available, and silently skipped if not.

## Quickstart

```python
from eigentune import EigenTuneConfig, get_eigentune_model, print_trainable_parameters, save_adapter, load_adapter

cfg = EigenTuneConfig(rank=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])  # method="diagonal"
model = get_eigentune_model(model, cfg)  # freezes the model, adapts it in place; works with Trainer or any loop
print_trainable_parameters(model)

# ... train as usual: only the per-direction scales have gradients ...

save_adapter(model, "my-adapter")  # a few tens of KB
load_adapter(fresh_base_model, "my-adapter")  # refuses to load against different base weights
```

A runnable CPU example is in [`examples/quickstart_cpu.py`](examples/quickstart_cpu.py); a Hugging Face `Trainer` fine-tune
with adapter-only checkpoints is in [`examples/finetune_gsm8k.py`](examples/finetune_gsm8k.py).

## The idea in 30 seconds

For a frozen linear layer, take the top `r` singular triplets `U_r, σ_r, V_r` of its weight and add

| method | trainable | update | parameters per layer |
|---|---|---|---|
| `diagonal` | `δ ∈ R^r` | `U_r diag(δ) V_rᵀ` | `r` |
| `spectral_core` | `C ∈ R^{r×r}` | `U_r C V_rᵀ` | `r²` |

Both start at zero, so the model is unchanged at step 0. The forward pass never builds the dense update:
`y = W x + U_r (scale(V_rᵀ x))`. Details, gradients and the memory argument are in [docs/math.md](docs/math.md).
Principal, minor or mixed directions (`selection`), additive or relative scaling (`update`), a banded core and an experimental
global rank budget are options.

## What it costs, honestly

"Few trainable parameters" is not "a small adapter that costs nothing to carry". Four different numbers
(Qwen3-0.6B, 196 adapted layers):

| | diagonal r=16 | diagonal r=64 | core r=8 | LoRA r=8 |
|---|---|---|---|---|
| trainable parameters | 3,136 | 12,544 | 12,544 | 5,046,272 |
| adapter file, bases rebuilt on load (default) | 60 KB | 97 KB | 97 KB | 20.2 MB |
| adapter file with exact bases embedded | 20.3 MB | 80.9 MB | 10.3 MB | 20.2 MB |
| frozen bases kept in memory at runtime | 20.2 MB | 80.8 MB | 10.1 MB | none |

The default adapter is tiny **because the bases are recomputed from your base model on load**, with a fingerprint and a signature
check to make sure they match. If you need an adapter that loads identically on any machine, embed the bases, and then it is as large as a
LoRA adapter. See [docs/serialization.md](docs/serialization.md).

## Measured results

One laptop GPU, one small model, one task: Qwen3-0.6B fine-tuned on GSM8K solutions (150 steps, two seeds, learning rate searched for
every method). Full tables, methodology and the other benchmarks are in [docs/benchmarks.md](docs/benchmarks.md).

| method | trainable params | eval loss (lower is better) | steps/s | peak VRAM |
|---|---|---|---|---|
| LoRA r=8 | 5,046,272 | **0.534** | 3.37 | 3,588 MiB |
| DoRA r=8 | 5,390,336 | 0.534 | 1.96 | 3,909 MiB |
| PiSSA r=8 | 5,046,272 | 0.545 | 3.40 | 3,588 MiB |
| LoRA r=1 | 630,784 | 0.547 | 3.51 | 3,537 MiB |
| EigenTune diagonal r=64 | 12,544 | 0.612 | 3.36 | 3,608 MiB |
| EigenTune core r=8 | 12,544 | 0.629 | 3.33 | 3,540 MiB |
| EigenTune diagonal r=16 | 3,136 | 0.664 | 3.38 | 3,550 MiB |

(Pretrained model: 1.388 on the first 128 of the 500 evaluation problems.)

**Where it is ahead:**
trainable parameters (50x to 1,600x fewer); adapter file size when the bases are rebuilt; and adapter activation memory: in a 4-layer
stack at 8,192 tokens it keeps the peak equal to a frozen model's (1,422 against 1,463 MiB) while LoRA with PEFT's default adds 1,118 MiB and DoRA 3,097 MiB.

**Where it is behind:**
- Quality: clearly worse than LoRA, DoRA, rsLoRA, LoRA+ and PiSSA here, and even LoRA with 50x more parameters than it (rank 1) is ahead.
- The `r × r` core does **not** beat a bigger diagonal at the same parameter count (0.629 against 0.612); it needs 8x fewer basis bytes, which is a storage trade-off.
- Minor directions and the relative update did not help.
- Initialization takes 4.5 - 14.5 s (SVDs of every adapted layer) against under 2 s for LoRA and PiSSA.
- Training speed and whole-model peak memory at small batch are the same as LoRA, not better.

EigenTune is not state of the art on any of the dimensions measured except trainable-parameter count, file size and adapter activation memory.

## Backends

`backend="auto"` (the default) picks per call from thresholds measured on one machine; anything not measured uses plain PyTorch.

| backend | what it is | status |
|---|---|---|
| `torch` | reference implementation, any device, fp32/fp16/bf16 | tested on CPU and CUDA |
| `triton` | fused kernels, wins at about 2,048+ tokens (up to ~2x) | tested on an RTX 5070 Laptop only |
| `native` | CUDA kernels, wins for few-token forward (up to 5.2x at 4 tokens, shape-dependent) and a fused backward reduction | tested on an RTX 5070 Laptop only |
| ROCm / HIP | HIP source via PyTorch's hipify, Triton on ROCm | **experimental: never compiled or run on AMD hardware** ([docs/rocm.md](docs/rocm.md)) |

Kernel speedups are real at the layer level, but inside a whole transformer training step they amount to 0 - 5% (the adapter is a small
part of a step). Tables, including the regimes where each backend loses, are in [docs/benchmarks.md](docs/benchmarks.md) and
[docs/kernels.md](docs/kernels.md). Third-party backends can be registered; see the same page.

## Saving, loading, merging

```python
from eigentune import merge_and_unload, merge_adapter, unmerge_adapter

merge_adapter(model)  # fold the update into the base weights (exact in fp32)
unmerge_adapter(model)
model = merge_and_unload(model)  # a plain model again: normal state_dict keys, save_pretrained works
```

An adapted model's own `state_dict()` has wrapper keys, so Trainer's built-in checkpoints are not loadable with a stock `from_pretrained`.
Use `EigenTuneCallback` (writes an adapter next to each checkpoint) or `merge_and_unload` first.

## Compatibility and what has not been tested

Python 3.9 - 3.12, PyTorch (verified on 2.13), CPU and NVIDIA CUDA. [docs/compatibility.md](docs/compatibility.md) lists what was run and what was
not. In short, **not tested**: AMD/ROCm hardware, 4-bit/8-bit quantized bases, DDP, FSDP and multi-GPU, any GPU other than the one benchmarked,
and any model larger than 0.6B parameters.

## Development

```bash
pip install -e ".[dev]"
pytest -q                       # CPU suite; GPU and ROCm tests skip themselves
ruff check . && ruff format --check .
python benchmarks/kernels.py    # and the other scripts in benchmarks/; results are stored in benchmarks/results/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](docs/architecture.md). Changes are in [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0.
