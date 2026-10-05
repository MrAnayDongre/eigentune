# Compatibility and what has actually been tested

"Declared" is what `pyproject.toml` allows; "tested" is what a test or benchmark exercised.

## Software

| | declared | tested |
|---|---|---|
| Python | >= 3.9 | 3.10 locally; CI runs the CPU suite on 3.9 - 3.12 |
| PyTorch | >= 2.1 | 2.13.0 (CUDA 13.0 wheel) locally; CI runs the latest release. Older versions have not been tried. |
| Triton | optional, >= 3.0 | 3.7.1 |
| Transformers | optional, >= 4.40 | 5.17.0 (tiny random Llama; no downloads) |
| safetensors | >= 0.4 | 0.9.0rc |

## Hardware

| | status |
|---|---|
| CPU | tested (full unit, correctness, serialization, integration suites) |
| NVIDIA RTX 5070 Laptop (sm_120, 8 GB) | tested: Triton and native CUDA kernels, layer-level and training |
| other NVIDIA GPUs | untested; kernels use no architecture-specific instructions, but thresholds in `policy.py` are measured on one machine |
| AMD / ROCm | **not tested**, see [rocm](rocm.md) |
| multi-GPU | not tested |

## Dtypes and shapes

fp32, fp16 and bf16 for every backend (kernel tests, rectangular and odd sizes, tails, non-contiguous inputs). Ranks 1 - 256 in
the reference, <= 256 diagonal and <= 64 core in Triton, <= 64 in native.

## Training features

| | status |
|---|---|
| gradient accumulation | tested (equals one big batch) |
| autocast (bf16) | tested on CPU |
| gradient checkpointing | tested through Transformers (identical gradients) |
| `torch.compile` | tested, `fullgraph=True`, `aot_eager` backend in CI (inductor also worked locally) on CPU |
| tied weights | the default `exclude_modules=["lm_head"]` leaves a tied head alone |
| DDP / FSDP / DeepSpeed | **not tested**. The trainable tensors are ordinary `nn.Parameter`s and the bases are buffers, so DDP should work; FSDP needs the bases handled explicitly. Treat both as unsupported until tested. |

## Quantized base models

The layer wraps any module that has `in_features`, `out_features` and `weight` and is called as `base(x)`. For a quantized
base, pass the full-precision weights for the SVD with `get_eigentune_model(model, cfg, weights={"<name>.weight": w})`;
without them the call raises instead of decomposing integer data.

**This has not been tested here.** `bitsandbytes`, GPTQ and AWQ were not installed, so 4-bit and 8-bit bases are an unverified design,
not a supported feature. Merging into a quantized base is not supported (`merge` needs a plain floating-point `nn.Linear`).

## Hugging Face PEFT

EigenTune does not integrate with PEFT's `get_peft_model` (see [architecture](architecture.md) for why). It works on the same
models PEFT does, and PEFT is the baseline in the benchmarks.

## Migrating from 0.1

| 0.1 | 0.2 |
|---|---|
| `get_eigentune_model(model, cfg, full_precision_state_dict=sd)` | `get_eigentune_model(model, cfg, weights=sd)`; the old keyword still works and warns |
| returned a `PeftModel` | returns the same `nn.Module`, adapted in place. `print_trainable_parameters` is `eigentune.print_trainable_parameters(model)` |
| `EigenTunedLayer(layer, rank, weight)` | still works (deprecated); use `EigenTuneLinear` |
| `U` and `Vh` in `state_dict()` (full-rank, fp32) | non-persistent buffers holding only the rank-`r` directions |
| no adapter file | `save_adapter` / `load_adapter` |
| `peft` was a hard dependency | no longer a dependency |
