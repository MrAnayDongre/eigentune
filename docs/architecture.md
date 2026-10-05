# Architecture

```
eigentune/
  config.py            EigenTuneConfig: one dataclass, validated, round-trips through JSON
  svd.py               compute_bases: exact | randomized (converged) | lowrank; selection, canonical signs,
                       fingerprint, on-disk cache, global rank allocation
  layers.py            EigenTuneLinear: wraps a base layer, owns the bases (buffers) and the trainable tensor
  model.py             get_eigentune_model, merge_adapter / unmerge_adapter, adapter_report
  serialization.py     save_adapter / load_adapter (safetensors + a versioned JSON header)
  compat.py            the 0.1 EigenTunedLayer constructor, deprecated
  kernels/
    reference.py       TorchBackend: the correctness oracle, plain PyTorch ops
    triton_backend.py  fused skinny-GEMM kernels (CUDA and ROCm through Triton)
    native_backend.py  JIT-built CUDA/HIP extension: csrc/eigentune_ops.cu
    autograd.py        one autograd.Function that works with any backend
    dispatch.py        backend registry and selection
    policy.py          which backend `auto` prefers, from measured thresholds
  csrc/eigentune_ops.cu
benchmarks/            kernels, initialization, memory, quality, compare_peft (+ results/)
tests/                 unit, correctness, serialization, integration, kernels, rocm
```

## Data flow of one forward call

1. `EigenTuneLinear.forward(x)` runs the frozen base layer: `y = base(x)`.
2. It flattens `x` and `y` to 2-D, builds the effective scale (`δ` or `C`, after the band mask and relative scaling) in the
   compute dtype of `y`, and calls `eigentune_update(x, y, Vh, U, w, kind, backend)`.
3. `eigentune_update` is one `torch.autograd.Function`. Forward asks `dispatch.select_backend` for a backend, which returns
   `y + scale(x Vhᵀ) Uᵀ` and `Q = x Vhᵀ`; `Q` and the bases are all it saves.
4. Backward asks for a backend again for the backward phase (it can differ: the best forward kernel at 4 tokens is not the best
   backward kernel) and returns `∂L/∂x`, `∂L/∂y` (a pass-through) and `∂L/∂w`.

## Design rules

* **Reference first.** Every optimised path is tested for equivalence against `TorchBackend`, which is plain PyTorch.
* **The accelerators are optional.** Importing `eigentune` needs only `torch` and `safetensors`. `triton_backend` and
  `native_backend` register themselves if they import and never raise at import time; a named backend that cannot run a given
  call falls back to `torch` instead of failing.
* **No PEFT dependency.** PEFT's custom-tuner path means writing into its private mapping tables, and the brittle
  0.1 approach (build LoRA layers, then replace them) is gone. EigenTune injects into a plain `nn.Module`, which is also what
  `transformers.Trainer` and a bare training loop expect. `peft` is used only by the benchmarks, as the baseline.
* **Dispatch follows measurements.** `kernels/policy.py` holds the thresholds; each number points at a benchmark in
  [benchmarks](benchmarks.md). Regimes nobody measured (fp32, ROCm) stay on `torch`.
* **Reductions are deterministic.** No atomics; per-block partial sums are reduced in a fixed order.

## Extending

* **A new backend** implements `available`, `supports`, `supports_bwd`, `forward`, `backward` and calls
  `eigentune.kernels.register_backend(...)`. See [kernels](kernels.md).
* **A new selection / update rule** is a branch in `svd.select_indices` or `EigenTuneLinear.effective`; the kernels
  only ever see the final vector or matrix `w`.
* **A new SVD strategy** is a function in `svd.py` returning `(U, S, Vh)` and a line in `compute_bases`.
