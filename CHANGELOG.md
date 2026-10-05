# Changelog

All notable changes. Versions before 1.0 may change interfaces between minor releases.

## 0.2.0 (unreleased)

### New
- `method="spectral_core"`: an `r x r` core between the frozen singular bases, optionally banded (`core_bandwidth`).
  Classic diagonal EigenTune is unchanged and remains the default.
- `selection` (`principal`, `minor`, `mixed`), `update` (`additive`, `relative`) and an experimental global
  `rank_budget` that spends singular directions where the spectral energy is.
- `save_adapter` / `load_adapter`: safetensors plus a versioned JSON header; base-weight fingerprints and a basis
  signature are verified on load; `save_bases=True` embeds the exact bases.
- `merge_adapter` / `unmerge_adapter`, `adapter_report` (trainable parameters, adapter bytes and runtime basis bytes, kept apart).
- Direct module injection: no PEFT dependency, works on a plain `nn.Module`.
- Backends: `torch` (reference), `triton` (CUDA and ROCm), `native` (CUDA; HIP source included, untested on AMD),
  chosen by `backend="auto"` from measured thresholds, or by name. Third-party backends can register.
- Converged randomized truncated SVD with a seeded generator and an on-disk cache.

### Performance
- Adapter activation memory: 0.25 MiB per 4096x4096 layer at 8192 tokens, rank 16, against 64-385 MiB for LoRA/DoRA (measured).
- Basis initialization on GPU: 0.07-0.45 s against 5.2 s for a full SVD, reconstruction error <= 1.6e-4 (Llama-2-7B weights).
- Few-token forward on CUDA: up to 3.9x over the PyTorch path (eager); large-token forward and backward: up to ~2x with Triton.
  Details and the regimes where each backend loses: `docs/benchmarks.md`.

### Compatibility
- Python >= 3.9, PyTorch >= 2.1 declared (see `docs/compatibility.md` for what was tested).
- `peft` is no longer a dependency.

### Breaking changes
- `get_eigentune_model` now adapts the model in place and returns it (0.1 returned a `PeftModel`).
  `full_precision_state_dict=` is renamed `weights=`; the old name still works and warns.
- `U`/`Vh` are no longer in `state_dict()`; only rank-`r` directions are kept (0.1 stored the full-rank matrices in fp32).
- `EigenTunedLayer` is deprecated in favour of `EigenTuneLinear` (the old constructor still works).
- The default for `rank` is now 8 (was 4).

### Known limitations
- ROCm/HIP: implemented and statically checked only; never compiled or run on AMD hardware.
- Quantized (4-bit/8-bit) bases, DDP, FSDP and multi-GPU are untested.
- Dispatch thresholds are measured on one GPU (RTX 5070 Laptop).
- Quality comparisons use one small model and task (Qwen3-0.6B on GSM8K solutions); see `docs/benchmarks.md` for their scope.
