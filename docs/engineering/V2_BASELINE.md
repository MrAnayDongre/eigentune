# EigenTune 0.1.0 baseline

Measured on the `0.1.0` tag (`8c14489`) before the v2 work. CPU, fp32 base weights, rank 16.

| Field | Value |
|---|---|
| CURRENT_VERSION | 0.1.0 |
| CURRENT_TEST_COUNT | 0 (no tests, no CI) |
| CURRENT_API | `EigenTuneConfig(rank, target_modules)`, `EigenTunedLayer`, `get_eigentune_model(model, config, full_precision_state_dict)` |
| CURRENT_FORWARD_PATH | `base(x) + ((x @ Vh_r.T) * delta) @ U_r.T`, with `U_r`/`Vh_r` sliced from the full buffers and re-cast on every call |
| CURRENT_BACKWARD_PATH | plain autograd, no custom function, no kernels |
| CURRENT_GPU_SUPPORT | whatever PyTorch does; no kernels, never tested on a GPU |
| CURRENT_AMD_SUPPORT | none |
| CURRENT_PACKAGE_SIZE | 515 lines of Python, one wheel, depends on `torch` and `peft` |

## Architecture

`get_eigentune_model` builds a throwaway LoRA model with PEFT as scaffolding, walks it, and swaps every `LoraLayer` for an `EigenTunedLayer`.
The base weight for the SVD must be passed in separately as a full-precision state dict, looked up with `name.replace('base_model.model.', '') + '.weight'`.
Layers whose key is missing are skipped with a `print`.

## The storage story the README does not tell

`EigenTunedLayer` runs `torch.linalg.svd(W, full_matrices=False)` and registers the **complete** `U` and `Vh` as fp32 buffers, although only the top `r` columns are ever used.

| Layer | Init (CPU) | Runtime buffers | `state_dict()` | Trainable |
|---|---|---|---|---|
| 1024 x 1024 | 2.2 s | 8 MiB | 12 MiB | 16 params |
| 2048 x 2048 | 1.9 s | 32 MiB | 48 MiB | 16 params |
| 4096 x 4096 | 12.5 s | 128 MiB | 192 MiB | 16 params |
| 11008 x 4096 | 10.5 s | 236 MiB | 408 MiB | 16 params |

The bf16 weight of the 4096 x 4096 layer is 32 MiB, so the runtime buffers are 4x the weight they adapt and the saved adapter is 6x.
"16 trainable parameters" is true; "tiny adapter" is not, as shipped.

## Limitations found

- `U`/`Vh` are in `state_dict()`, so a Trainer checkpoint carries them.
- Full SVD is recomputed on every run; no truncated SVD, no cache.
- No adapter save/load. `examples/inference.py` imports `replace_peft_with_eigentune`, which does not exist.
- No merge, no unmerge, no tests of the math.
- The PEFT hijack depends on PEFT's private layer classes and on a key-naming convention.
- Singular-vector signs and repeated singular values are not addressed, so bases are not reproducible across runs by contract.
- The README calls the method "novel". It is not: see `docs/research/RELATED_WORK.md`.
- `pyproject.toml` allows Python 3.8 while `torch>=2.0` does not guarantee it, and the author email there looks mistyped.
