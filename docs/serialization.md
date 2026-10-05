# Serialization and storage

## Four different byte counts

"How big is the adapter" has several honest answers. EigenTune reports them separately (`adapter_report`).

| name | what it is |
|---|---|
| `trainable_parameters` | numbers the optimiser updates: `r` (diagonal) or `r²` (core) per adapted layer |
| `adapter_bytes` | the file written by `save_adapter`: those numbers plus a small JSON header |
| `runtime_basis_bytes` | the frozen `U_r`, `V_r`, `σ_r` that sit next to the model while it runs: `r (in + out)` numbers per layer |
| `adapter + bases` | `save_adapter(..., save_bases=True)`: the adapter file with the exact bases embedded |

Measured on Qwen3-0.6B, 196 adapted layers, rank 16 (`benchmarks/results/quality`): 3,136 trainable parameters,
about 60 KB adapter file, about 20 MB of runtime bases (20 MB also if embedded). So the adapter is tiny to store and
ship **only because the bases are rebuilt from the base model**, and the runtime footprint of the bases is not zero.

## What is saved

`save_adapter(model, dir)` writes:

* `adapter_model.safetensors`: `<layer>.delta` (or `<layer>.core`) for every adapted layer. Nothing else by default.
* `eigentune_config.json`: format version, EigenTune version, the full `EigenTuneConfig`, and per layer a 24-hex-char
  fingerprint of the base weight plus an 8-number signature of the basis.

The frozen bases are registered as non-persistent buffers, so they are never in `state_dict()`: a Trainer checkpoint of
an adapted model carries only the trainable values (and the frozen base weights, as usual).

## Loading

`load_adapter(model, dir)` recomputes the bases from the model it is given, then checks, before anything is copied:

1. the same layers exist,
2. every layer's base-weight fingerprint (a hash of the exact weight bytes plus rank, selection, SVD algorithm and
   parameters, dtype and format version) matches,
3. the recomputed basis matches the saved signature.

A mismatch raises `AdapterMismatchError` naming the layer. A corrupt or too-new header raises `ValueError`.

## Why there is a signature as well as a fingerprint

The fingerprint proves the *input* is the same. The signature checks the *output* is the same, because SVD is not
perfectly reproducible across devices and library versions: on real Llama-2-7B weights, CUDA `gesvd` and CPU LAPACK
differ in the rank-r reconstruction by about 1e-3 (`benchmarks/results/initialization.json`). For the diagonal method that is
harmless. For a dense core, or near-repeated singular values, a different basis means different semantics for the same
numbers.

If you need an adapter to be exactly portable, save it with `save_bases=True`: the exact bases are embedded, no SVD runs at
load, and the result does not depend on the machine that loads it. That is the one case where adapter size scales with the layer width.
