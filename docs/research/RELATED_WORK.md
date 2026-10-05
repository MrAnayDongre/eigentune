# Related work

EigenTune adapts a frozen weight `W = U Σ Vᵀ` through its own singular directions.
Several published methods already work in this space, and some are the same idea. This page says which, so the
README does not have to guess.

Paper links are the arXiv IDs used by the Hugging Face PEFT documentation, or confirmed against the paper's own page.

## The methods EigenTune is closest to

| Method | What it trains | How it relates to EigenTune |
|---|---|---|
| **SVFit** ([2409.05926](https://arxiv.org/abs/2409.05926)) | the top-`r` singular values, scaling the corresponding subspaces | The same adaptation as classic (`diagonal`) EigenTune. EigenTune did not introduce it. |
| **LoRA-XS** ([2405.17604](https://arxiv.org/abs/2405.17604)) | an `r × r` matrix between frozen low-rank matrices taken from a truncated SVD of `W` | The same adaptation as `spectral_core`. |
| **SVFT** ([2405.19597](https://arxiv.org/abs/2405.19597)) | a sparse set of coefficients over outer products of singular vectors (diagonal, banded, random, top-`k` patterns) | A superset: `spectral_core` with a band is one of its patterns. EigenTune implements diagonal, banded and dense only. |
| **PiSSA** ([2404.02948](https://arxiv.org/abs/2404.02948)) | full LoRA factors `A`, `B` initialised from the principal singular components; the residual is written back into `W` | Same principal subspace, but PiSSA trains the factors and modifies the base weight. EigenTune freezes both bases and leaves `W` untouched. |
| **MiLoRA** ([2406.09044](https://arxiv.org/abs/2406.09044)), **MiCA** ([2604.01694](https://arxiv.org/abs/2604.01694)) | LoRA factors initialised from the *minor* singular components | `selection="minor"` selects the same subspace; the trained parameters differ. |
| **EVA** ([2410.07170](https://arxiv.org/abs/2410.07170)) | LoRA factors initialised from an SVD of *activations*, with ranks redistributed across layers | Activation-driven where EigenTune is weight-driven. Not implemented here; benchmarked through PEFT. |

So "SVD for PEFT" is not new, and neither is training only singular-value scales.

## Other PEFT baselines used in the benchmarks

All of these are available in Hugging Face PEFT and are run from there, not reimplemented.

| Method | PEFT option | Paper |
|---|---|---|
| LoRA | default | [2106.09685](https://arxiv.org/abs/2106.09685) |
| rsLoRA | `use_rslora=True` | [2312.03732](https://arxiv.org/abs/2312.03732) |
| DoRA | `use_dora=True` | [2402.09353](https://arxiv.org/abs/2402.09353) |
| PiSSA | `init_lora_weights="pissa"` / `"pissa_niter_N"` | [2404.02948](https://arxiv.org/abs/2404.02948) |
| LoRA+ | `create_loraplus_optimizer` | [2402.12354](https://arxiv.org/abs/2402.12354) |
| LoRA-FA | `create_lorafa_optimizer` | [2308.03303](https://arxiv.org/abs/2308.03303) |
| EVA | `init_lora_weights="eva"` | [2410.07170](https://arxiv.org/abs/2410.07170) |
| CorDA | `init_lora_weights="corda"` | [2406.05223](https://arxiv.org/abs/2406.05223) |
| LoRA-GA | `lora_ga_config` | [2407.05000](https://arxiv.org/abs/2407.05000) |
| LoftQ | `init_lora_weights="loftq"` | [2310.08659](https://arxiv.org/abs/2310.08659) |
| OLoRA | `init_lora_weights="olora"` | (see PEFT docs) |
| VeRA | `VeraConfig` | [2310.11454](https://arxiv.org/abs/2310.11454) |

## What EigenTune does differently

Not a new adaptation. The differences are in what is implemented and how:

1. **One frozen-basis family.** `{diagonal, banded core, dense core} x {principal, minor, mixed} x {additive, relative}`
   share one layer, one kernel interface and one serialization format, so they can be ablated against each other
   without changing code. Identity at initialisation holds for all of them.
2. **The base weight is never modified.** Unlike PiSSA-style residual surgery, the base layer (including a quantized
   one) is used as is, which is also what makes merge/unmerge a plain add of `U C Vᵀ`.
3. **Storage is stated honestly.** Trainable parameters, adapter-file bytes and runtime basis bytes are reported
   separately (`adapter_report`), because the frozen bases are `r (d_in + d_out)` numbers per layer even though only
   `r` or `r²` are trained. See [serialization](../serialization.md).
4. **Systems work.** A custom autograd function that saves `Q = X Vᵀ` (`[tokens, r]`) rather than the activation,
   fused Triton kernels, native CUDA kernels for the few-token regime, a randomized truncated SVD with a cache, and a
   backend dispatcher driven by benchmarks. Where a vendor GEMM wins, the dispatcher uses it.

Anything stronger than this (faster convergence, better quality, lower memory than a given method) is a measured
claim and lives in [benchmarks](../benchmarks.md) together with the cases where EigenTune loses.

## Systems references

PyTorch `torch.svd_lowrank` documents that it implements Algorithm 5.1 of Halko, Martinsson and Tropp (2009) and that the
full `torch.linalg.svd` is usually faster for dense matrices of moderate size; the `auto` SVD backend therefore only
switches to a truncated solver above a size threshold chosen from `benchmarks/initialization.py`.
