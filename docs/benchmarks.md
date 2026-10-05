# Benchmarks


## Environment

GPU: NVIDIA GeForce RTX 5070 Laptop GPU; PyTorch 2.13.0+cu130; Triton 3.7.1; CUDA 13.0; Python 3.10.20; EigenTune 0.2.0.dev0 at `db0c51a`.

One laptop GPU that also drives the display, run one process at a time under a thermal governor. Every table below is regenerated from `benchmarks/results/*.json` by `benchmarks/report.py`.

## Kernels

Speedup over the `torch` backend (> 1 is faster), as a range over layer shapes (2048x2048 ... 8192x8192, 4096x11008) and ranks (8, 16, 64); *wins* counts shapes where the backend was faster. *Eager* includes Python/launch overhead (what an eager user sees); *graph* is the same calls replayed from a CUDA graph (GPU time only). bf16, diagonal. `benchmarks/kernels.py`.

**Native CUDA vs torch**

| tokens | forward eager | forward graph | backward eager | backward graph |
|---|---|---|---|---|
| 1 | 1.44 - 3.85x (wins 15/15) | 0.50 - 2.53x (wins 11/15) | 1.65 - 2.79x (wins 15/15) | 1.37 - 2.04x (wins 15/15) |
| 4 | 0.74 - 5.16x (wins 13/15) | 0.32 - 7.92x (wins 11/15) | 1.17 - 2.00x (wins 15/15) | 1.15 - 1.73x (wins 15/15) |
| 16 | 0.25 - 3.64x (wins 9/15) | 0.12 - 3.71x (wins 7/15) | 1.18 - 2.11x (wins 15/15) | 1.16 - 2.10x (wins 15/15) |
| 64 | 0.26 - 1.39x (wins 2/10) | 0.13 - 0.62x (wins 0/10) | 1.43 - 2.31x (wins 10/10) | 1.82 - 2.67x (wins 10/10) |
| 128 | 0.14 - 0.79x (wins 0/10) | 0.08 - 0.37x (wins 0/10) | 0.89 - 2.58x (wins 9/10) | 1.72 - 2.39x (wins 10/10) |

**Triton vs torch**

| tokens | forward eager | forward graph | backward eager | backward graph |
|---|---|---|---|---|
| 1 | 0.40 - 0.79x (wins 0/15) | 0.31 - 0.92x (wins 0/15) | 0.59 - 0.96x (wins 0/15) | 0.28 - 1.37x (wins 3/15) |
| 4 | 0.50 - 1.05x (wins 2/15) | 0.30 - 2.50x (wins 11/15) | 0.56 - 0.87x (wins 0/15) | 0.30 - 4.91x (wins 5/15) |
| 16 | 0.53 - 0.79x (wins 0/15) | 0.33 - 2.46x (wins 7/15) | 0.63 - 0.77x (wins 0/15) | 0.33 - 5.02x (wins 6/15) |
| 64 | 0.29 - 0.67x (wins 0/15) | 0.18 - 1.01x (wins 1/15) | 0.52 - 0.84x (wins 0/15) | 0.24 - 1.50x (wins 4/15) |
| 128 | 0.28 - 0.68x (wins 0/15) | 0.20 - 1.10x (wins 4/15) | 0.29 - 0.72x (wins 0/15) | 0.28 - 1.81x (wins 4/15) |
| 512 | 0.54 - 1.35x (wins 5/15) | 0.56 - 1.86x (wins 12/15) | 0.63 - 1.38x (wins 3/15) | 0.61 - 1.89x (wins 10/15) |
| 1024 | 0.81 - 1.46x (wins 13/15) | 1.06 - 2.25x (wins 15/15) | 0.75 - 1.87x (wins 13/15) | 1.07 - 2.29x (wins 15/15) |
| 2048 | 1.20 - 1.80x (wins 15/15) | 1.22 - 2.15x (wins 15/15) | 0.89 - 1.87x (wins 12/15) | 0.89 - 2.58x (wins 12/15) |

**Native CUDA vs torch, fp16**

| tokens | forward eager | forward graph | backward eager | backward graph |
|---|---|---|---|---|
| 1 | 0.99 - 3.16x (wins 14/15) | 0.55 - 2.31x (wins 11/15) | 1.69 - 2.76x (wins 15/15) | 1.39 - 2.06x (wins 15/15) |
| 4 | 0.73 - 4.90x (wins 12/15) | 0.28 - 7.09x (wins 10/15) | 1.16 - 2.08x (wins 15/15) | 1.15 - 1.83x (wins 15/15) |
| 16 | 0.26 - 3.26x (wins 9/15) | 0.11 - 3.31x (wins 7/15) | 1.18 - 1.97x (wins 15/15) | 1.16 - 2.62x (wins 15/15) |

**Triton vs torch, fp16**

| tokens | forward eager | forward graph | backward eager | backward graph |
|---|---|---|---|---|
| 512 | 0.52 - 1.41x (wins 7/15) | 0.54 - 1.78x (wins 12/15) | 0.59 - 1.21x (wins 1/15) | 0.56 - 1.59x (wins 9/15) |
| 2048 | 1.12 - 1.72x (wins 15/15) | 1.15 - 1.97x (wins 15/15) | 0.84 - 1.72x (wins 10/15) | 0.87 - 2.19x (wins 12/15) |

**Native CUDA vs torch, core method, bf16**

| tokens | forward eager | forward graph | backward eager | backward graph |
|---|---|---|---|---|
| 1 | 0.83 - 3.34x (wins 14/15) | 0.42 - 2.63x (wins 10/15) | 0.96 - 1.01x (wins 2/15) | 1.00 - 1.06x (wins 12/15) |
| 16 | 0.19 - 3.66x (wins 9/15) | 0.09 - 3.72x (wins 7/15) | 0.96 - 1.02x (wins 5/15) | 0.98 - 1.18x (wins 7/15) |

**Triton vs torch, core method, bf16**

| tokens | forward eager | forward graph | backward eager | backward graph |
|---|---|---|---|---|
| 1 | 0.59 - 0.65x (wins 0/15) | 0.34 - 1.03x (wins 2/15) | 0.61 - 0.72x (wins 0/15) | 0.25 - 1.27x (wins 3/15) |
| 16 | 0.60 - 1.14x (wins 1/15) | 0.33 - 2.49x (wins 8/15) | 0.68 - 0.83x (wins 0/15) | 0.40 - 5.64x (wins 6/15) |
| 512 | 0.52 - 1.54x (wins 7/15) | 0.53 - 1.94x (wins 11/15) | 0.52 - 1.29x (wins 6/15) | 0.50 - 3.02x (wins 9/15) |
| 2048 | 1.18 - 1.71x (wins 15/15) | 1.22 - 2.25x (wins 15/15) | 1.07 - 2.34x (wins 15/15) | 1.07 - 3.24x (wins 15/15) |

Where a column shows a range that dips below 1, that backend is slower than the PyTorch path for some shapes in that regime; the dispatch policy in `eigentune/kernels/policy.py` only selects a backend where it won across the measured shapes.

## Basis initialization

Real weights from Llama-2-7B layer 16, fetched tensor by tensor. Error is `||W_r(approx) - W_r(exact)||_F / ||W_r(exact)||_F` against the CPU exact SVD. `benchmarks/initialization.py`.

| weight | rank | device | strategy | seconds | peak GPU MiB | reconstruction error | singular-value error |
|---|---|---|---|---|---|---|---|
| q_proj 4096x4096 | 16 | cpu | exact | 5.163 | - | 0.0e+00 | 0.0e+00 |
| q_proj 4096x4096 | 16 | cpu | randomized (converged, default) | 0.428 | - | 8.0e-06 | 9.5e-07 |
| q_proj 4096x4096 | 16 | cpu | randomized (niter=4, p=8) | 0.673 | - | 4.7e-01 | 4.2e-02 |
| q_proj 4096x4096 | 16 | cpu | torch.svd_lowrank (niter=2) | 0.187 | - | 6.7e-01 | 8.7e-02 |
| q_proj 4096x4096 | 16 | cuda | exact | 5.231 | 387 | 9.9e-04 | 4.4e-04 |
| q_proj 4096x4096 | 16 | cuda | randomized (converged, default) | 0.065 | 75 | 2.4e-05 | 2.1e-06 |
| q_proj 4096x4096 | 16 | cuda | randomized (niter=4, p=8) | 0.037 | 71 | 4.5e-01 | 4.2e-02 |
| q_proj 4096x4096 | 16 | cuda | torch.svd_lowrank (niter=2) | 0.035 | 71 | 6.6e-01 | 9.4e-02 |
| q_proj 4096x4096 | 64 | cpu | exact | 4.674 | - | 0.0e+00 | 0.0e+00 |
| q_proj 4096x4096 | 64 | cpu | randomized (converged, default) | 4.174 | - | 3.6e-05 | 9.8e-07 |
| q_proj 4096x4096 | 64 | cpu | randomized (niter=4, p=8) | 0.578 | - | 3.9e-01 | 5.5e-02 |
| q_proj 4096x4096 | 64 | cpu | torch.svd_lowrank (niter=2) | 0.438 | - | 5.6e-01 | 1.1e-01 |
| q_proj 4096x4096 | 64 | cuda | exact | 5.249 | 388 | 8.0e-04 | 4.4e-04 |
| q_proj 4096x4096 | 64 | cuda | randomized (converged, default) | 0.123 | 83 | 5.3e-05 | 4.4e-06 |
| q_proj 4096x4096 | 64 | cuda | randomized (niter=4, p=8) | 0.041 | 77 | 3.9e-01 | 5.1e-02 |
| q_proj 4096x4096 | 64 | cuda | torch.svd_lowrank (niter=2) | 0.038 | 77 | 5.6e-01 | 1.1e-01 |
| up_proj 11008x4096 | 16 | cpu | exact | 7.507 | - | 0.0e+00 | 0.0e+00 |
| up_proj 11008x4096 | 16 | cpu | randomized (converged, default) | 1.094 | - | 3.9e-05 | 4.0e-07 |
| up_proj 11008x4096 | 16 | cpu | randomized (niter=4, p=8) | 0.159 | - | 5.7e-01 | 5.0e-02 |
| up_proj 11008x4096 | 16 | cpu | torch.svd_lowrank (niter=2) | 0.121 | - | 8.3e-01 | 1.5e-01 |
| up_proj 11008x4096 | 16 | cuda | exact | 5.229 | 837 | 8.6e-04 | 4.1e-04 |
| up_proj 11008x4096 | 16 | cuda | randomized (converged, default) | 0.162 | 186 | 4.4e-05 | 3.1e-06 |
| up_proj 11008x4096 | 16 | cuda | randomized (niter=4, p=8) | 0.097 | 180 | 6.1e-01 | 5.5e-02 |
| up_proj 11008x4096 | 16 | cuda | torch.svd_lowrank (niter=2) | 0.092 | 180 | 8.6e-01 | 1.6e-01 |
| up_proj 11008x4096 | 64 | cpu | exact | 7.663 | - | 0.0e+00 | 0.0e+00 |
| up_proj 11008x4096 | 64 | cpu | randomized (converged, default) | 5.064 | - | 1.6e-04 | 1.6e-06 |
| up_proj 11008x4096 | 64 | cpu | randomized (niter=4, p=8) | 0.259 | - | 6.3e-01 | 7.2e-02 |
| up_proj 11008x4096 | 64 | cpu | torch.svd_lowrank (niter=2) | 0.201 | - | 8.3e-01 | 1.5e-01 |
| up_proj 11008x4096 | 64 | cuda | exact | 5.227 | 840 | 1.6e-03 | 4.1e-04 |
| up_proj 11008x4096 | 64 | cuda | randomized (converged, default) | 0.445 | 199 | 5.5e-05 | 3.7e-06 |
| up_proj 11008x4096 | 64 | cuda | randomized (niter=4, p=8) | 0.101 | 190 | 6.2e-01 | 7.5e-02 |
| up_proj 11008x4096 | 64 | cuda | torch.svd_lowrank (niter=2) | 0.097 | 190 | 8.4e-01 | 1.5e-01 |

## Activation memory

MiB of activations autograd keeps for the backward pass, per adapted 4096x4096 layer in bf16 (adapter and base; device-independent, counted with `saved_tensors_hooks`). `benchmarks/memory.py`.

| tokens | rank | input X | frozen | eigentune_diag | eigentune_core | lora | lora_bf16 | dora |
|---|---|---|---|---|---|---|---|---|
| 256 | 8 | 2 | 0.000 | 0.004 | 0.004 | 4.008 | 2.004 | 74.180 |
| 256 | 16 | 2 | 0.000 | 0.008 | 0.008 | 4.016 | 2.008 | 74.312 |
| 2048 | 8 | 16 | 0.000 | 0.031 | 0.031 | 32.062 | 16.031 | 144.234 |
| 2048 | 16 | 16 | 0.000 | 0.063 | 0.063 | 32.125 | 16.062 | 144.422 |
| 8192 | 8 | 64 | 0.000 | 0.125 | 0.125 | 128.250 | 64.125 | 384.422 |
| 8192 | 16 | 64 | 0.000 | 0.250 | 0.250 | 128.500 | 64.250 | 384.797 |

`lora` is PEFT's default (adapters in fp32, so the input is cast and a copy kept); `lora_bf16` keeps adapters in the base dtype. Plain autograd with frozen bases also keeps only `Q` (measured: 0.250 MiB at 8192 tokens, rank 16), so the custom autograd function is not what saves this memory.
