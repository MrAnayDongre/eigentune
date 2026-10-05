# Benchmarks

Everything here was measured on one laptop (RTX 5070 Laptop GPU, 8 GB), one small model (Qwen3-0.6B) and one task
(GSM8K solutions). It does not show how EigenTune behaves on larger models, other tasks or other hardware. The tables below the
summary are regenerated from the stored results and are the source of truth.

## Summary

**Where EigenTune is ahead, measured**

* **Trainable parameters.** 3,136 to 12,544, against 630,784 (LoRA rank 1) to 5,046,272 (LoRA rank 8): 50x to 1,600x fewer.
* **Adapter file size, if the bases are rebuilt on load.** 59,587 to 97,483 bytes against 2.6 MB (LoRA rank 1) to 20.2 MB
  (LoRA rank 8). This is the default `save_adapter`.
* **Activation memory.** In a 4-layer Qwen3-shaped stack at 8,192 tokens, EigenTune's peak is 1,422 MiB, the same as a frozen
  model that only backpropagates (1,463 MiB); LoRA with PEFT's default fp32 adapters adds 1,118 MiB, LoRA with bf16 adapters 279 MiB, DoRA
  3,097 MiB (and DoRA runs out of memory at 12,288 tokens under the 6 GiB cap).
* **Training speed.** The same steps per second as LoRA (3.3 - 3.4 against 3.4 - 3.5); DoRA is about 1.7x slower.

**Where EigenTune is behind, measured**

* **Quality.** Eval loss 0.612 (best EigenTune, diagonal rank 64, 12,544 parameters) against 0.534 for LoRA rank 8. LoRA rank 1, with
  630,784 parameters, reaches 0.547, ahead of every EigenTune configuration. The pretrained model scores 1.388 on the first 128 of the 500 evaluation problems.
* **The `r x r` core does not beat the diagonal per parameter.** At the same 12,544 trainable parameters, diagonal rank 64 gets 0.612 and the
  core at rank 8 gets 0.629. The core reaches that with 8x fewer basis bytes (10.3 MB against 80.9 MB with bases embedded), so it is a
  storage trade-off, not a quality win.
* **Minor directions are not better than principal ones** here (diagonal rank 16: 0.684 against 0.664, one seed), and the relative update
  makes no difference (0.666 against 0.664).
* **Storage with the bases.** The frozen bases are not free: 20 to 81 MB at runtime, and the same if embedded in the file for exact
  portability, which is as large as or larger than the LoRA adapter (20.2 MB).
* **Initialization.** 4.5 to 14.5 s for the SVDs of 196 layers, against 0.6 s for LoRA and 1.4 s for PiSSA.
* **Whole-model peak memory in the training runs** is the same as LoRA (3.54 - 3.61 GB against 3.59 GB), because at that batch size the adapter is not
  what dominates memory.
* **Kernels do not change a training step much.** The kernel-level speedups below are real (up to 5.2x for a 4-token forward) but inside a
  transformer step the adapter is a small part of the time: the accelerated backends are within 0 - 5% of the PyTorch path at 3,072 and
  8,192 tokens, and within noise at small token counts (`benchmarks/results/training.json`).

**State of the art?** In none of the dimensions measured except trainable-parameter count, adapter file size (with bases rebuilt)
and adapter activation memory. On downstream quality EigenTune is clearly behind LoRA, DoRA, rsLoRA, LoRA+ and PiSSA on this task.

## How the quality comparison was run

Qwen3-0.6B (bf16), seven attention and MLP projections per layer adapted, 2,000 GSM8K training solutions, 150 optimizer steps,
effective batch 8 (micro-batch 2 with 4 accumulation steps), sequence length 192, cosine schedule, AdamW, no weight decay. The metric is
cross-entropy on the answer tokens of 500 held-out GSM8K test problems. For every method the learning rate was searched on seed 0 starting from a
three-value prior and extended by 3x toward whichever edge held the best value (at most four runs), so no method was judged on a grid that stops short
of its optimum; the best rate was then rerun on a second seed. **Two seeds** only: the spread between them is about 0.001, which
shows the run-to-run noise from data order and initialization is small, not that the ranking is established beyond this setup.
Ablations (minor selection, relative update) are single-seed. LoRA-FA and OLoRA are supported by the harness but were not run, to fit the thermal
budget of the machine; EVA, CorDA and MiCA were not run.

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

## Whole-model activation memory

Peak GPU memory above the resident weights for one forward and backward through a 4-layer Qwen3-shaped stack (1024 hidden, 3072 MLP), bf16, rank 16, all seven projections adapted, cap 6 GiB. `frozen` backpropagates through the unadapted layers only. `benchmarks/memory.py --model`.

| tokens | frozen | eigentune_diag | eigentune_core | lora | lora_bf16 | dora |
|---|---|---|---|---|---|---|
| 2048 | 366 | 356 | 356 | 645 | 436 | 1182 |
| 8192 | 1463 | 1422 | 1422 | 2581 | 1742 | 4560 |
| 12288 | 2196 | 2134 | 2134 | 3873 | 2614 | OOM |

## Quality against PEFT baselines

Qwen3-0.6B (bf16) fine-tuned on 2,000 GSM8K training solutions for 150 optimizer steps (effective batch 8, sequence 192, cosine schedule), seven attention and MLP projections per layer adapted. Metric: cross-entropy on the answer tokens of 500 held-out GSM8K test problems (lower is better). The pretrained model scores 1.388 on the first 128 of those. Learning rate swept per method on seed 0 (three values), then the best rate rerun on seeds 1 and 2; ablations are single-seed. `benchmarks/compare_peft.py`.

| method | rank | trainable params | adapter file (B) | adapter + bases (B) | runtime bases (B) | eval loss | steps/s | peak VRAM (MiB) | init (s) | lr | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | 8 | 5,046,272 | 20,242,825 | 20,242,825 | 0 | 0.5341 +- 0.0001 | 3.37 | 3588 | 0.6 | 0.001 | 2 |
| dora | 8 | 5,390,336 | 21,645,976 | 21,645,976 | 0 | 0.5343 +- 0.0008 | 1.96 | 3909 | 0.7 | 0.001 | 2 |
| rslora | 8 | 5,046,272 | 20,242,824 | 20,242,824 | 0 | 0.5344 +- 0.0007 | 3.38 | 3588 | 0.6 | 0.0003 | 2 |
| lora_plus | 8 | 5,046,272 | 20,242,825 | 20,242,825 | 0 | 0.5360 +- 0.0019 | 3.37 | 3588 | 0.6 | 0.00015 | 2 |
| pissa | 8 | 5,046,272 | 20,242,837 | 20,242,837 | 0 | 0.5449 +- 0.0009 | 3.40 | 3588 | 1.4 | 0.0001 | 2 |
| lora | 1 | 630,784 | 2,580,176 | 2,580,176 | 0 | 0.5469 +- 0.0004 | 3.51 | 3537 | 0.6 | 0.003 | 2 |
| eigentune_diag | 64 | 12,544 | 97,482 | 80,947,537 | 80,790,528 | 0.6119 +- 0.0014 | 3.36 | 3608 | 14.5 | 0.1 | 2 |
| eigentune_core | 8 | 12,544 | 97,483 | 10,254,418 | 10,098,816 | 0.6285 +- 0.0008 | 3.33 | 3540 | 4.5 | 0.03 | 2 |
| eigentune_core_minor | 8 | 12,544 | 97,447 | 10,254,382 | 10,098,816 | 0.6323 +- 0.0000 | 3.31 | 3541 | 11.9 | 0.03 | 1 |
| eigentune_diag | 16 | 3,136 | 59,587 | 20,316,426 | 20,197,632 | 0.6635 +- 0.0009 | 3.38 | 3550 | 5.6 | 0.3 | 2 |
| eigentune_diag_relative | 16 | 3,136 | 59,587 | 20,316,426 | 20,197,632 | 0.6656 +- 0.0000 | 3.34 | 3550 | 5.6 | 0.09 | 1 |
| eigentune_diag_minor | 16 | 3,136 | 59,563 | 20,316,402 | 20,197,632 | 0.6842 +- 0.0000 | 3.40 | 3550 | 11.8 | 0.3 | 1 |

### Learning-rate sweep (seed 0, final eval loss)

| method | rank | learning rates tried (final eval loss) |
|---|---|---|
| lora | 8 | 0.0003: 0.5441; 0.001: 0.5342; 0.003: 0.6050 |
| dora | 8 | 0.0003: 0.5443; 0.001: 0.5351; 0.003: 0.6074 |
| rslora | 8 | 0.0001: 0.5504; 0.0003: 0.5336; 0.001: 0.5538 |
| lora_plus | 8 | 5e-05: 0.5424; 0.00015: 0.5379; 0.0005: 0.5837; 0.0015: 0.9327 |
| pissa | 8 | 3.33333e-05: 0.5720; 0.0001: 0.5440; 0.0003: 0.5492; 0.001: 0.6250 |
| lora | 1 | 0.001: 0.5498; 0.003: 0.5473; 0.01: 0.6591 |
| eigentune_diag | 64 | 0.03: 0.6243; 0.1: 0.6105; 0.3: 0.6476 |
| eigentune_core | 8 | 0.01: 0.6774; 0.03: 0.6277; 0.1: 0.6319 |
| eigentune_core_minor | 8 | 0.03: 0.6323 |
| eigentune_diag | 16 | 0.1: 0.6713; 0.3: 0.6626; 0.9: 0.6905 |
| eigentune_diag_relative | 16 | 0.03: 0.6728; 0.09: 0.6656 |
| eigentune_diag_minor | 16 | 0.3: 0.6842 |

## Training-step latency by backend

One forward and backward through the same 4-layer stack, rank 16, median of 5 repetitions; the figure in brackets is the speedup over the `torch` backend. Differences under about 20% at small token counts are within run-to-run noise (launch-bound eager execution). `benchmarks/training.py`.

| method | tokens | torch | triton | native | auto |
|---|---|---|---|---|---|
| diagonal | 1 | 10.51 ms (x1.00) | 10.78 ms (x0.97) | 7.94 ms (x1.32) | 8.02 ms (x1.31) |
| diagonal | 16 | 8.43 ms (x1.00) | 10.93 ms (x0.77) | 8.11 ms (x1.04) | 9.07 ms (x0.93) |
| diagonal | 768 | 11.06 ms (x1.00) | 11.07 ms (x1.00) | 9.20 ms (x1.20) | 8.94 ms (x1.24) |
| diagonal | 3072 | 40.29 ms (x1.00) | 38.91 ms (x1.04) | 40.98 ms (x0.98) | 38.94 ms (x1.03) |
| diagonal | 8192 | 154.21 ms (x1.00) | 150.15 ms (x1.03) | 157.41 ms (x0.98) | 154.67 ms (x1.00) |
| spectral_core | 1 | 8.41 ms (x1.00) | 10.85 ms (x0.78) | 8.45 ms (x1.00) | 8.24 ms (x1.02) |
| spectral_core | 16 | 11.19 ms (x1.00) | 10.95 ms (x1.02) | 8.70 ms (x1.29) | 8.81 ms (x1.27) |
| spectral_core | 768 | 9.39 ms (x1.00) | 11.28 ms (x0.83) | 9.72 ms (x0.97) | 9.42 ms (x1.00) |
| spectral_core | 3072 | 40.66 ms (x1.00) | 38.89 ms (x1.05) | 40.78 ms (x1.00) | 38.90 ms (x1.05) |
| spectral_core | 8192 | 154.87 ms (x1.00) | 153.14 ms (x1.01) | 154.69 ms (x1.00) | 151.57 ms (x1.02) |
