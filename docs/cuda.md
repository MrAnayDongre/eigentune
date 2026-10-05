# CUDA

The native backend is `eigentune/csrc/eigentune_ops.cu`, built on first use with `torch.utils.cpp_extension.load`
(about 30 s with nvcc 13.0, cached afterwards). It is optional: `pip install eigentune` needs no CUDA toolkit, and if the
build fails the backend reports itself unavailable and everything runs on `torch`.

## Kernels

* `down_partial`: one warp per `(token, rank)` pair and per K-slice; 16-byte vector loads; warp-shuffle reduction; writes an
  fp32 partial per slice. The slice count is chosen to fill the GPU.
* `up_fused`: one pass per block of output columns. It reduces the partials, rounds `Q` to the compute dtype (as the
  reference does), applies the diagonal scale or the `r × r` core in shared memory, multiplies by `U`'s row held in
  registers and adds the base output. Block 0 also writes `Q` for the backward.
* `bwd_reduce`: `grad_δ = Σ_n Q ⊙ P` and `P ⊙ δ` in one pass, reduced in a fixed order.

Large GEMMs (the `P = G U` and `∂L/∂x` products of the backward) stay on cuBLAS: a hand-written kernel does not beat it there.

## Where it wins and loses (RTX 5070 Laptop, sm_120, CUDA 13.0, bf16)

From `benchmarks/results/kernels_bf16.json`. Speedup over the `torch` backend, range over layer shapes (2048-8192) and ranks (8/16/64):

| | 1 token | 4 tokens | 16 tokens | 128 tokens |
|---|---|---|---|---|
| forward, eager | 1.44x - 3.85x | 0.74x - 5.16x | 0.25x - 3.64x | 0.14x - 0.79x |
| backward, eager | 1.65x - 2.79x | 1.17x - 2.00x | 1.18x - 2.11x | 0.89x - 2.58x |

So it is a decode-time kernel: it wins when launches and latency dominate and loses once there is real GEMM work. The dispatcher
reflects that.

## Requirements and limits

* NVIDIA GPU supported by your PyTorch build; verified on compute capability 12.0 only.
* `in_features % 8 == 0` for 16-bit dtypes, `% 4 == 0` for fp32; `tokens × rank ≤ 2048`, `rank ≤ 64`, `tokens ≤ 256`.
* Results are deterministic run to run; they are checked against the reference and an fp64 oracle for fp32, fp16 and bf16.

Set `EIGENTUNE_VERBOSE_BUILD=1` to see the compiler output if the extension fails to build.
