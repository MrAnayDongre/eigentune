# Kernels and dispatch

EigenTune's extra work per layer is two skinny products (`X V_rᵀ` with a small output width, then `Z U_rᵀ` with a small inner
dimension) and a scale in between. With `r` in the tens, that is bandwidth- and launch-bound, not FLOP-bound, so the
interesting kernels are about what they fuse, not how fast they multiply.

## Backends

| backend | what it is | fuses |
|---|---|---|
| `torch` | reference. `Q = X @ Vhᵀ`, `Z = Q * w` (or `Q @ wᵀ`), `torch.addmm(base, Z, Uᵀ)` | the residual add into the up-projection GEMM (`addmm`) |
| `triton` | `down` / `up` / `bwd_down` kernels in `triton_backend.py` (CUDA and ROCm) | scale and residual add in the `up` epilogue (no `[N, r]` scaled tensor, no extra `[N, out]` pass); `Q⊙P` reduction and `P·scale` in the backward `down` epilogue |
| `native` | `csrc/eigentune_ops.cu`, JIT-built | forward: one partial-sum kernel plus one kernel that reduces, scales, multiplies by `Uᵀ` and adds the base output; backward: the `grad_δ` reduction and `P ⊙ δ` in one pass (diagonal only) |

Details that matter:

* The rank dimension is padded to a power of two (at least 16, the minimum for `tl.dot`) and kept in one tile.
* `fp32` inputs use `input_precision="ieee"`, not TF32. 16-bit inputs accumulate in fp32.
* `grad_w` is reduced from per-block partials with an ordinary sum: deterministic, no atomics.
* The Triton core method supports rank up to 64 (the `r × r` matrix lives in registers); the diagonal up to 256.
* The native kernels need `in_features` to be a multiple of 8 (16 bytes of 16-bit data) and `tokens × rank ≤ 2048`
  (shared-memory staging); outside that they report `supports() == False` and the call goes elsewhere.

## Dispatch (`backend="auto"`)

`kernels/policy.py`, measured on one machine (RTX 5070 Laptop, bf16 and fp16, 2048-8192 wide layers, ranks 8/16/64):

| phase | condition | backend |
|---|---|---|
| forward | `tokens × rank ≤ 128` | `native` |
| forward | `tokens ≥ 2048` | `triton` |
| backward, diagonal | `tokens × rank ≤ 1024` | `native` |
| backward | `tokens ≥ 2048` and (core or `rank ≥ 16`) | `triton` |
| otherwise, and any fp32 / ROCm / CPU | | `torch` |

The cuBLAS path wins in between, and eager Triton loses below ~2k tokens because of its Python launch overhead.
Full tables, including the cases where each backend loses, are in [benchmarks](benchmarks.md). The thresholds are
module constants in `eigentune.kernels.policy`, so they can be re-measured with `benchmarks/kernels.py` and overridden for
other hardware.

## Adding a backend

```python
from eigentune.kernels import register_backend

class MyBackend:
    name = "mine"
    def available(self): ...
    def supports(self, x, rank, kind): ...        # forward: x is [N, in]; kind is "diag" or "core"
    def supports_bwd(self, q, rank, kind): ...    # backward: q is [N, r]
    def forward(self, x, Vh, U, w, kind, base_out): ...   # -> (y, Q)
    def backward(self, g, Q, Vh, U, w, kind, need_x): ... # -> (grad_x or None, grad_w)

register_backend(MyBackend())
```

Then run `pytest tests/kernels` (it picks up every registered non-`torch` backend automatically) and
`benchmarks/kernels.py` to see where it wins.
