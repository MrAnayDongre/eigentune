# EigenTune 0.2 design

Goals: a trustworthy reference, honest accounting of storage and memory, optional accelerated paths that earn their
place by benchmark, and a layout other people can extend. Non-goals: a new adaptation algorithm (see
[RELATED_WORK](../research/RELATED_WORK.md)), PEFT-registry integration, quantized-base support beyond what is stated in
[compatibility](../compatibility.md).

## Decisions, with the evidence behind them

| decision | why | evidence |
|---|---|---|
| Keep only rank-`r` bases, as non-persistent buffers, in the compute dtype | 0.1 kept the full `U`/`Vh` in fp32: 4x the weight at runtime, 6x in the saved file | [V2_BASELINE](V2_BASELINE.md) |
| Rebuild bases on load; verify with a fingerprint and a signature; `save_bases=True` to embed | adapter file stays ~60 KB for 3,136 parameters; an exact-portability option exists for the cases SVD cannot guarantee | `tests/serialization`, [serialization](../serialization.md) |
| Canonical sign for every singular pair | a dense core is not invariant to sign flips | `test_sign_convention_is_canonical` |
| Randomized SVD iterates to convergence, capped | a fixed 2-4 iterations gave 40-85% reconstruction error on real Llama-2-7B weights; converged gives <= 1.6e-4 in 0.07-0.45 s on GPU (5.2 s exact) | `benchmarks/results/initialization.json`, `test_converged_randomized_svd_is_accurate...` |
| `auto` SVD choice depends on shape and rank only, not device | the algorithm is part of the basis fingerprint; an adapter trained on a GPU must load on a CPU | `fingerprint` in `svd.py` |
| One `autograd.Function` for all backends | lets backends replace forward and backward independently | `kernels/autograd.py` |
| Custom autograd is *not* sold as a memory saving | plain autograd with frozen buffers already keeps only `Q` (0.25 MiB vs a 64 MiB input) | `benchmarks/memory.py` |
| Two kernels (down, up) with epilogue fusion rather than one monolithic kernel | see below | kernel benchmarks |
| Backend choice per phase | the best forward and backward backends differ | `benchmarks/results/kernels_*.json` |
| Dispatch thresholds are measured constants; unmeasured regimes use `torch` | a custom kernel that loses must not be selected | `kernels/policy.py` |
| No PEFT dependency | PEFT's extension point is its private mapping tables; 0.1's "build LoRA then swap" broke on key conventions | [architecture](../architecture.md) |
| Deterministic reductions | per-block partials summed in order, no atomics | `test_deterministic` |

## Fusion: what was and was not tried

Tried and kept: the scale and the residual add fused into the up-projection epilogue (Triton and native), and the
`Q ⊙ P` reduction plus `P · scale` fused into the backward down-projection epilogue.

**Not implemented:** a single kernel that computes `Q` and the output in one pass. It would have to recompute `Q` for
every output tile (reading `X` `out / BN` times) or synchronise across blocks. That is an argument from the structure of
the problem, not a measurement; it was not benchmarked.

## What is deliberately missing

SSO-style integrations, 4-bit/8-bit bases tested end to end, FSDP, multi-GPU, ROCm hardware validation, and a published
training recipe. Each is listed with its status in [compatibility](../compatibility.md).
