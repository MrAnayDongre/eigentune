# Mathematics

Notation: a frozen linear layer `y = W x` with `W ∈ R^{out×in}`, `W = U Σ Vᵀ`. For a set `I` of `r` singular directions,
`U_r = U[:, I]`, `V_r = V[:, I]`, `σ_r = σ[I]`. Tokens are rows: `X ∈ R^{N×in}`.

## The update

| method | trainable | update to `W` |
|---|---|---|
| `diagonal` (classic) | `δ ∈ R^r` | `ΔW = U_r diag(δ) V_rᵀ` |
| `spectral_core` | `C ∈ R^{r×r}` | `ΔW = U_r C V_rᵀ` |
| `spectral_core`, `core_bandwidth=b` | `C` masked to `|i−j| ≤ b` | same, with `C_ij = 0` outside the band |

`update="relative"` trains `δ` as a multiplier on each singular value (`σ'_i = σ_i (1 + δ_i)`), implemented as
`δ_eff = σ ⊙ δ` (rows of `C` scaled by `σ` for the core). It changes the optimisation geometry, not the function class.

All start at zero, so `W' = W` exactly at step 0 (checked bitwise by `test_identity_at_zero`).
`selection` chooses `I`: the top `r` (`principal`), the bottom `r` (`minor`), or half and half (`mixed`).

## Forward

```
Q  = X V_r                      [N, r]     (stored as Vh = V_rᵀ, so Q = X @ Vh.T)
Z  = Q ⊙ δ        (diagonal)    [N, r]
Z  = Q Cᵀ         (core)
Y  = X Wᵀ + Z U_rᵀ              [N, out]
```

Only `r (in + out)` multiplies per token on top of the base layer, and only `r` or `r²` trained numbers.
The code multiplies `X` by `V_r` and `U_r` directly and never forms the dense `ΔW`.

## Backward

For an upstream gradient `G = ∂L/∂Y`, let `P = G U_r` (`[N, r]`).

```
diagonal:  ∂L/∂δ = Σ_n  Q[n,:] ⊙ P[n,:]            ∂L/∂Q = P ⊙ δ
core:      ∂L/∂C = Pᵀ Q                              ∂L/∂Q = P C
           ∂L/∂X (update path) = (∂L/∂Q) V_rᵀ
```

These are what `kernels/reference.py` computes. They are verified three ways in `tests/correctness/test_math.py`: against
autograd on the explicit formula, with `torch.autograd.gradcheck` in float64, and (on a GPU) every optimised backend
against the reference and an fp64 oracle. Reductions over tokens accumulate in float32 whatever the compute dtype.

## What the backward has to remember

The tensors that must survive until the backward are `Q` (`[N, r]`) and the frozen bases, never `X` (`[N, in]`).
Measured with `saved_tensors_hooks` (`benchmarks/memory.py`), per adapted 4096x4096 layer, 8192 tokens, bf16:

| | MiB kept for backward |
|---|---|
| input `X` | 64 |
| EigenTune, rank 16 | 0.25 |
| LoRA, rank 16, bf16 adapters | 64.25 |
| LoRA, rank 16, PEFT default (fp32 adapters) | 128.5 |
| DoRA, rank 16 | 384.8 |

Two things worth knowing:

* A frozen base `nn.Linear` whose input requires grad keeps nothing for backward (0 MiB measured), so the saving is real
  and comes from the adapter not retaining `X`.
* **Plain autograd already does this** when the bases are frozen buffers: writing the same computation with ordinary
  ops also keeps 0.25 MiB. The custom autograd function is not what saves the memory. It exists so that fused kernels
  can replace both directions, so that the `grad_δ` reduction accumulates in float32, and so that the residual add is
  fused into the up-projection GEMM.

## Sign and degeneracy

A singular pair `(u_i, v_i)` is only defined up to a joint sign. For the diagonal method the sign cancels
(`u_i v_iᵀ` is invariant), but for a dense core it does not, so bases are put in a canonical sign (the largest-magnitude entry of each
`v_i` positive). Repeated singular values make the *subspace* defined but the individual directions arbitrary; EigenTune
warns when the spectrum has no gap at the edge of the selection. See [serialization](serialization.md) for what
that means for loading an adapter on another machine.
