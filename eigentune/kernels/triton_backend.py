"""Triton backend (NVIDIA CUDA and AMD ROCm).

EigenTune is two skinny GEMMs around a tiny scale, with ``r`` in the tens. The kernels here keep the whole rank
dimension in one tile, which is what makes fusing worthwhile:

* ``down``:  ``Q = A @ B`` with ``K`` large, ``N`` rows, ``r`` columns.
* ``up``:    ``out = base + scale(Z) @ B^T`` with ``K = r``: the scale and the residual add live in the epilogue, so the
  ``[N, r]`` scaled activation and the extra ``[N, out]`` add pass never touch memory.
* ``bwd_down``: ``P = G @ U`` whose epilogue also produces the ``grad_w`` partial sums and ``P * scale``.

``grad_w`` is reduced from per-block partials with an ordinary sum, so results are deterministic (no atomics).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)
_KINDS = {"diag": 0, "core": 1, "none": 2}  # "none": no scale (used for grad_x, already scaled)
MAX_RANK_DIAG = 256
MAX_RANK_CORE = 64


@triton.jit
def _down_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    K,
    R,
    sam,
    sak,
    sbk,
    sbr,
    som,
    sor,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    PREC: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rr = tl.arange(0, BR)
    acc = tl.zeros((BM, BR), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(
            a_ptr + rm[:, None] * sam + rk[None, :] * sak, mask=(rm[:, None] < N) & (rk[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptr + rk[:, None] * sbk + rr[None, :] * sbr, mask=(rk[:, None] < K) & (rr[None, :] < R), other=0.0
        )
        acc = tl.dot(a, b, acc, input_precision=PREC)
    tl.store(
        out_ptr + rm[:, None] * som + rr[None, :] * sor,
        acc.to(out_ptr.dtype.element_ty),
        mask=(rm[:, None] < N) & (rr[None, :] < R),
    )


@triton.jit
def _up_kernel(
    z_ptr,
    w_ptr,
    b_ptr,
    base_ptr,
    out_ptr,
    N,
    M,
    R,
    szm,
    szr,
    swr,
    swc,
    sbn,
    sbr,
    sbasem,
    sbasen,
    som,
    son,
    KIND: tl.constexpr,
    HAS_BASE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    PREC: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rr = tl.arange(0, BR)
    z = tl.load(z_ptr + rm[:, None] * szm + rr[None, :] * szr, mask=(rm[:, None] < N) & (rr[None, :] < R), other=0.0)
    if KIND == 0:  # diagonal: Z * w
        w = tl.load(w_ptr + rr * swr, mask=rr < R, other=0.0)
        z = (z.to(tl.float32) * w.to(tl.float32)[None, :]).to(z_ptr.dtype.element_ty)
    elif KIND == 1:  # core: Z @ w^T
        wt = tl.load(
            w_ptr + rr[:, None] * swc + rr[None, :] * swr, mask=(rr[:, None] < R) & (rr[None, :] < R), other=0.0
        )
        z = tl.dot(z, wt.to(z_ptr.dtype.element_ty), input_precision=PREC).to(z_ptr.dtype.element_ty)
    b = tl.load(b_ptr + rr[:, None] * sbr + rn[None, :] * sbn, mask=(rr[:, None] < R) & (rn[None, :] < M), other=0.0)
    acc = tl.dot(z, b, input_precision=PREC)
    mask = (rm[:, None] < N) & (rn[None, :] < M)
    if HAS_BASE:
        acc += tl.load(base_ptr + rm[:, None] * sbasem + rn[None, :] * sbasen, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + rm[:, None] * som + rn[None, :] * son, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_down_kernel(
    g_ptr,
    u_ptr,
    q_ptr,
    w_ptr,
    gq_ptr,
    gw_ptr,
    N,
    M,
    R,
    sgm,
    sgk,
    suk,
    sur,
    sqm,
    sqr,
    swr,
    swc,
    sgqm,
    sgqr,
    KIND: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    PREC: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rr = tl.arange(0, BR)
    acc = tl.zeros((BM, BR), tl.float32)
    for k0 in range(0, M, BK):
        rk = k0 + tl.arange(0, BK)
        g = tl.load(
            g_ptr + rm[:, None] * sgm + rk[None, :] * sgk, mask=(rm[:, None] < N) & (rk[None, :] < M), other=0.0
        )
        u = tl.load(
            u_ptr + rk[:, None] * suk + rr[None, :] * sur, mask=(rk[:, None] < M) & (rr[None, :] < R), other=0.0
        )
        acc = tl.dot(g, u, acc, input_precision=PREC)  # P = G @ U  (fp32)
    q = tl.load(q_ptr + rm[:, None] * sqm + rr[None, :] * sqr, mask=(rm[:, None] < N) & (rr[None, :] < R), other=0.0)
    q32 = q.to(tl.float32)
    if KIND == 0:
        w = tl.load(w_ptr + rr * swr, mask=rr < R, other=0.0).to(tl.float32)
        tl.store(gw_ptr + pid * BR + rr, tl.sum(q32 * acc, axis=0))  # partial grad_w for this block of rows
        gq = acc * w[None, :]
    else:
        # dL/dC = P^T Q;  gQ = P @ C
        part = tl.dot(tl.trans(acc), q32, input_precision="ieee")
        ri = tl.arange(0, BR)
        tl.store(gw_ptr + pid * BR * BR + ri[:, None] * BR + rr[None, :], part)
        wm = tl.load(
            w_ptr + rr[:, None] * swr + rr[None, :] * swc, mask=(rr[:, None] < R) & (rr[None, :] < R), other=0.0
        )
        gq = tl.dot(acc, wm.to(tl.float32), input_precision="ieee")
    tl.store(
        gq_ptr + rm[:, None] * sgqm + rr[None, :] * sgqr,
        gq.to(gq_ptr.dtype.element_ty),
        mask=(rm[:, None] < N) & (rr[None, :] < R),
    )


def _prec(t: torch.Tensor) -> str:
    return "ieee" if t.dtype == torch.float32 else "tf32"  # "tf32" is ignored for 16-bit inputs


def _block_m(n: int) -> int:
    return max(16, min(64, triton.next_power_of_2(n)))


def _block_k(k: int) -> int:
    return 128 if k >= 2048 else 64


def _pad_rank(r: int) -> int:
    return max(16, triton.next_power_of_2(r))


def down(a: torch.Tensor, b: torch.Tensor, b_k_dim: int) -> torch.Tensor:
    """``A [N, K] @ B`` where ``B`` is ``[r, K]`` (``b_k_dim=1``) or ``[K, r]`` (``b_k_dim=0``)."""
    N, K = a.shape
    R = b.shape[1 - b_k_dim]
    out = torch.empty(N, R, device=a.device, dtype=a.dtype)
    BM = _block_m(N)
    _down_kernel[(triton.cdiv(N, BM),)](
        a,
        b,
        out,
        N,
        K,
        R,
        a.stride(0),
        a.stride(1),
        b.stride(b_k_dim),
        b.stride(1 - b_k_dim),
        out.stride(0),
        out.stride(1),
        BM=BM,
        BK=_block_k(K),
        BR=_pad_rank(R),
        PREC=_prec(a),
        num_warps=4,
    )
    return out


def up(z: torch.Tensor, w: torch.Tensor, b: torch.Tensor, b_n_dim: int, kind: str, base=None) -> torch.Tensor:
    """``base + scale(Z) @ B^T``; ``B`` is ``[M, r]`` (``b_n_dim=0``) or ``[r, M]`` (``b_n_dim=1``)."""
    N, R = z.shape
    M = b.shape[b_n_dim]
    out = torch.empty(N, M, device=z.device, dtype=z.dtype)
    BM = _block_m(N)
    base_t = base if base is not None else out
    swr = w.stride(0) if w.dim() else 0
    swc = w.stride(1) if w.dim() == 2 else 0
    _up_kernel[(triton.cdiv(N, BM), triton.cdiv(M, 128))](
        z,
        w,
        b,
        base_t,
        out,
        N,
        M,
        R,
        z.stride(0),
        z.stride(1),
        swr,
        swc,
        b.stride(b_n_dim),
        b.stride(1 - b_n_dim),
        base_t.stride(0),
        base_t.stride(1),
        out.stride(0),
        out.stride(1),
        KIND=_KINDS[kind],
        HAS_BASE=base is not None,
        BM=BM,
        BN=128,
        BR=_pad_rank(R),
        PREC=_prec(z),
        num_warps=4,
    )
    return out


class TritonBackend:
    name = "triton"

    def available(self) -> bool:
        return torch.cuda.is_available()

    def supports(self, x: torch.Tensor, rank: int, kind: str) -> bool:
        limit = MAX_RANK_DIAG if kind == "diag" else MAX_RANK_CORE
        return x.is_cuda and x.dtype in _SUPPORTED and 1 <= rank <= limit and x.shape[0] > 0 and x.shape[1] > 0

    def supports_bwd(self, q: torch.Tensor, rank: int, kind: str) -> bool:
        return self.supports(q, rank, kind)

    def forward(self, x, Vh, U, w, kind, base_out):
        x = x if x.stride(1) == 1 else x.contiguous()
        Q = down(x, Vh, b_k_dim=1)  # [N, r] = X Vh^T
        base_out = base_out if base_out.stride(1) == 1 else base_out.contiguous()
        return up(Q, w, U, b_n_dim=0, kind=kind, base=base_out), Q  # base + scale(Q) U^T

    def backward(self, g, Q, Vh, U, w, kind, need_x):
        N, R = Q.shape
        BM = _block_m(N)
        BR = _pad_rank(R)
        blocks = triton.cdiv(N, BM)
        gq = torch.empty(N, R, device=g.device, dtype=g.dtype)
        part = torch.empty(blocks, BR if kind == "diag" else BR * BR, device=g.device, dtype=torch.float32)
        swr = w.stride(0)
        swc = w.stride(1) if w.dim() == 2 else 0
        _bwd_down_kernel[(blocks,)](
            g,
            U,
            Q,
            w,
            gq,
            part,
            N,
            U.shape[0],
            R,
            g.stride(0),
            g.stride(1),
            U.stride(0),
            U.stride(1),
            Q.stride(0),
            Q.stride(1),
            swr,
            swc,
            gq.stride(0),
            gq.stride(1),
            KIND=0 if kind == "diag" else 1,
            BM=BM,
            BK=_block_k(U.shape[0]),
            BR=BR,
            PREC=_prec(g),
            num_warps=4,
        )
        if kind == "diag":
            gw = part.sum(0)[:R]
        else:
            gw = part.sum(0).view(BR, BR)[:R, :R]
        gx = up(gq, w, Vh, b_n_dim=1, kind="none") if need_x else None  # gQ is already scaled
        return gx, gw
