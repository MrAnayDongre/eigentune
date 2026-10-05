"""Native CUDA/HIP backend: JIT-built on first use, optional, never required.

Forward is the hand-written kernel pair for few tokens; the large GEMMs of the backward stay on the vendor library
(``torch.matmul``) and only the fused scale/reduction is custom. Anything it cannot run falls back to ``torch``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .reference import TorchBackend

_ext = None
_failed = None
_SRC = Path(__file__).resolve().parent.parent / "csrc" / "eigentune_ops.cu"
MAX_TOKENS = 256
MAX_RANK = 64
MAX_ELEMS = 2048  # tokens * rank must fit the kernel's shared-memory staging


def _load():
    global _ext, _failed
    if _ext is not None or _failed is not None:
        return _ext
    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="eigentune_native",
            sources=[str(_SRC)],
            extra_cuda_cflags=["-O3"],
            verbose=bool(os.environ.get("EIGENTUNE_VERBOSE_BUILD")),
        )
    except Exception as exc:  # no toolchain, unsupported arch, ...: the backend just reports unavailable
        _failed = exc
    return _ext


class NativeBackend(TorchBackend):
    """Inherits the reference ops for anything it does not accelerate (core backward, large GEMMs)."""

    name = "native"

    def available(self) -> bool:
        return torch.cuda.is_available() and _SRC.exists() and _load() is not None

    def supports(self, x: torch.Tensor, rank: int, kind: str) -> bool:
        vec = 16 // x.element_size()
        return (
            x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 1 <= x.shape[0] <= MAX_TOKENS
            and 1 <= rank <= MAX_RANK
            and x.shape[0] * rank <= MAX_ELEMS
            and x.shape[1] % vec == 0
        )

    def supports_bwd(self, q: torch.Tensor, rank: int, kind: str) -> bool:
        return q.is_cuda and q.dtype in (torch.float16, torch.bfloat16, torch.float32) and kind == "diag" and rank >= 1

    def forward(self, x, Vh, U, w, kind, base_out):
        x, base_out = x.contiguous(), base_out.contiguous()
        y, Q = _ext.forward(x, Vh.contiguous(), U.contiguous(), w.contiguous(), kind == "core", base_out)
        return y, Q

    def backward(self, g, Q, Vh, U, w, kind, need_x):
        if kind != "diag":
            return super().backward(g, Q, Vh, U, w, kind, need_x)
        P = g @ U
        gw, gQ = _ext.bwd_reduce(Q.contiguous(), P.contiguous(), w.contiguous())
        return (gQ @ Vh if need_x else None), gw
