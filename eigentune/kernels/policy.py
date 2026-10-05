"""Which backend ``backend="auto"`` prefers, per phase. Every number here comes from ``benchmarks/kernels.py``.

Measured on one machine (NVIDIA RTX 5070 Laptop, CUDA 13, bf16 and fp16, 2048-8192 wide layers, ranks 8/16/64);
see ``docs/benchmarks.md`` for the tables, including the regimes where each backend loses. Thresholds are module
constants so they can be overridden for other hardware. Anything not measured (fp32, ROCm) stays on ``torch``.
"""

from __future__ import annotations

from typing import List

import torch

NATIVE_FWD_MAX_TOKENS_X_RANK = 128    # native forward wins while tokens * rank is small (launch/latency bound)
NATIVE_BWD_MAX_TOKENS_X_RANK = 1024   # native fused reduction (diagonal only)
TRITON_MIN_TOKENS = 2048              # Triton wins once the GEMMs are bandwidth bound
TRITON_BWD_MIN_RANK_DIAG = 16         # below this the diagonal backward is a coin flip against cuBLAS

_MEASURED_DTYPES = (torch.bfloat16, torch.float16)


def preference(x: torch.Tensor, rank: int, kind: str, phase: str = "fwd") -> List[str]:
    """Backends to try, best first. ``torch`` is always the last resort."""
    if not x.is_cuda or torch.version.hip is not None or x.dtype not in _MEASURED_DTYPES:
        return ["torch"]
    n = x.shape[0]
    if phase == "fwd":
        if n * rank <= NATIVE_FWD_MAX_TOKENS_X_RANK:
            return ["native", "torch"]
        if n >= TRITON_MIN_TOKENS:
            return ["triton", "torch"]
    else:
        if kind == "diag" and n * rank <= NATIVE_BWD_MAX_TOKENS_X_RANK:
            return ["native", "torch"]
        if n >= TRITON_MIN_TOKENS and (kind == "core" or rank >= TRITON_BWD_MIN_RANK_DIAG):
            return ["triton", "torch"]
    return ["torch"]
