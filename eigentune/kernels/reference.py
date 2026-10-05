"""The reference backend: ordinary PyTorch ops. This is the correctness oracle for every other backend."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class TorchBackend:
    name = "torch"

    def available(self) -> bool:
        return True

    def supports(self, x: torch.Tensor, rank: int, kind: str) -> bool:
        return True

    def supports_bwd(self, q: torch.Tensor, rank: int, kind: str) -> bool:
        return True

    def forward(self, x, Vh, U, w, kind, base_out) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = x @ Vh.T  # [N, r]
        Z = Q * w if kind == "diag" else Q @ w.T  # [N, r]
        return torch.addmm(base_out, Z, U.T), Q  # base_out + Z U^T, fused into one GEMM

    def backward(self, g, Q, Vh, U, w, kind, need_x) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        P = g @ U  # [N, r]
        if kind == "diag":
            gw = (Q.float() * P.float()).sum(0)  # reductions over tokens accumulate in fp32
            gQ = P * w
        else:
            gw = P.float().T @ Q.float()  # [r, r]: dL/dC[j, k] = sum_n P[n, j] Q[n, k]
            gQ = P @ w
        return (gQ @ Vh if need_x else None), gw
