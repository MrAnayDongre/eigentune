"""The adapted linear layer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import EigenTuneConfig
from .kernels import eigentune_update
from .svd import Bases


class EigenTuneLinear(nn.Module):
    """``y = base(x) + (x V_r^T) (scale) U_r^T`` with frozen ``U_r``/``V_r`` and a trainable diagonal or core.

    The bases are non-persistent buffers: they are not in ``state_dict()`` (so a Trainer checkpoint does not
    carry them) and are rebuilt from the base weight when an adapter is loaded.
    """

    def __init__(
        self, base: nn.Module, bases: Bases, cfg: EigenTuneConfig, name: str = "", dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.name = name
        self.cfg = cfg
        self.kind = cfg.kind
        self.rank = bases.U.shape[1]
        self.fingerprint = bases.fingerprint
        self.indices = bases.indices
        self.merged = False
        dtype = dtype or bases.U.dtype
        self.register_buffer("U", bases.U.to(dtype), persistent=False)
        self.register_buffer("Vh", bases.Vh.to(dtype), persistent=False)
        self.register_buffer("S", bases.S.float(), persistent=False)
        if self.kind == "diag":
            self.delta = nn.Parameter(torch.zeros(self.rank, device=bases.U.device))
            self.mask = None
        else:
            self.core = nn.Parameter(torch.zeros(self.rank, self.rank, device=bases.U.device))
            if cfg.core_bandwidth is None:
                self.mask = None
            else:
                i = torch.arange(self.rank, device=bases.U.device)
                self.register_buffer(
                    "mask", ((i[:, None] - i[None, :]).abs() <= cfg.core_bandwidth).float(), persistent=False
                )

    # ------------------------------------------------------------------ parameters
    @property
    def in_features(self) -> int:
        return self.Vh.shape[1]

    @property
    def out_features(self) -> int:
        return self.U.shape[0]

    def trainable(self) -> nn.Parameter:
        return self.delta if self.kind == "diag" else self.core

    def effective(self, dtype: torch.dtype) -> torch.Tensor:
        """The vector or matrix the kernels multiply by, after the band mask and the relative scaling."""
        w = self.trainable()
        if self.mask is not None:
            w = w * self.mask
        if self.cfg.update == "relative":
            w = w * self.S if self.kind == "diag" else w * self.S[:, None]
        return w.to(dtype)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.merged:
            return y
        dt = y.dtype  # compute in the base output's dtype (under autocast that is the low-precision one)
        U, Vh = (self.U, self.Vh) if self.U.dtype == dt else (self.U.to(dt), self.Vh.to(dt))
        out = eigentune_update(
            x.reshape(-1, x.shape[-1]).to(dt),
            y.reshape(-1, y.shape[-1]),
            Vh,
            U,
            self.effective(dt),
            self.kind,
            self.cfg.backend,
        )
        return out.view(y.shape)

    # ------------------------------------------------------------------ merge
    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """``U_r C V_r^T`` in float32, ``[out, in]``."""
        w = self.effective(torch.float32)
        U, Vh = self.U.float(), self.Vh.float()
        return (U * w) @ Vh if self.kind == "diag" else (U @ w) @ Vh

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the update into the base weight (plain ``nn.Linear`` bases only)."""
        if self.merged:
            return
        weight = getattr(self.base, "weight", None)
        if not isinstance(self.base, nn.Linear) or not weight.is_floating_point():
            raise NotImplementedError("merge needs a plain floating-point nn.Linear base layer")
        weight.add_(self.delta_weight().to(weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        weight = self.base.weight
        weight.sub_(self.delta_weight().to(weight.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return f"rank={self.rank}, kind={self.kind}, update={self.cfg.update}, backend={self.cfg.backend}"
