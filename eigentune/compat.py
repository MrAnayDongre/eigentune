"""The 0.1 ``EigenTunedLayer`` constructor, kept so existing code keeps working. New code uses ``EigenTuneLinear``."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .config import EigenTuneConfig
from .layers import EigenTuneLinear
from .svd import compute_bases


class EigenTunedLayer(EigenTuneLinear):
    """Deprecated: ``EigenTunedLayer(original_layer, rank, full_precision_weight)``."""

    def __init__(self, original_layer: nn.Module, rank: int, full_precision_weight: torch.Tensor):
        warnings.warn(
            "EigenTunedLayer is deprecated; use get_eigentune_model or EigenTuneLinear",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg = EigenTuneConfig(rank=rank, svd_backend="exact")
        super().__init__(original_layer, compute_bases(full_precision_weight, cfg), cfg, dtype=torch.float32)

    @property
    def original_layer(self) -> nn.Module:
        return self.base

    @property
    def delta_s(self) -> nn.Parameter:
        return self.delta
