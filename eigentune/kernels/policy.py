"""Which backend ``backend="auto"`` prefers. Thresholds come from ``benchmarks/kernels.py`` results."""

from __future__ import annotations

from typing import List

import torch


def preference(x: torch.Tensor, rank: int, kind: str) -> List[str]:
    """Backends to try, best first. ``torch`` is always the last resort."""
    return ["torch"]
