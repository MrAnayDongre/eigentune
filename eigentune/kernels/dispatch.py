"""Backend registry and selection."""

from __future__ import annotations

from typing import Dict, List, Protocol

import torch


class Backend(Protocol):
    name: str

    def available(self) -> bool: ...
    def supports(self, x: torch.Tensor, rank: int, kind: str) -> bool: ...
    def forward(self, x, Vh, U, w, kind, base_out): ...
    def backward(self, g, Q, Vh, U, w, kind, need_x): ...


_REGISTRY: Dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    """Add (or replace) a backend. Third-party kernels can register themselves from their own package."""
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_backends() -> List[str]:
    return [n for n, b in _REGISTRY.items() if b.available()]


def select_backend(requested: str, x: torch.Tensor, rank: int, kind: str) -> Backend:
    """Resolve ``requested`` (a name or ``"auto"``) to a backend that can run this call.

    A named backend that cannot run the call falls back to ``torch`` rather than failing; ``auto``
    picks the accelerator only where benchmarks showed it wins (see ``eigentune.kernels.policy``).
    """
    from . import policy

    if requested == "auto":
        for name in policy.preference(x, rank, kind):
            b = _REGISTRY.get(name)
            if b is not None and b.available() and b.supports(x, rank, kind):
                return b
        return _REGISTRY["torch"]
    b = get_backend(requested)
    return b if b.available() and b.supports(x, rank, kind) else _REGISTRY["torch"]
