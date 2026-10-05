"""Backend registry and selection."""

from __future__ import annotations

import warnings
from typing import Dict, List, Protocol

import torch


class Backend(Protocol):
    name: str

    def available(self) -> bool: ...
    def supports(self, x: torch.Tensor, rank: int, kind: str) -> bool: ...
    def supports_bwd(self, q: torch.Tensor, rank: int, kind: str) -> bool: ...
    def forward(self, x, Vh, U, w, kind, base_out): ...
    def backward(self, g, Q, Vh, U, w, kind, need_x): ...


_REGISTRY: Dict[str, Backend] = {}
_OPTIONAL = {"triton", "native"}  # accelerators that are legitimately absent on some installs
_warned: set = set()


def register_backend(backend: Backend) -> None:
    """Add (or replace) a backend. Third-party kernels can register themselves from their own package."""
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_backends() -> List[str]:
    return [n for n, b in _REGISTRY.items() if b.available()]


def _ok(b: Backend, x: torch.Tensor, rank: int, kind: str, phase: str) -> bool:
    if not b.available():
        return False
    check = getattr(b, "supports_bwd", None) if phase == "bwd" else None
    if check is None:  # explicit None test: older Dynamo cannot trace the truthiness of a bound method
        check = b.supports
    return check(x, rank, kind)


def select_backend(requested: str, x: torch.Tensor, rank: int, kind: str, phase: str = "fwd") -> Backend:
    """Resolve ``requested`` (a name or ``"auto"``) to a backend that can run this call.

    A named backend that cannot run the call falls back to ``torch`` rather than failing; ``auto`` follows
    ``eigentune.kernels.policy`` (benchmark-derived). ``x`` is the ``[N, in]`` input for ``phase="fwd"`` and the
    ``[N, r]`` activation for ``phase="bwd"``.
    """
    from . import policy

    if requested == "auto":
        for name in policy.preference(x, rank, kind, phase):
            b = _REGISTRY.get(name)
            if b is not None and _ok(b, x, rank, kind, phase):
                return b
        return _REGISTRY["torch"]
    b = _REGISTRY.get(requested)
    if b is None:
        if requested not in _OPTIONAL:
            get_backend(requested)  # raises the "unknown backend" error for typos
        if requested not in _warned:
            _warned.add(requested)
            warnings.warn(f"backend {requested!r} is not available in this installation; using 'torch'", stacklevel=3)
        return _REGISTRY["torch"]
    return b if _ok(b, x, rank, kind, phase) else _REGISTRY["torch"]
