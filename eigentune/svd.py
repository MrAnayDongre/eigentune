"""Computing and caching the frozen singular bases.

Only ``rank`` singular triplets are ever needed, so the strategies here avoid materialising the full
decomposition where they can. All of them return bases with a canonical sign convention, so a basis
recomputed from the same weight is the same basis.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .config import FORMAT_VERSION, EigenTuneConfig


@dataclass
class Bases:
    """``W ~ U diag(S) Vh`` restricted to the selected directions. ``U: [out, r]``, ``Vh: [r, in]``."""

    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor
    indices: Tuple[int, ...]
    fingerprint: str


def select_indices(k_total: int, rank: int, selection: str) -> Tuple[int, ...]:
    """Which of the ``k_total`` singular directions (sorted by decreasing value) are adapted."""
    r = min(rank, k_total)
    if selection == "principal":
        return tuple(range(r))
    if selection == "minor":
        return tuple(range(k_total - r, k_total))
    head = (r + 1) // 2
    return tuple(range(head)) + tuple(range(k_total - (r - head), k_total))


def _canonical_signs(U: torch.Tensor, Vh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fix the sign ambiguity: the largest-magnitude entry of every right singular vector is positive."""
    idx = Vh.abs().argmax(dim=1, keepdim=True)
    sign = torch.sign(torch.gather(Vh, 1, idx))
    sign[sign == 0] = 1
    return U * sign.T, Vh * sign


def _exact(W: torch.Tensor, indices: Tuple[int, ...]):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    _warn_if_degenerate(S, indices)
    ix = torch.tensor(indices, device=W.device)
    return U[:, ix], S[ix], Vh[ix]


def _warn_if_degenerate(S: torch.Tensor, indices: Tuple[int, ...], rel_tol: float = 1e-6) -> None:
    """A selected subspace is only well defined if the spectrum has a gap at every edge of the selection."""
    chosen = set(indices)
    for i in indices:
        for j in (i - 1, i + 1):
            if 0 <= j < len(S) and j not in chosen and float((S[i] - S[j]).abs()) <= rel_tol * float(S[0]):
                warnings.warn(
                    "EigenTune: singular values are (nearly) repeated at the edge of the selected directions, so the "
                    "selected subspace is not unique. Save adapters with save_bases=True to make them portable.",
                    stacklevel=3,
                )
                return


def _randomized(W: torch.Tensor, rank: int, oversampling: int, niter: int, seed: int, tol: float):
    """Randomized subspace iteration (Halko et al.) with a seeded generator, run until the rank-r subspace converges.

    On real LLM weights the spectrum decays slowly, so a fixed small number of iterations is not accurate
    (relative error 0.4-0.8 at niter=2..4). Iterating until the principal-angle change drops below ``tol`` reaches
    ~1e-4 relative reconstruction error in a few dozen iterations, still ~100x faster than a full SVD.
    """
    out, inn = W.shape
    q = min(rank + oversampling, min(out, inn))
    gen = torch.Generator(device=W.device).manual_seed(seed)
    Q = torch.linalg.qr(W @ torch.randn(inn, q, device=W.device, dtype=W.dtype, generator=gen))[0]
    prev = None
    for it in range(1, niter + 1):
        Q = torch.linalg.qr(W @ torch.linalg.qr(W.T @ Q)[0])[0]
        if it % 4 == 0 or it == niter:
            Ub, S, Vh = torch.linalg.svd(Q.T @ W, full_matrices=False)
            U = (Q @ Ub)[:, :rank]
            if prev is not None and (U - prev @ (prev.T @ U)).norm() / rank**0.5 < tol:
                break
            prev = U
    return U, S[:rank], Vh[:rank]


def _lowrank(W: torch.Tensor, rank: int, oversampling: int, niter: int, seed: int):
    devices = [W.device] if W.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        U, S, V = torch.svd_lowrank(W, q=min(rank + oversampling, min(W.shape)), niter=niter)
    return U[:, :rank], S[:rank], V.T[:rank]


def resolve_backend(requested: str, shape: Tuple[int, int], rank: int, selection: str) -> str:
    """``auto``: the converged randomized solver for large matrices and small ranks, exact otherwise."""
    if selection != "principal":
        return "exact"  # truncated solvers find the top of the spectrum; minor/mixed need the tail
    if requested != "auto":
        return requested
    return "randomized" if min(shape) >= 1024 and rank * 4 <= min(shape) else "exact"


def fingerprint(weight: torch.Tensor, cfg: EigenTuneConfig, backend: str) -> str:
    """Identifies a basis: the exact weight bytes and every setting that changes the result."""
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "fmt": FORMAT_VERSION,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "rank": cfg.rank,
                "selection": cfg.selection,
                "backend": backend,
                "svd": [cfg.svd_oversampling, cfg.svd_niter, cfg.svd_seed] + ([cfg.svd_tol] if backend == "randomized" else [])
        if backend != "exact" else None,
            },
            sort_keys=True,
        ).encode()
    )
    h.update(weight.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def compute_bases(weight: torch.Tensor, cfg: EigenTuneConfig, device: Optional[torch.device] = None) -> Bases:
    """Selected singular triplets of ``weight`` ``[out, in]``, computed in float32."""
    if weight.dim() != 2:
        raise ValueError("expected a 2-D weight")
    backend = resolve_backend(cfg.svd_backend, tuple(weight.shape), cfg.rank, cfg.selection)
    fp = fingerprint(weight, cfg, backend)
    cached = _cache_load(cfg.cache_dir, fp, weight.device if device is None else device)
    if cached is not None:
        return cached
    W = weight.detach().to(device=device or weight.device, dtype=torch.float32)
    k_total = min(W.shape)
    indices = select_indices(k_total, cfg.rank, cfg.selection)
    if backend == "exact":
        U, S, Vh = _exact(W, indices)
    elif backend == "randomized":
        U, S, Vh = _randomized(W, len(indices), cfg.svd_oversampling, cfg.svd_niter, cfg.svd_seed, cfg.svd_tol)
    else:
        U, S, Vh = _lowrank(W, len(indices), cfg.svd_oversampling, cfg.svd_niter, cfg.svd_seed)
    U, Vh = _canonical_signs(U, Vh)
    bases = Bases(U.contiguous(), S.contiguous(), Vh.contiguous(), indices, fp)
    _cache_store(cfg.cache_dir, bases)
    return bases


def basis_signature(bases: Bases, seed: int = 0) -> torch.Tensor:
    """A few numbers that change if the basis changes: probe the bases with fixed random vectors."""
    gen = torch.Generator().manual_seed(seed)
    g = torch.randn(bases.Vh.shape[1], generator=gen)
    h = torch.randn(bases.U.shape[0], generator=gen)
    return torch.cat([bases.Vh.cpu().float() @ g, bases.U.cpu().float().T @ h])


# ----------------------------------------------------------------------------- cache
def _cache_path(cache_dir: str, fp: str) -> str:
    return os.path.join(cache_dir, f"{fp}.pt")


def _cache_load(cache_dir: Optional[str], fp: str, device: torch.device) -> Optional[Bases]:
    if not cache_dir:
        return None
    path = _cache_path(cache_dir, fp)
    if not os.path.exists(path):
        return None
    try:
        blob: Dict = torch.load(path, map_location=device, weights_only=True)
        if blob["fingerprint"] != fp:
            return None
        return Bases(blob["U"], blob["S"], blob["Vh"], tuple(blob["indices"]), fp)
    except Exception as exc:  # a corrupt cache entry is a miss, not an error
        warnings.warn(f"ignoring unreadable basis cache entry {path}: {exc}", stacklevel=2)
        return None


def _cache_store(cache_dir: Optional[str], b: Bases) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, b.fingerprint)
    tmp = path + f".tmp{os.getpid()}"
    torch.save(
        {"U": b.U.cpu(), "S": b.S.cpu(), "Vh": b.Vh.cpu(), "indices": list(b.indices), "fingerprint": b.fingerprint},
        tmp,
    )
    os.replace(tmp, path)
