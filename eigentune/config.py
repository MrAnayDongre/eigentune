"""Configuration for an EigenTune adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

FORMAT_VERSION = 1

METHODS = ("diagonal", "spectral_core")
SELECTIONS = ("principal", "minor", "mixed")
UPDATES = ("additive", "relative")
SVD_BACKENDS = ("auto", "exact", "randomized", "lowrank")
BACKENDS = ("auto", "torch", "triton", "native")


@dataclass
class EigenTuneConfig:
    """What to adapt and how.

    Classic EigenTune (``method="diagonal"``) trains one scalar per selected singular direction:
    ``W' = W + U_r diag(delta) V_r^T``.

    ``method="spectral_core"`` trains an ``r x r`` matrix ``C`` between the same frozen bases:
    ``W' = W + U_r C V_r^T``. ``core_bandwidth`` restricts ``C`` to a band around the diagonal
    (``0`` is classic EigenTune, ``None`` is a dense core).

    Both start at zero, so the adapted model is identical to the base model at step 0.
    """

    rank: int = 8
    target_modules: Optional[Union[List[str], str]] = None
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    method: str = "diagonal"
    core_bandwidth: Optional[int] = None
    selection: str = "principal"
    update: str = "additive"
    svd_backend: str = "auto"
    svd_oversampling: int = 32
    svd_niter: int = 100  # cap on subspace iterations; the randomized solver stops earlier once converged
    svd_tol: float = 1e-4  # stop when the rank-r subspace moves less than this between checks
    svd_seed: int = 0
    backend: str = "auto"
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        for name, allowed in (
            ("method", METHODS),
            ("selection", SELECTIONS),
            ("update", UPDATES),
            ("svd_backend", SVD_BACKENDS),
            ("backend", BACKENDS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r} is not one of {allowed}")
        if self.core_bandwidth is not None and self.core_bandwidth < 0:
            raise ValueError("core_bandwidth must be >= 0 or None")
        if self.method == "diagonal" and self.core_bandwidth not in (None, 0):
            raise ValueError("core_bandwidth only applies to method='spectral_core'")

    @property
    def kind(self) -> str:
        """The compute shape of the update: a vector (``diag``) or a matrix (``core``)."""
        return "diag" if self.method == "diagonal" or self.core_bandwidth == 0 else "core"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EigenTuneConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
