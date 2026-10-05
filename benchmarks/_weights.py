"""Real model weights for benchmarks, fetched one tensor at a time (HTTP range requests), cached on disk."""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import torch

CACHE = Path.home() / ".cache" / "eigentune-bench"
_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def remote_tensor(repo: str, name: str, shard: str | None = None) -> torch.Tensor:
    """Fetch tensor ``name`` from ``repo`` without downloading the whole checkpoint."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / (repo.replace("/", "--") + "--" + name + ".pt")
    if cached.exists():
        return torch.load(cached, weights_only=True)
    base = f"https://huggingface.co/{repo}/resolve/main/"
    if shard is None:
        index = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json", timeout=60).read())
        shard = index["weight_map"][name]
    url = base + shard
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    header = json.loads(_get(url, 8, 8 + n - 1))[name]
    a, b = header["data_offsets"]
    raw = bytearray(_get(url, 8 + n + a, 8 + n + b - 1))
    t = torch.frombuffer(raw, dtype=_DTYPES[header["dtype"]]).reshape(header["shape"]).clone()
    torch.save(t, cached)
    return t
