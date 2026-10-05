"""Saving and loading adapters: safetensors tensors plus a versioned JSON header.

By default an adapter holds only what training changed (``delta`` or ``core`` per layer). The frozen bases are
rebuilt from the base model on load, checked against a per-layer fingerprint of the base weight and a small
signature of the basis. ``save_bases=True`` embeds the exact bases instead (larger file, no SVD, exactly
reproducible across devices and library versions).
"""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from . import __version__
from .config import FORMAT_VERSION, EigenTuneConfig
from .model import get_eigentune_model, iter_eigentune_layers, source_fingerprints
from .svd import Bases, basis_signature, select_indices

CONFIG_NAME = "eigentune_config.json"
FP_LEN = 24  # hex chars of the base-weight fingerprint kept in the header (96 bits: a mismatch check, not security)
WEIGHTS_NAME = "adapter_model.safetensors"


class AdapterMismatchError(ValueError):
    """The adapter was trained against different base weights (or settings) than the model it is loaded into."""


def save_adapter(model: nn.Module, path: str, save_bases: bool = False) -> None:
    layers = dict(iter_eigentune_layers(model))
    if not layers:
        raise ValueError("model has no EigenTune layers")
    os.makedirs(path, exist_ok=True)
    tensors, meta = {}, {}
    cfg_rank_budget = model.eigentune_config.rank_budget is not None
    for name, layer in layers.items():
        tensors[f"{name}.{'delta' if layer.kind == 'diag' else 'core'}"] = layer.trainable().detach().cpu().contiguous()
        if save_bases:
            tensors[f"{name}.U"] = layer.U.detach().cpu().contiguous()
            tensors[f"{name}.Vh"] = layer.Vh.detach().cpu().contiguous()
            tensors[f"{name}.S"] = layer.S.detach().cpu().contiguous()
        sig = basis_signature(Bases(layer.U, layer.S, layer.Vh, layer.indices, layer.fingerprint))
        # compact on purpose: indices, shape and dtype are derivable, and the header must not dwarf the tensors
        meta[name] = {"fp": layer.fingerprint[:FP_LEN], "sig": [round(v, 5) for v in sig.tolist()]}
        if cfg_rank_budget:
            meta[name]["r"] = layer.rank
    cfg: EigenTuneConfig = model.eigentune_config
    header = {
        "format_version": FORMAT_VERSION,
        "eigentune_version": __version__,
        "config": cfg.to_dict(),
        "bases_embedded": save_bases,
        "layers": meta,
        "base_model": getattr(getattr(model, "config", None), "_name_or_path", None),
    }
    save_file(tensors, os.path.join(path, WEIGHTS_NAME))
    with open(os.path.join(path, CONFIG_NAME), "w") as f:
        json.dump(header, f, separators=(",", ":"))


def _read_header(path: str) -> dict:
    try:
        with open(os.path.join(path, CONFIG_NAME)) as f:
            header = json.load(f)
        version = int(header["format_version"])
        header["config"], header["layers"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not a readable EigenTune adapter: {exc}") from exc
    if version > FORMAT_VERSION:
        raise ValueError(f"adapter format {version} is newer than this EigenTune ({FORMAT_VERSION}); upgrade eigentune")
    return header


def load_adapter(
    model: nn.Module, path: str, weights: Optional[Mapping[str, torch.Tensor]] = None, signature_tol: float = 1e-3
) -> nn.Module:
    """Inject adapters into ``model`` and load their trained values.

    Raises :class:`AdapterMismatchError` if the base weights differ from the ones the adapter was trained on.
    """
    header = _read_header(path)
    cfg = EigenTuneConfig.from_dict(header["config"])
    meta = header["layers"]
    expected = source_fingerprints(model, cfg, weights)
    if set(expected) != set(meta):
        raise AdapterMismatchError(f"adapter layers {sorted(set(meta) ^ set(expected))[:5]} differ from the model's")
    wrong = [n for n in meta if expected[n][:FP_LEN] != meta[n]["fp"]]
    if wrong:
        raise AdapterMismatchError(
            f"{len(wrong)} layer(s), e.g. {wrong[0]}, have different base weights or SVD settings than the adapter "
            "was trained with"
        )
    tensors = load_file(os.path.join(path, WEIGHTS_NAME))
    embedded = None
    if header.get("bases_embedded"):
        embedded = {
            n: Bases(
                tensors[f"{n}.U"],
                tensors[f"{n}.S"],
                tensors[f"{n}.Vh"],
                select_indices(
                    min(tensors[f"{n}.U"].shape[0], tensors[f"{n}.Vh"].shape[1]),
                    tensors[f"{n}.U"].shape[1],
                    cfg.selection,
                ),
                expected[n],
            )
            for n in meta
        }
    ranks = {n: m["r"] for n, m in meta.items()} if cfg.rank_budget is not None else None
    get_eigentune_model(model, cfg, weights, bases=embedded, ranks=ranks)
    for name, layer in iter_eigentune_layers(model):
        sig = torch.tensor(meta[name]["sig"])
        now = basis_signature(Bases(layer.U, layer.S, layer.Vh, layer.indices, layer.fingerprint))
        if embedded is None and float((sig - now).abs().max()) > signature_tol * max(1.0, float(sig.abs().max())):
            raise AdapterMismatchError(
                f"{name}: the recomputed basis differs from the one used in training (different device, library or "
                "repeated singular values). Re-save the adapter with save_bases=True."
            )
        key = f"{name}.{'delta' if layer.kind == 'diag' else 'core'}"
        value = tensors[key]
        if value.shape != layer.trainable().shape:
            raise AdapterMismatchError(f"{key}: shape {tuple(value.shape)} != {tuple(layer.trainable().shape)}")
        with torch.no_grad():
            layer.trainable().copy_(value)
    return model
