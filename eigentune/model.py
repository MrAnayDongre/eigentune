"""Injecting EigenTune layers into a model, and reporting what that cost."""

from __future__ import annotations

import re
import warnings
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .config import EigenTuneConfig
from .layers import EigenTuneLinear
from .svd import Bases, compute_bases, fingerprint, resolve_backend


def _is_linear(module: nn.Module) -> bool:
    """``nn.Linear`` and lookalikes (e.g. bitsandbytes ``Linear4bit``): anything with in/out features and a weight."""
    return (
        hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
        and not isinstance(module, EigenTuneLinear)
    )


def _matches(name: str, cfg: EigenTuneConfig) -> bool:
    if any(name == e or name.endswith("." + e) for e in cfg.exclude_modules):
        return False
    t = cfg.target_modules
    if t is None:
        return True
    if isinstance(t, str):
        return re.fullmatch(t, name) is not None
    return any(name == m or name.endswith("." + m) for m in t)


def iter_eigentune_layers(model: nn.Module) -> Iterator[Tuple[str, EigenTuneLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, EigenTuneLinear):
            yield name, module


def get_eigentune_model(
    model: nn.Module,
    config: Optional[EigenTuneConfig] = None,
    weights: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    full_precision_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    bases: Optional[Mapping[str, Bases]] = None,
) -> nn.Module:
    """Freeze ``model`` and wrap its target linear layers with trainable EigenTune adapters, in place.

    Args:
        model: any ``nn.Module``; all of its parameters are frozen.
        config: what to adapt (default: rank 8 diagonal on every linear layer except ``lm_head``).
        weights: optional ``{"<module name>.weight": tensor}`` to take the SVD from, for layers whose own weight
            cannot be decomposed (quantized bases). Defaults to each layer's own weight.
        full_precision_state_dict: deprecated alias of ``weights`` (the 0.1 argument name).
        bases: precomputed bases per layer name (used when loading adapters that embed them).
    """
    cfg = config or EigenTuneConfig()
    if full_precision_state_dict is not None:
        warnings.warn("full_precision_state_dict is deprecated; pass weights=", DeprecationWarning, stacklevel=2)
        weights = weights or full_precision_state_dict
    model.requires_grad_(False)
    adapted = 0
    for name, module in list(model.named_modules()):
        if not name or not _is_linear(module) or not _matches(name, cfg):
            continue
        key = f"{name}.weight"
        w = weights[key] if weights is not None and key in weights else module.weight
        if w.dim() != 2 or not w.is_floating_point():
            raise ValueError(
                f"cannot decompose {name}: its weight is {tuple(w.shape)} {w.dtype} (quantized?). "
                f"Pass weights={{'{key}': full_precision_weight}}."
            )
        b = bases[name] if bases is not None and name in bases else compute_bases(w, cfg)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, EigenTuneLinear(module, b, cfg, name=name, dtype=w.dtype))
        adapted += 1
    if adapted == 0:
        raise ValueError("no linear layer matched target_modules; nothing was adapted")
    model.eigentune_config = cfg
    return model


def source_fingerprints(
    model: nn.Module, cfg: EigenTuneConfig, weights: Optional[Mapping[str, torch.Tensor]] = None
) -> Dict[str, str]:
    """Fingerprints the model's target layers *would* get (used to verify an adapter against a base model)."""
    out = {}
    for name, module in model.named_modules():
        if name and _is_linear(module) and _matches(name, cfg):
            w = weights[f"{name}.weight"] if weights is not None and f"{name}.weight" in weights else module.weight
            out[name] = fingerprint(w, cfg, resolve_backend(cfg.svd_backend, tuple(w.shape), cfg.rank, cfg.selection))
    return out


def merge_adapter(model: nn.Module) -> nn.Module:
    for _, layer in iter_eigentune_layers(model):
        layer.merge()
    return model


def unmerge_adapter(model: nn.Module) -> nn.Module:
    for _, layer in iter_eigentune_layers(model):
        layer.unmerge()
    return model


def adapter_report(model: nn.Module) -> Dict[str, int]:
    """What the adapter costs, kept apart on purpose.

    ``trainable_parameters``/``adapter_bytes`` are what training updates and what an adapter file holds by
    default; ``runtime_basis_bytes`` is the frozen bases that live next to the model while it runs.
    """
    trainable = adapter = basis = 0
    layers = 0
    for _, layer in iter_eigentune_layers(model):
        p = layer.trainable()
        trainable += p.numel()
        adapter += p.numel() * p.element_size()
        basis += sum(t.numel() * t.element_size() for t in (layer.U, layer.Vh, layer.S))
        layers += 1
    total = sum(p.numel() for p in model.parameters())
    return {
        "layers": layers,
        "trainable_parameters": trainable,
        "adapter_bytes": adapter,
        "runtime_basis_bytes": basis,
        "total_parameters": total,
    }


def print_trainable_parameters(model: nn.Module) -> None:
    r = adapter_report(model)
    print(
        f"trainable params: {r['trainable_parameters']:,} | all params: {r['total_parameters']:,} | "
        f"runtime bases: {r['runtime_basis_bytes'] / 2**20:.1f} MiB"
    )
