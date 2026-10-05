"""Injecting EigenTune layers into a model, and reporting what that cost."""

from __future__ import annotations

import re
import warnings
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .config import EigenTuneConfig
from .layers import EigenTuneLinear
from .svd import Bases, allocate_ranks, compute_bases, fingerprint, resolve_backend, truncate


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
    ranks: Optional[Mapping[str, int]] = None,
) -> nn.Module:
    """Freeze ``model`` and wrap its target linear layers with trainable EigenTune adapters, in place.

    Args:
        model: any ``nn.Module``; all of its parameters are frozen.
        config: what to adapt (default: rank 8 diagonal on every linear layer except ``lm_head``).
        weights: optional ``{"<module name>.weight": tensor}`` to take the SVD from, for layers whose own weight
            cannot be decomposed (quantized bases). Defaults to each layer's own weight.
        full_precision_state_dict: deprecated alias of ``weights`` (the 0.1 argument name).
        bases: precomputed bases per layer name (used when loading adapters that embed them).
        ranks: per-layer ranks (used when loading adapters trained with ``rank_budget``).
    """
    cfg = config or EigenTuneConfig()
    if full_precision_state_dict is not None:
        warnings.warn("full_precision_state_dict is deprecated; pass weights=", DeprecationWarning, stacklevel=2)
        weights = weights or full_precision_state_dict
    model.requires_grad_(False)
    targets = []
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
        targets.append((name, module, w))
    if not targets:
        raise ValueError("no linear layer matched target_modules; nothing was adapted")
    computed = {n: bases[n] if bases is not None and n in bases else compute_bases(w, cfg) for n, _, w in targets}
    if cfg.rank_budget is not None and ranks is None:
        ranks = allocate_ranks(
            {n: b.S for n, b in computed.items()},
            {n: float(w.detach().float().pow(2).sum()) for n, _, w in targets},
            cfg.rank_budget,
        )
    for name, module, w in targets:
        b = computed[name]
        if ranks is not None and ranks[name] < b.U.shape[1]:
            b = truncate(b, ranks[name])
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, EigenTuneLinear(module, b, cfg, name=name, dtype=w.dtype))
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


def _restore_base_layers(model: nn.Module) -> nn.Module:
    for name, layer in list(iter_eigentune_layers(model)):
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, layer.base)
    if hasattr(model, "eigentune_config"):
        del model.eigentune_config
    return model


def merge_and_unload(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight and restore the original layers, in place.

    The result is the plain model again (same module tree and ``state_dict`` keys as before adaptation) with the
    fine-tuning baked in, so ``save_pretrained`` / ``from_pretrained`` work as for any model. Needs plain
    floating-point ``nn.Linear`` bases; merging is exact in fp32 and rounds to the weight dtype otherwise.
    """
    merge_adapter(model)
    return _restore_base_layers(model)


def unload(model: nn.Module) -> nn.Module:
    """Discard the adapters and restore the original layers (the base weights are untouched unless already merged)."""
    return _restore_base_layers(unmerge_adapter(model))


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
