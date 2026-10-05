"""Build every compared PEFT method behind one function, so a benchmark changes only the method name."""

from __future__ import annotations

import os
import tempfile
import time

import torch

from eigentune import EigenTuneConfig, adapter_report, get_eigentune_model, save_adapter

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PEFT_METHODS = [
    "lora",
    "rslora",
    "dora",
    "pissa",
    "olora",
    "lora_plus",
    "lora_fa",
]  # lora_fa/olora: supported, not in the default comparison
EIGEN_METHODS = [
    "eigentune_diag",
    "eigentune_core",
    "eigentune_diag_minor",
    "eigentune_core_minor",
    "eigentune_diag_relative",
]
ALL = EIGEN_METHODS + PEFT_METHODS


def _dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


def build(method: str, model, rank: int, targets=TARGETS):
    """Returns ``(model, info, make_optimizer)``; ``info`` has trainable params, adapter bytes, init seconds."""
    t0 = time.perf_counter()
    info = {"method": method, "rank": rank}
    if method.startswith("eigentune"):
        kw = {}
        if "core" in method:
            kw.update(method="spectral_core")
        if "minor" in method:
            kw.update(selection="minor")
        if "relative" in method:
            kw.update(update="relative")
        cfg = EigenTuneConfig(rank=rank, target_modules=targets, **kw)
        model = get_eigentune_model(model, cfg)
        torch.cuda.synchronize()
        info["init_seconds"] = time.perf_counter() - t0
        rep = adapter_report(model)
        info.update(trainable_parameters=rep["trainable_parameters"], runtime_basis_bytes=rep["runtime_basis_bytes"])
        with tempfile.TemporaryDirectory() as d:
            save_adapter(model, d)
            info["adapter_bytes"] = _dir_bytes(d)
        with tempfile.TemporaryDirectory() as d:
            save_adapter(model, d, save_bases=True)
            info["adapter_bytes_with_bases"] = _dir_bytes(d)
        params = [p for p in model.parameters() if p.requires_grad]
        return model, info, lambda lr: torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    from peft import LoraConfig, get_peft_model

    kw = dict(r=rank, lora_alpha=2 * rank, target_modules=targets, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    if method == "rslora":
        kw["use_rslora"] = True
    if method == "dora":
        kw["use_dora"] = True
    if method == "pissa":
        kw["init_lora_weights"] = "pissa_niter_16"
    if method == "olora":
        kw["init_lora_weights"] = "olora"
    model = get_peft_model(model, LoraConfig(**kw))
    torch.cuda.synchronize()
    info["init_seconds"] = time.perf_counter() - t0
    info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["runtime_basis_bytes"] = 0
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        info["adapter_bytes"] = _dir_bytes(d)
    info["adapter_bytes_with_bases"] = info["adapter_bytes"]
    params = [p for p in model.parameters() if p.requires_grad]
    if method == "lora_plus":
        from peft.optimizers import create_loraplus_optimizer

        return model, info, lambda lr: create_loraplus_optimizer(model, torch.optim.AdamW, lr=lr, loraplus_lr_ratio=16)
    if method == "lora_fa":
        from peft.optimizers import create_lorafa_optimizer

        return model, info, lambda lr: create_lorafa_optimizer(model, r=rank, lora_alpha=2 * rank, lr=lr)
    return model, info, lambda lr: torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
