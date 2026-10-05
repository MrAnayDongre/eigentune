"""EigenTune end to end on a toy model, on CPU, with no downloads.

python examples/quickstart_cpu.py
"""

import tempfile

import torch
import torch.nn as nn

from eigentune import (
    EigenTuneConfig,
    adapter_report,
    get_eigentune_model,
    load_adapter,
    merge_adapter,
    save_adapter,
)


def toy():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))


# 1. adapt: freezes the model and wraps its linear layers
model = get_eigentune_model(toy(), EigenTuneConfig(rank=8, method="diagonal"))
rep = adapter_report(model)
print(
    f"trainable parameters: {rep['trainable_parameters']}   frozen bases at runtime: {rep['runtime_basis_bytes']} bytes"
)

# 2. train as usual: only the per-direction scales move
x, y = torch.randn(256, 32), torch.randn(256, 8)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
for _ in range(100):
    loss = nn.functional.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"loss after training: {loss.item():.4f}")

# 3. save only what training changed, reload into a fresh copy of the base model
with tempfile.TemporaryDirectory() as d:
    save_adapter(model, d)
    restored = load_adapter(toy(), d)
    assert torch.equal(model(x), restored(x))
    print("save/load round trip: identical outputs")

# 4. fold the adapter into the base weights for inference without any adapter overhead
before = model(x)
merge_adapter(model)
print("merged; max difference from the unmerged model:", (model(x) - before).abs().max().item())
