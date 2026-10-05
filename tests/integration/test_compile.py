import pytest
import torch
import torch.nn as nn

from eigentune import EigenTuneConfig, get_eigentune_model


@pytest.mark.parametrize("method", ["diagonal", "spectral_core"])
def test_torch_compile_fullgraph_matches_eager(method):
    torch.manual_seed(0)
    cfg = EigenTuneConfig(rank=8, method=method, backend="torch")
    m = get_eigentune_model(nn.Sequential(nn.Linear(64, 96), nn.ReLU(), nn.Linear(96, 32)), cfg)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(std=0.1)
    x = torch.randn(16, 64)
    ref = m(x)
    ref.sum().backward()
    ref_grads = [p.grad.clone() for p in m.parameters() if p.requires_grad]
    m.zero_grad()
    torch._dynamo.reset()
    out = torch.compile(m, backend="aot_eager", fullgraph=True)(x)
    out.sum().backward()
    assert torch.allclose(out, ref, atol=1e-5)
    for a, b in zip([p.grad for p in m.parameters() if p.requires_grad], ref_grads):
        assert torch.allclose(a, b, atol=1e-5)
