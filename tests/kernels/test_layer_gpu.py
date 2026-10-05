"""The whole layer on a GPU under backend="auto", at token counts that route to each backend."""

import pytest
import torch
import torch.nn as nn

from eigentune import EigenTuneConfig
from eigentune.kernels import policy, select_backend
from eigentune.layers import EigenTuneLinear
from eigentune.svd import compute_bases

pytestmark = pytest.mark.cuda


def build(backend, method="diagonal", dtype=torch.bfloat16, r=16, seed=0):
    torch.manual_seed(seed)
    base = nn.Linear(256, 192, bias=True).to("cuda", dtype)
    cfg = EigenTuneConfig(rank=r, method=method, backend=backend)
    layer = EigenTuneLinear(base, compute_bases(base.weight.detach(), cfg), cfg, dtype=dtype)
    with torch.no_grad():
        layer.trainable().copy_(torch.randn_like(layer.trainable(), generator=torch.Generator("cuda").manual_seed(1)))
    return layer


@pytest.mark.parametrize("method", ["diagonal", "spectral_core"])
@pytest.mark.parametrize("n,expected", [(1, "native"), (6, "native"), (300, "torch"), (4096, "triton")])
def test_auto_matches_torch_in_values_and_gradients(method, n, expected):
    r = 8 if n == 6 else 16
    x = torch.randn(n, 256, device="cuda", dtype=torch.bfloat16)
    chosen = select_backend("auto", x, r, "diag" if method == "diagonal" else "core").name
    assert chosen == expected
    ref_layer, auto_layer = build("torch", method, r=r), build("auto", method, r=r)
    g = torch.randn(n, 192, device="cuda", dtype=torch.bfloat16)
    outs, grads = [], []
    for layer in (ref_layer, auto_layer):
        xx = x.clone().requires_grad_()
        y = layer(xx)
        y.backward(g)
        outs.append(y)
        grads.append((xx.grad, layer.trainable().grad))
    scale = outs[0].float().abs().max().item()
    assert torch.allclose(outs[1].float(), outs[0].float(), atol=0.03 * scale, rtol=0.03)
    for a, b in zip(grads[1], grads[0]):
        assert torch.allclose(a.float(), b.float(), atol=0.05 * b.float().abs().max().item(), rtol=0.05)


def test_thresholds_in_policy_module_are_the_ones_tested_here():
    assert policy.TRITON_MIN_TOKENS == 2048 and policy.NATIVE_FWD_MAX_TOKENS_X_RANK == 128
