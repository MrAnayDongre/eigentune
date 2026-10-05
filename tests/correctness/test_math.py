import pytest
import torch
import torch.nn as nn

from eigentune import EigenTuneConfig, merge_adapter, unmerge_adapter
from eigentune.kernels import eigentune_update
from eigentune.layers import EigenTuneLinear
from eigentune.svd import compute_bases

SHAPES = [(8, 8), (24, 16), (16, 40), (65, 33)]  # rectangular, tiny, odd


def make(out, inn, cfg, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    base = nn.Linear(inn, out, bias=True).to(dtype)
    w = base.weight.detach().clone()
    layer = EigenTuneLinear(base, compute_bases(w, cfg), cfg, dtype=dtype)
    return base, layer, w


def explicit_delta(layer):
    return layer.delta_weight()


@pytest.mark.parametrize("out,inn", SHAPES)
@pytest.mark.parametrize("method,bw", [("diagonal", None), ("spectral_core", None), ("spectral_core", 1)])
def test_identity_at_zero(out, inn, method, bw):
    cfg = EigenTuneConfig(rank=4, method=method, core_bandwidth=bw)
    base, layer, _ = make(out, inn, cfg)
    x = torch.randn(3, 5, inn)
    assert torch.equal(layer(x), base(x))


@pytest.mark.parametrize("out,inn", SHAPES)
@pytest.mark.parametrize("selection", ["principal", "minor", "mixed"])
@pytest.mark.parametrize("update", ["additive", "relative"])
def test_diagonal_matches_explicit_weight(out, inn, selection, update):
    cfg = EigenTuneConfig(rank=3, selection=selection, update=update)
    base, layer, w = make(out, inn, cfg)
    with torch.no_grad():
        layer.delta.normal_()
    x = torch.randn(7, inn)
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    ix = torch.tensor(layer.indices)
    d = layer.delta.detach() * (S[ix] if update == "relative" else 1)
    W2 = w + (U[:, ix] * d) @ Vh[ix]
    assert torch.allclose(layer(x), x @ W2.T + base.bias, atol=1e-4)


@pytest.mark.parametrize("bw", [None, 0, 1, 2])
def test_core_matches_explicit_weight_and_band(bw):
    cfg = EigenTuneConfig(rank=5, method="spectral_core", core_bandwidth=bw)
    base, layer, w = make(24, 16, cfg)
    with torch.no_grad():
        layer.trainable().normal_()
    x = torch.randn(6, 16)
    C = layer.trainable().detach().clone()
    if bw is not None:
        i = torch.arange(5)
        C = C * ((i[:, None] - i[None]).abs() <= bw)
    # a dense core depends on the sign convention of the bases, so the oracle uses the layer's own (canonical) bases ...
    W2 = w + layer.U @ C @ layer.Vh
    assert torch.allclose(layer(x), x @ W2.T + base.bias, atol=1e-4)
    # ... and those must be the truncated SVD of the weight
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    assert torch.allclose((layer.U * layer.S) @ layer.Vh, (U[:, :5] * S[:5]) @ Vh[:5], atol=1e-4)


@pytest.mark.parametrize("kind", ["diag", "core"])
def test_gradients_match_plain_autograd(kind):
    torch.manual_seed(1)
    N, inn, out, r = 11, 20, 14, 4
    x = torch.randn(N, inn, requires_grad=True)
    base_out = torch.randn(N, out, requires_grad=True)
    Vh, U = torch.randn(r, inn), torch.randn(out, r)
    w = torch.randn(r) if kind == "diag" else torch.randn(r, r)
    w.requires_grad_(True)
    g = torch.randn(N, out)
    y = eigentune_update(x, base_out, Vh, U, w, kind)
    y.backward(g)
    got = [t.grad.clone() for t in (x, base_out, w)]
    for t in (x, base_out, w):
        t.grad = None
    Z = (x @ Vh.T) * w if kind == "diag" else (x @ Vh.T) @ w.T
    (base_out + Z @ U.T).backward(g)
    for a, b in zip(got, (x.grad, base_out.grad, w.grad)):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["diag", "core"])
def test_gradcheck_float64(kind):
    torch.manual_seed(2)
    N, inn, out, r = 5, 7, 6, 3
    x = torch.randn(N, inn, dtype=torch.float64, requires_grad=True)
    b = torch.randn(N, out, dtype=torch.float64, requires_grad=True)
    Vh, U = torch.randn(r, inn, dtype=torch.float64), torch.randn(out, r, dtype=torch.float64)
    w = (torch.randn(r) if kind == "diag" else torch.randn(r, r)).double().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x, b, w: eigentune_update(x, b, Vh, U, w, kind), (x, b, w))


def test_saves_q_not_x():
    """The autograd function keeps Q ([N, r]) and the bases, not the [N, in] activation."""
    N, inn, out, r = 64, 128, 96, 4
    x = torch.randn(N, inn)
    base_out = torch.zeros(N, out)
    Vh, U, w = torch.randn(r, inn), torch.randn(out, r), torch.zeros(r, requires_grad=True)
    y = eigentune_update(x, base_out, Vh, U, w, "diag")
    saved = {tuple(t.shape) for t in y.grad_fn.saved_tensors}
    assert (N, r) in saved and (N, inn) not in saved


@pytest.mark.parametrize("method", ["diagonal", "spectral_core"])
def test_merge_unmerge_roundtrip(method):
    cfg = EigenTuneConfig(rank=4, method=method)
    base, layer, w = make(24, 16, cfg)
    with torch.no_grad():
        layer.trainable().normal_()
    x = torch.randn(9, 16)
    expected = layer(x)
    model = nn.Sequential(layer)
    model.eigentune_config = cfg
    merge_adapter(model)
    assert torch.allclose(layer(x), expected, atol=1e-5)
    unmerge_adapter(model)
    assert torch.allclose(layer.base.weight, w, atol=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_bf16_forward_close_to_fp32():
    cfg = EigenTuneConfig(rank=4)
    base, layer, w = make(32, 32, cfg, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.delta.normal_()
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    ref_layer = make(32, 32, cfg, dtype=torch.float32)[1]
    ref_layer.base.load_state_dict({k: v.float() for k, v in layer.base.state_dict().items()})
    ref_layer.delta.data.copy_(layer.delta.data)
    assert torch.allclose(layer(x).float(), ref_layer(x.float()), atol=0.1, rtol=0.05)
