"""Every optimized backend against the PyTorch reference and an fp64 oracle."""

import pytest
import torch

from eigentune.kernels import available_backends, get_backend

DEVICE = "cuda"
TOL = {torch.float32: (2e-4, 2e-4), torch.float16: (4e-3, 4e-3), torch.bfloat16: (3e-2, 3e-2)}
# (tokens, in, out, rank): tails, odd sizes, decode-like, prefill-like
SHAPES = [(1, 64, 48, 4), (5, 130, 77, 8), (16, 256, 192, 16), (33, 1000, 520, 24), (130, 512, 384, 32),
          (64, 4096, 4096, 16)]


def backends():
    return [b for b in available_backends() if b != "torch"]


def oracle(x, Vh, U, w, kind, base_out):
    x, Vh, U, w, b = (t.double() for t in (x, Vh, U, w, base_out))
    Q = x @ Vh.T
    Z = Q * w if kind == "diag" else Q @ w.T
    return b + Z @ U.T


def make(N, inn, out, r, dtype, kind, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    mk = lambda *s: torch.randn(*s, device=DEVICE, generator=g).to(dtype)  # noqa: E731
    x, base_out = mk(N, inn), mk(N, out)
    Vh, U = mk(r, inn) / inn**0.5, mk(out, r) / r**0.5
    w = (mk(r) if kind == "diag" else mk(r, r) / r**0.5).float()
    return x, base_out, Vh, U, w.to(dtype)


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not backends(), reason="no accelerated backend registered")]


@pytest.mark.parametrize("name", backends())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["diag", "core"])
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_reference_and_oracle(name, dtype, kind, shape):
    N, inn, out, r = shape
    x, base_out, Vh, U, w = make(N, inn, out, r, dtype, kind)
    b, ref = get_backend(name), get_backend("torch")
    if not b.supports(x, r, kind):
        pytest.skip(f"{name} does not handle this shape")
    y, Q = b.forward(x, Vh, U, w, kind, base_out)
    y_ref, Q_ref = ref.forward(x, Vh, U, w, kind, base_out)
    rtol, atol = TOL[dtype]
    exact = oracle(x, Vh, U, w, kind, base_out)
    assert y.dtype == dtype and y.shape == (N, out) and Q.shape == (N, r)
    # the kernel must be at least as close to the exact answer as the reference is (plus rounding slack)
    err_k = (y.double() - exact).abs().max().item()
    err_r = (y_ref.double() - exact).abs().max().item()
    assert err_k <= max(2 * err_r, atol * 4)
    assert torch.allclose(y.float(), y_ref.float(), rtol=rtol, atol=atol * 4)
    assert torch.allclose(Q.float(), Q_ref.float(), rtol=rtol, atol=atol * 4)


@pytest.mark.parametrize("name", backends())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["diag", "core"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("need_x", [True, False])
def test_backward_matches_reference(name, dtype, kind, shape, need_x):
    N, inn, out, r = shape
    x, base_out, Vh, U, w = make(N, inn, out, r, dtype, kind)
    g = torch.randn(N, out, device=DEVICE).to(dtype)
    b, ref = get_backend(name), get_backend("torch")
    if not b.supports(x, r, kind):
        pytest.skip(f"{name} does not handle this shape")
    _, Q = ref.forward(x, Vh, U, w, kind, base_out)
    gx, gw = b.backward(g, Q, Vh, U, w, kind, need_x)
    gx_ref, gw_ref = ref.backward(g, Q, Vh, U, w, kind, need_x)
    rtol, atol = TOL[dtype]
    assert (gx is None) == (not need_x)
    if need_x:
        scale = gx_ref.float().abs().max().item()
        assert torch.allclose(gx.float(), gx_ref.float(), rtol=rtol, atol=atol * max(scale, 1) * 4)
    scale = gw_ref.float().abs().max().item()
    assert gw.shape == gw_ref.shape
    assert torch.allclose(gw.float(), gw_ref.float(), rtol=rtol * 2, atol=atol * max(scale, 1) * 4)


@pytest.mark.parametrize("name", backends())
def test_deterministic(name):
    N, inn, out, r = 100, 512, 384, 16
    x, base_out, Vh, U, w = make(N, inn, out, r, torch.bfloat16, "diag")
    g = torch.randn(N, out, device=DEVICE).bfloat16()
    b = get_backend(name)
    y1, Q = b.forward(x, Vh, U, w, "diag", base_out)
    y2, _ = b.forward(x, Vh, U, w, "diag", base_out)
    assert torch.equal(y1, y2)
    a, c = b.backward(g, Q, Vh, U, w, "diag", True), b.backward(g, Q, Vh, U, w, "diag", True)
    assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])


@pytest.mark.parametrize("name", backends())
def test_noncontiguous_inputs(name):
    N, inn, out, r = 20, 256, 192, 8
    x, base_out, Vh, U, w = make(N, inn, out, r, torch.float16, "diag")
    xt = torch.randn(inn, N, device=DEVICE).half().T   # strided view
    b, ref = get_backend(name), get_backend("torch")
    y, _ = b.forward(xt, Vh, U, w, "diag", base_out)
    y_ref, _ = ref.forward(xt, Vh, U, w, "diag", base_out)
    assert torch.allclose(y.float(), y_ref.float(), rtol=4e-3, atol=2e-2)


@pytest.mark.parametrize("name", backends())
def test_end_to_end_autograd_matches_reference(name):
    from eigentune.kernels import eigentune_update
    N, inn, out, r = 40, 300, 200, 8
    x, base_out, Vh, U, w = make(N, inn, out, r, torch.float32, "diag")
    g = torch.randn(N, out, device=DEVICE)
    grads = {}
    for backend in (name, "torch"):
        xx, bb, ww = x.clone().requires_grad_(), base_out.clone().requires_grad_(), w.clone().requires_grad_()
        eigentune_update(xx, bb, Vh, U, ww, "diag", backend).backward(g)
        grads[backend] = (xx.grad, bb.grad, ww.grad)
    for a, c in zip(grads[name], grads["torch"]):
        assert torch.allclose(a, c, rtol=1e-3, atol=1e-3)
