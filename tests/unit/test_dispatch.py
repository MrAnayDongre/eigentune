import pytest
import torch

from eigentune.kernels import available_backends, get_backend, register_backend, select_backend
from eigentune.kernels import policy


def test_torch_is_always_available():
    assert "torch" in available_backends()


def test_unknown_backend_is_an_error():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nope")


def test_cpu_auto_is_torch():
    x = torch.randn(4, 16)
    assert select_backend("auto", x, 4, "diag").name == "torch"
    assert select_backend("auto", x, 4, "diag", "bwd").name == "torch"


@pytest.mark.parametrize("name", ["triton", "native"])
def test_named_backend_that_cannot_run_falls_back_to_torch(name):
    # CPU tensors: no accelerator can run this, so the call must still succeed through the reference
    assert select_backend(name, torch.randn(4, 16), 4, "diag").name == "torch"


def test_third_party_backend_can_register_and_is_selectable():
    class Fake:
        name = "fake"
        def available(self): return True  # noqa: E704
        def supports(self, x, rank, kind): return True  # noqa: E704
        def forward(self, *a): return get_backend("torch").forward(*a)  # noqa: E704
        def backward(self, *a): return get_backend("torch").backward(*a)  # noqa: E704

    register_backend(Fake())
    assert select_backend("fake", torch.randn(2, 4), 2, "diag").name == "fake"


class _Dev:
    """Stand-in tensor for policy decisions, so thresholds are testable without a GPU."""
    def __init__(self, n, dtype=torch.bfloat16, cuda=True):
        self.shape, self.dtype, self.is_cuda = (n, 4096), dtype, cuda


@pytest.mark.parametrize("n,rank,kind,phase,expected", [
    (1, 16, "diag", "fwd", "native"), (16, 8, "diag", "fwd", "native"), (16, 16, "diag", "fwd", "torch"),
    (512, 16, "diag", "fwd", "torch"), (2048, 16, "diag", "fwd", "triton"), (4096, 64, "core", "fwd", "triton"),
    (64, 16, "diag", "bwd", "native"), (64, 16, "core", "bwd", "torch"), (128, 16, "diag", "bwd", "torch"),
    (2048, 16, "diag", "bwd", "triton"), (2048, 8, "diag", "bwd", "torch"), (2048, 8, "core", "bwd", "triton"),
])
def test_policy_thresholds(n, rank, kind, phase, expected, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert policy.preference(_Dev(n), rank, kind, phase)[0] == expected


def test_policy_never_picks_unmeasured_regimes(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert policy.preference(_Dev(1, torch.float32), 8, "diag") == ["torch"]       # fp32 not benchmarked
    assert policy.preference(_Dev(1, cuda=False), 8, "diag") == ["torch"]
    monkeypatch.setattr(torch.version, "hip", "6.2")
    assert policy.preference(_Dev(1), 8, "diag") == ["torch"]                       # ROCm not validated
