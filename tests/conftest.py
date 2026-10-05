import pytest
import torch

torch.set_default_dtype(torch.float32)


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs an NVIDIA GPU")
    config.addinivalue_line("markers", "rocm: needs an AMD GPU (ROCm build of PyTorch)")


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available() and torch.version.cuda is not None
    has_rocm = torch.cuda.is_available() and torch.version.hip is not None
    for item in items:
        if item.get_closest_marker("cuda") and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if item.get_closest_marker("rocm") and not has_rocm:
            item.add_marker(pytest.mark.skip(reason="no ROCm device"))
