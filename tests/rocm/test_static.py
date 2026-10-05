"""Static ROCm/HIP checks that need no AMD hardware.

These prove the native source translates with PyTorch's own hipify and does not bake in NVIDIA assumptions. They do
NOT prove it compiles or runs on AMD: that needs a ROCm toolchain and a GPU (see docs/rocm.md).
"""

import re
import shutil
from pathlib import Path

import pytest

import eigentune

SRC = Path(eigentune.__file__).resolve().parent / "csrc" / "eigentune_ops.cu"  # the installed (or source) package


def test_no_hardcoded_nvidia_warp_assumptions():
    text = SRC.read_text()
    code = re.sub(r"//.*", "", text)
    assert code.count("__shfl_xor_sync") == 1, "NVIDIA-only shuffles must live in the ET_SHFL_XOR macro only"
    assert "warpSize" in code and "at::cuda::warp_size()" in code
    # no lane arithmetic against a literal 32 in the warp-level kernel
    kernel = code[code.index("down_partial_kernel") : code.index("up_fused_kernel")]
    assert "% 32" not in kernel and "/ 32" not in kernel


def test_source_hipifies_cleanly(tmp_path):
    hipify = pytest.importorskip("torch.utils.hipify.hipify_python")
    shutil.copy(SRC, tmp_path / SRC.name)
    res = hipify.hipify(
        project_directory=str(tmp_path),
        output_directory=str(tmp_path),
        header_include_dirs=[],
        includes=[str(tmp_path / "*")],
        extra_files=[str(tmp_path / SRC.name)],
        show_detailed=False,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    assert all(r.status == "[ok]" for r in res.values()), res
    out = (tmp_path / SRC.with_suffix(".hip").name).read_text()
    assert '#include "hip/hip_runtime.h"' in out
    assert "<<<" not in out, "kernel launches must be translated to hipLaunchKernelGGL"
    assert "cudaStream_t" not in out
