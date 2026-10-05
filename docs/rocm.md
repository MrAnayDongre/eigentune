# ROCm / AMD

**Status: experimental. Nothing here has been run on an AMD GPU.** The development machine has an NVIDIA GPU and no ROCm
toolchain, so the AMD path is written to be portable and checked statically, and is not validated.

| component | implemented | compile-tested | hardware-tested | performance-tested |
|---|---|---|---|---|
| `torch` reference backend (CPU / any PyTorch device incl. ROCm builds) | yes | n/a | no (CPU and CUDA only) | no |
| Triton kernels (`triton_backend.py`) | yes, written to the Triton language, which also targets ROCm | no | no | no |
| Native kernels (`csrc/eigentune_ops.cu`) built as HIP | yes, via PyTorch's hipify | **no** (no `hipcc` available) | no | no |
| `backend="auto"` on ROCm | resolves to `torch` | n/a | no | n/a |

What *is* checked, without hardware (`tests/rocm/test_static.py`, runs on every CI job):

* the native source translates with PyTorch's own `hipify` (`#include "hip/hip_runtime.h"`, kernel launches become
  `hipLaunchKernelGGL`, no `cudaStream_t` left);
* there is exactly one NVIDIA-only shuffle (`__shfl_xor_sync`) and it lives behind the `ET_SHFL_XOR` macro, which maps to
  `__shfl_xor` on HIP;
* the warp-level kernel does no lane arithmetic against a literal 32: it reads `warpSize` on the device and
  `at::cuda::warp_size()` on the host, so AMD wavefronts of 64 are accounted for.

These checks show the code has no *known* NVIDIA assumptions. They do not show that it compiles, that the results are right, or that it is fast.

## Using it

On a ROCm build of PyTorch, `eigentune` imports and runs on the `torch` backend, which is plain PyTorch. To try the
accelerated paths explicitly:

```python
EigenTuneConfig(rank=16, backend="triton")   # Triton on ROCm: untested here
EigenTuneConfig(rank=16, backend="native")   # builds the HIP extension on first use: untested here
```

If either cannot run, the call falls back to `torch` instead of failing. `auto` will not pick them on ROCm until someone
measures them: `policy.preference` returns `["torch"]` when `torch.version.hip` is set.

## How to validate it (contributors with an AMD GPU)

```bash
pip install -e ".[dev,triton]"
pytest tests/kernels                 # compares every registered backend to the reference and an fp64 oracle
python benchmarks/kernels.py --out kernels_rocm
```

If those pass, open a PR that adds the results and, only then, extends `kernels/policy.py` for ROCm.
Things to watch: wavefront size 64 on CDNA vs 32 on RDNA, the 16-byte vector loads in `load_vec`, and Triton's `tl.dot`
input-precision options on ROCm.
