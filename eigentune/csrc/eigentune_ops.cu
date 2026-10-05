// EigenTune native kernels (CUDA, and HIP through PyTorch's hipify).
//
// Where a hand-written kernel can plausibly beat the vendor GEMM: very few tokens, tiny rank. There the work is two
// skinny products and a scale, and the cost is launches and memory latency, not FLOPs. So:
//
//   down_partial : Q = X Vh^T, split over K. One warp per (token, rank) pair writes a fp32 partial per K-split.
//   up_fused     : sums the partials, applies the scale (diagonal or core), multiplies by U^T and adds the base
//                  output, all in one pass; also writes Q for the backward.
//   bwd_reduce   : grad_w = sum_n Q*P and gQ = P*w in one pass (diagonal).
//
// Reductions use fixed orders (no atomics), so results are deterministic. Large GEMMs stay on the vendor library.
// Warp/wavefront width is read from the hardware (`warpSize`), never assumed to be 32.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#define ET_SHFL_XOR(v, m) __shfl_xor((v), (m))
#else
#define ET_SHFL_XOR(v, m) __shfl_xor_sync(0xffffffffu, (v), (m))
#endif

namespace {

constexpr int kBlock = 256;
constexpr int kMaxRank = 64;
constexpr int kMaxTokens = 256;
constexpr int kMaxElems = 2048;  // N * rank staged in shared memory (q and z: 8 KB each)

#define ET_DISPATCH(type, name, ...)                                                                   \
  AT_DISPATCH_SWITCH(type, name, AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__)                              \
                                     AT_DISPATCH_CASE(at::kHalf, __VA_ARGS__)                           \
                                         AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__))

template <typename T>
__device__ __forceinline__ void load_vec(const T* p, float* out) {
  constexpr int V = 16 / sizeof(T);
  T tmp[V];
  memcpy(tmp, p, sizeof(tmp));  // one 16-byte load
#pragma unroll
  for (int i = 0; i < V; ++i) out[i] = static_cast<float>(tmp[i]);
}

// ---------------------------------------------------------------------------------------------- down_partial
template <typename T>
__global__ void down_partial_kernel(const T* __restrict__ x, const T* __restrict__ vh, float* __restrict__ part,
                                    int N, int K, int R, int slice) {
  constexpr int V = 16 / sizeof(T);
  const int lane = threadIdx.x % warpSize;
  const int warps = blockDim.x / warpSize;
  const int pair = blockIdx.x * warps + threadIdx.x / warpSize;  // (n, j) pair
  if (pair >= N * R) return;
  const int n = pair / R, j = pair % R;
  const int k0 = blockIdx.y * slice;
  const int k1 = min(K, k0 + slice);
  const T* xr = x + (size_t)n * K;
  const T* vr = vh + (size_t)j * K;
  float acc = 0.f;
  for (int k = k0 + lane * V; k + V <= k1; k += warpSize * V) {
    float a[V], b[V];
    load_vec(xr + k, a);
    load_vec(vr + k, b);
#pragma unroll
    for (int i = 0; i < V; ++i) acc += a[i] * b[i];
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) acc += ET_SHFL_XOR(acc, off);
  if (lane == 0) part[((size_t)blockIdx.y * N + n) * R + j] = acc;
}

// ---------------------------------------------------------------------------------------------- up_fused
// out[n, o] = base[n, o] + sum_j z[n, j] * U[o, j],  z = scale(Q).   RP = rank padded to a supported size.
template <typename T, int RP>
__global__ void up_fused_kernel(const float* __restrict__ part, int S, const T* __restrict__ w, bool core,
                                const T* __restrict__ u, const T* __restrict__ base, T* __restrict__ out,
                                T* __restrict__ qout, int N, int M, int R) {
  __shared__ float q[kMaxElems];
  __shared__ float z[kMaxElems];
  __shared__ float wsh[kMaxRank * kMaxRank];
  const int nr = N * R;
  for (int i = threadIdx.x; i < nr; i += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < S; ++k) s += part[(size_t)k * nr + i];
    T qt = static_cast<T>(s);  // Q is stored (and scaled) in the compute dtype, like the reference
    q[i] = static_cast<float>(qt);
    if (blockIdx.x == 0) qout[i] = qt;
  }
  for (int i = threadIdx.x; i < (core ? R * R : R); i += blockDim.x) wsh[i] = static_cast<float>(w[i]);
  __syncthreads();
  for (int i = threadIdx.x; i < nr; i += blockDim.x) {
    const int n = i / R, j = i % R;
    float v;
    if (!core) {
      v = q[i] * wsh[j];
    } else {
      v = 0.f;
      for (int k = 0; k < R; ++k) v += wsh[j * R + k] * q[n * R + k];
    }
    z[i] = static_cast<float>(static_cast<T>(v));
  }
  __syncthreads();
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= M) return;
  float ur[RP];
#pragma unroll
  for (int j = 0; j < RP; ++j) ur[j] = j < R ? static_cast<float>(u[(size_t)o * R + j]) : 0.f;
  for (int n = 0; n < N; ++n) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RP; ++j) acc += ur[j] * (j < R ? z[n * R + j] : 0.f);
    out[(size_t)n * M + o] = static_cast<T>(static_cast<float>(base[(size_t)n * M + o]) + acc);
  }
}

// ---------------------------------------------------------------------------------------------- bwd_reduce
template <typename T>
__global__ void bwd_reduce_kernel(const T* __restrict__ q, const T* __restrict__ p, const T* __restrict__ w,
                                  float* __restrict__ gw, T* __restrict__ gq, int N, int R) {
  __shared__ float red[8][33];
  const int jl = threadIdx.x % 32, nl = threadIdx.x / 32;  // 32 ranks x 8 token lanes
  const int j = blockIdx.x * 32 + jl;
  float acc = 0.f;
  if (j < R) {
    const float wj = static_cast<float>(w[j]);
    for (int n = nl; n < N; n += 8) {
      const float pv = static_cast<float>(p[(size_t)n * R + j]);
      acc += static_cast<float>(q[(size_t)n * R + j]) * pv;
      gq[(size_t)n * R + j] = static_cast<T>(pv * wj);
    }
  }
  red[nl][jl] = acc;
  __syncthreads();
  if (nl == 0 && j < R) {
    float s = 0.f;
    for (int i = 0; i < 8; ++i) s += red[i][jl];
    gw[j] = s;
  }
}

template <typename T, int RP>
void launch_up(const at::Tensor& part, int S, const at::Tensor& w, bool core, const at::Tensor& u,
               const at::Tensor& base, at::Tensor& out, at::Tensor& qout, int N, int M, int R, cudaStream_t st) {
  up_fused_kernel<T, RP><<<(M + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      part.data_ptr<float>(), S, w.data_ptr<T>(), core, u.data_ptr<T>(), base.data_ptr<T>(), out.data_ptr<T>(),
      qout.data_ptr<T>(), N, M, R);
}

template <typename T>
void dispatch_up(const at::Tensor& part, int S, const at::Tensor& w, bool core, const at::Tensor& u,
                 const at::Tensor& base, at::Tensor& out, at::Tensor& qout, int N, int M, int R, cudaStream_t st) {
  if (R <= 8) launch_up<T, 8>(part, S, w, core, u, base, out, qout, N, M, R, st);
  else if (R <= 16) launch_up<T, 16>(part, S, w, core, u, base, out, qout, N, M, R, st);
  else if (R <= 32) launch_up<T, 32>(part, S, w, core, u, base, out, qout, N, M, R, st);
  else launch_up<T, 64>(part, S, w, core, u, base, out, qout, N, M, R, st);
}

}  // namespace

std::vector<at::Tensor> forward(const at::Tensor& x, const at::Tensor& vh, const at::Tensor& u, const at::Tensor& w,
                                bool core, const at::Tensor& base) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && vh.is_contiguous() && u.is_contiguous() && base.is_contiguous(),
              "eigentune native: inputs must be contiguous CUDA tensors");
  const int N = x.size(0), K = x.size(1), R = vh.size(0), M = u.size(0);
  TORCH_CHECK(N <= kMaxTokens && R <= kMaxRank && N * R <= kMaxElems,
              "eigentune native: rank <= 64 and tokens * rank <= 2048");
  const c10::cuda::CUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  auto out = at::empty({N, M}, x.options());
  auto q = at::empty({N, R}, x.options());
  ET_DISPATCH(x.scalar_type(), "eigentune_forward", [&] {
    constexpr int V = 16 / sizeof(scalar_t);
    TORCH_CHECK(K % V == 0, "eigentune native: in_features must be a multiple of ", V);
    const int wsz = at::cuda::warp_size();  // 32 on NVIDIA, 32 or 64 on AMD
    const int warps = kBlock / wsz;
    const int blocks_x = (N * R + warps - 1) / warps;
    // enough blocks to fill the GPU, but never a slice shorter than one warp-iteration
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int S = std::max(1, std::min((4 * sms + blocks_x - 1) / blocks_x, K / (wsz * V)));
    const int slice = ((K + S - 1) / S + V - 1) / V * V;
    S = (K + slice - 1) / slice;
    auto part = at::empty({S, N, R}, x.options().dtype(at::kFloat));
    down_partial_kernel<scalar_t><<<dim3(blocks_x, S), kBlock, 0, st>>>(
        x.data_ptr<scalar_t>(), vh.data_ptr<scalar_t>(), part.data_ptr<float>(), N, K, R, slice);
    dispatch_up<scalar_t>(part, S, w, core, u, base, out, q, N, M, R, st);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, q};
}

std::vector<at::Tensor> bwd_reduce(const at::Tensor& q, const at::Tensor& p, const at::Tensor& w) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && p.is_contiguous(), "eigentune native: contiguous CUDA tensors");
  const int N = q.size(0), R = q.size(1);
  const c10::cuda::CUDAGuard guard(q.device());
  auto st = at::cuda::getCurrentCUDAStream();
  auto gw = at::empty({R}, q.options().dtype(at::kFloat));
  auto gq = at::empty({N, R}, q.options());
  ET_DISPATCH(q.scalar_type(), "eigentune_bwd_reduce", [&] {
    bwd_reduce_kernel<scalar_t><<<(R + 31) / 32, 256, 0, st>>>(
        q.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), gw.data_ptr<float>(),
        gq.data_ptr<scalar_t>(), N, R);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gw, gq};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "EigenTune forward (down_partial + up_fused)");
  m.def("bwd_reduce", &bwd_reduce, "EigenTune diagonal backward reduction");
}
