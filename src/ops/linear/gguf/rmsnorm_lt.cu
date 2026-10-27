#include "ops/linear/gguf/rmsnorm_lt.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/lt_fp4.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

#if defined(NINFER_SM120_FP8)
constexpr int kHiddenBlock = 256;
constexpr int kHeadThreads = 512;
constexpr int kHeadWarps   = kHeadThreads / kWarpSize;

// rmsnorm's store that also stages the block's column, pairs first .. of the output, in shared
// memory for the encode.
struct StagedPairStore {
    __nv_bfloat162* out;
    __nv_bfloat162* staged;
    std::int64_t first;
    __device__ __forceinline__ void operator()(std::int64_t index, __nv_bfloat162 value) const {
        out[index]            = value;
        staged[index - first] = value;
    }
};

// The NVFP4 activation of the block's column, staged in shared memory as the k values of the BF16
// output [columns][k]: lt_fp4_encode_kernel's encode of each 16-value group under the gather (a
// thread a group), and the block's max |value| into the slot.
template <int Block>
__device__ __forceinline__ void encode_staged(const __nv_bfloat162* staged, int column, int k,
                                              const std::int32_t* input_columns,
                                              const gguf::LtFp4Parts& parts) {
    __syncthreads();
    const auto* values = reinterpret_cast<const __nv_bfloat16*>(staged);
    float block_amax   = 0.0f;
#pragma unroll 1
    for (int group = static_cast<int>(threadIdx.x); group < k / 16; group += Block) {
        float v[16];
        gguf::detail::lt_fp4_load16(values, input_columns, k, 0, group, v);
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) { amax = fmaxf(amax, fabsf(v[j])); }
        gguf::detail::lt_fp4_encode_group(v, amax, parts.codes, parts.scales, parts.shifts, k,
                                          column, group);
        block_amax = fmaxf(block_amax, amax);
    }
    gguf::detail::lt_fp4_block_max<Block>(block_amax, parts.slot);
}

// A row a block.
__launch_bounds__(kHiddenBlock) __global__
    void rmsnorm_lt_fp4_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                               __nv_bfloat162* out, std::int64_t rows, float eps,
                               gguf::LtFp4Parts parts) {
    constexpr int kPairs = kRmsLtHidden / 2;
    __shared__ __align__(16) __nv_bfloat162 staged[kPairs];
    const std::int64_t first = std::int64_t(blockIdx.x) * kPairs;
    rmsnorm_cta_bf16x2_row<RmsEpilogue::Offset, kHiddenBlock, 10, true, kRmsLtHidden>(
        x, weight, nullptr, StagedPairStore{out, staged, first}, kRmsLtHidden, rows, eps);
    encode_staged<kHiddenBlock>(staged, static_cast<int>(blockIdx.x), kRmsLtHidden, nullptr,
                                parts);
}

// A column a block: its k / 128 head rows, a warp a row, loaded kHeadRowsInFlight rows at a time
// before any is normalized (the route's arithmetic; the loads only move ahead), then the column's
// encode.
constexpr int kHeadRowsInFlight = 3;

__launch_bounds__(kHeadThreads) __global__
    void gated_rmsnorm_lt_fp4_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out, float eps, int k,
                                     const std::int32_t* input_columns, gguf::LtFp4Parts parts) {
    extern __shared__ __align__(16) __nv_bfloat162 staged_column[];
    const int heads          = k / kRmsLtHead;
    const std::int64_t first = std::int64_t(blockIdx.x) * (k / 2);
    const StagedPairStore store{out, staged_column, first};
    const std::int64_t row0  = std::int64_t(blockIdx.x) * heads;
    for (int base = static_cast<int>(threadIdx.x) / kWarpSize; base < heads;
         base += kHeadWarps * kHeadRowsInFlight) {
        RmsWarpRow rows[kHeadRowsInFlight];
#pragma unroll
        for (int i = 0; i < kHeadRowsInFlight; ++i) {
            if (base + i * kHeadWarps < heads) {
                rmsnorm_warp_bf16x2_load<RmsEpilogue::Gated, true, kRmsLtHead>(
                    x, weight, z, kRmsLtHead, row0 + base + i * kHeadWarps, rows[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < kHeadRowsInFlight; ++i) {
            if (base + i * kHeadWarps < heads) {
                rmsnorm_warp_bf16x2_store<RmsEpilogue::Gated, true, kRmsLtHead>(
                    weight, z, store, kRmsLtHead, row0 + base + i * kHeadWarps, eps, rows[i]);
            }
        }
    }
    encode_staged<kHeadThreads>(staged_column, static_cast<int>(blockIdx.x), k, input_columns,
                                parts);
}
#endif

} // namespace

void rmsnorm_lt_fp4(const __nv_bfloat16* x, const __nv_bfloat16* weight, float eps, int columns,
                    __nv_bfloat16* out, void* activation, cudaStream_t stream) {
#if defined(NINFER_SM120_FP8)
    const gguf::LtFp4Parts parts =
        gguf::begin_lt_fp4_activation(activation, kRmsLtHidden, columns, stream);
    rmsnorm_lt_fp4_kernel<<<static_cast<unsigned>(columns), kHiddenBlock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(x), reinterpret_cast<const __nv_bfloat162*>(weight),
        reinterpret_cast<__nv_bfloat162*>(out), columns, eps, parts);
    CUDA_CHECK(cudaGetLastError());
    gguf::finish_lt_fp4_activation(out, nullptr, kRmsLtHidden, columns, parts, stream);
#else
    (void)x, (void)weight, (void)eps, (void)columns, (void)out, (void)activation, (void)stream;
    throw std::invalid_argument("rmsnorm NVFP4 activation: this build has no NVFP4 form");
#endif
}

void gated_rmsnorm_lt_fp4(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                          const __nv_bfloat16* z, float eps, int columns, int k,
                          const std::int32_t* input_columns, __nv_bfloat16* out, void* activation,
                          cudaStream_t stream) {
    if (columns <= 0 || k <= 0 || k % 256 != 0 || k > kRmsLtMaxHeads * kRmsLtHead) {
        throw std::invalid_argument("gated rmsnorm NVFP4 activation: invalid heads or columns");
    }
#if defined(NINFER_SM120_FP8)
    const gguf::LtFp4Parts parts = gguf::begin_lt_fp4_activation(activation, k, columns, stream);
    gated_rmsnorm_lt_fp4_kernel<<<static_cast<unsigned>(columns), kHeadThreads,
                                  std::size_t(k) * sizeof(__nv_bfloat16), stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(x), reinterpret_cast<const __nv_bfloat162*>(weight),
        reinterpret_cast<const __nv_bfloat162*>(z), reinterpret_cast<__nv_bfloat162*>(out), eps, k,
        input_columns, parts);
    CUDA_CHECK(cudaGetLastError());
    gguf::finish_lt_fp4_activation(out, input_columns, k, columns, parts, stream);
#else
    (void)x, (void)weight, (void)z, (void)eps, (void)input_columns, (void)out, (void)activation,
        (void)stream;
    throw std::invalid_argument("gated rmsnorm NVFP4 activation: this build has no NVFP4 form");
#endif
}

} // namespace ninfer::ops::detail
