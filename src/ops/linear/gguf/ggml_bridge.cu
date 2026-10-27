// Entry points into the vendored ggml CUDA kernels and the vector kernel (ggml_bridge_vec.cuh). The
// activation quantizers follow ggml's q8_1, reading BF16 and optionally gathering the input columns.

#include "ggml_bridge_internal.cuh"

#include "ggml-cuda/dequantize.cuh"

#include <cub/block/block_scan.cuh>
#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

// ggml's GGML_ABORT/GGML_ASSERT land here; only the vendored headers' own invariants reach it.
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml abort at %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

namespace ninfer::ops::gguf {
namespace detail {

const DeviceFacts& device_facts() {
    static std::array<DeviceFacts, 16> facts{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    if (device < 0 || device >= int(facts.size())) {
        throw std::invalid_argument("gguf: device ordinal out of range");
    }
    std::call_once(once[device], [&] {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        facts[device].cc                     = 100 * properties.major + 10 * properties.minor;
        facts[device].sm_count               = properties.multiProcessorCount;
        facts[device].shared_per_block_optin = properties.sharedMemPerBlockOptin;
    });
    return facts[device];
}

namespace {

// x BF16 [columns][k] -> the vector kernel's planar q8 activation: ggml's q8_1 numbers, one warp per
// 32 values, element i of column c read from x[c][input_columns[i]] when a gather is given.
constexpr int kQuantizeThreads = 256;

__launch_bounds__(kQuantizeThreads) __global__
    void quantize_vector_kernel(const __nv_bfloat16* __restrict__ x,
                                const std::int32_t* __restrict__ input_columns,
                                std::int8_t* __restrict__ qs, half2* __restrict__ ds, int k) {
    const int i      = blockDim.x * blockIdx.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    const int source = input_columns != nullptr ? input_columns[i] : i;
    const float xi   = __bfloat162float(x[std::int64_t(column) * k + source]);
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float sum  = warp_reduce_sum<QK8_1>(xi);
    const float d    = amax / 127.0f;
    qs[std::int64_t(column) * k + i] = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
    if (i % QK8_1 == 0) { ds[std::int64_t(column) * (k / QK8_1) + i / QK8_1] = make_half2(d, sum); }
}

// x BF16 [columns][k] -> the narrow kernel's activation (narrow_activation_bytes): the vector
// activation's values and d (its FP16 rounding, held as FP32) in the narrow kernel's order.
// Columns [columns, columns_pad) write zero scales only.
__launch_bounds__(kQuantizeThreads) __global__
    void quantize_narrow_kernel(const __nv_bfloat16* __restrict__ x,
                                const std::int32_t* __restrict__ input_columns,
                                std::int8_t* __restrict__ qs, float* __restrict__ ds, int k,
                                int columns, int columns_pad) {
    const int i      = blockDim.x * blockIdx.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    const int slice = i / QK8_1, v = i % QK8_1;
    float* d_at     = ds + (std::int64_t(slice / 2) * columns_pad + column) * 2 + slice % 2;
    if (column >= columns) {
        if (v == 0) { *d_at = 0.0f; }
        return;
    }
    const int source = input_columns != nullptr ? input_columns[i] : i;
    const float xi   = __bfloat162float(x[std::int64_t(column) * k + source]);
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float d    = amax / 127.0f;
    qs[std::int64_t(column) * k + 64 * (slice / 2) + 16 * (v / 8) + 8 * (slice % 2) + v % 8] =
        amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
    if (v == 0) { *d_at = __half2float(__float2half(d)); }
}

using VectorLaunch = void (*)(const VecArgs&, int, bool, cudaStream_t);
using MoeLaunch    = void (*)(const MoeVecArgs&, int, int, bool, cudaStream_t);

MoeLaunch moe_launcher(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0:
        return moe_launch<GGML_TYPE_Q8_0>;
    case GgmlType::Q4_0:
        return moe_launch<GGML_TYPE_Q4_0>;
    case GgmlType::Q5_0:
        return moe_launch<GGML_TYPE_Q5_0>;
    case GgmlType::Q2_0:
        return moe_launch<GGML_TYPE_Q2_0>;
    case GgmlType::Q2_K:
        return moe_launch<GGML_TYPE_Q2_K>;
    case GgmlType::Q3_K:
        return moe_launch<GGML_TYPE_Q3_K>;
    case GgmlType::Q4_K:
        return moe_launch<GGML_TYPE_Q4_K>;
    case GgmlType::Q5_K:
        return moe_launch<GGML_TYPE_Q5_K>;
    case GgmlType::Q6_K:
        return moe_launch<GGML_TYPE_Q6_K>;
    case GgmlType::IQ2_XXS:
        return moe_launch<GGML_TYPE_IQ2_XXS>;
    case GgmlType::IQ2_XS:
        return moe_launch<GGML_TYPE_IQ2_XS>;
    case GgmlType::IQ2_S:
        return moe_launch<GGML_TYPE_IQ2_S>;
    case GgmlType::IQ3_XXS:
        return moe_launch<GGML_TYPE_IQ3_XXS>;
    case GgmlType::IQ3_S:
        return moe_launch<GGML_TYPE_IQ3_S>;
    case GgmlType::IQ1_S:
        return moe_launch<GGML_TYPE_IQ1_S>;
    case GgmlType::IQ1_M:
        return moe_launch<GGML_TYPE_IQ1_M>;
    case GgmlType::IQ4_NL:
        return moe_launch<GGML_TYPE_IQ4_NL>;
    case GgmlType::IQ4_XS:
        return moe_launch<GGML_TYPE_IQ4_XS>;
    }
    throw std::invalid_argument("gguf moe product: unsupported block type");
}

template <class Args>
using DecodeLaunch = void (*)(const Args&, int, cudaStream_t);

template <class Fn, template <ggml_type> class Pick>
Fn decode_launcher(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return Pick<GGML_TYPE_Q8_0>::launch;
    case GgmlType::Q4_0: return Pick<GGML_TYPE_Q4_0>::launch;
    case GgmlType::Q5_0: return Pick<GGML_TYPE_Q5_0>::launch;
    case GgmlType::Q2_0: return Pick<GGML_TYPE_Q2_0>::launch;
    case GgmlType::Q2_K: return Pick<GGML_TYPE_Q2_K>::launch;
    case GgmlType::Q3_K: return Pick<GGML_TYPE_Q3_K>::launch;
    case GgmlType::Q4_K: return Pick<GGML_TYPE_Q4_K>::launch;
    case GgmlType::Q5_K: return Pick<GGML_TYPE_Q5_K>::launch;
    case GgmlType::Q6_K: return Pick<GGML_TYPE_Q6_K>::launch;
    case GgmlType::IQ2_XXS: return Pick<GGML_TYPE_IQ2_XXS>::launch;
    case GgmlType::IQ2_XS: return Pick<GGML_TYPE_IQ2_XS>::launch;
    case GgmlType::IQ2_S: return Pick<GGML_TYPE_IQ2_S>::launch;
    case GgmlType::IQ3_XXS: return Pick<GGML_TYPE_IQ3_XXS>::launch;
    case GgmlType::IQ3_S: return Pick<GGML_TYPE_IQ3_S>::launch;
    case GgmlType::IQ1_S: return Pick<GGML_TYPE_IQ1_S>::launch;
    case GgmlType::IQ1_M: return Pick<GGML_TYPE_IQ1_M>::launch;
    case GgmlType::IQ4_NL: return Pick<GGML_TYPE_IQ4_NL>::launch;
    case GgmlType::IQ4_XS: return Pick<GGML_TYPE_IQ4_XS>::launch;
    }
    throw std::invalid_argument("gguf moe decode: unsupported block type");
}

template <ggml_type type>
struct PickUp {
    static constexpr auto launch = &moe_decode_up_launch<type>;
};
template <ggml_type type>
struct PickDown {
    static constexpr auto launch = &moe_decode_down_launch<type>;
};
template <ggml_type type>
struct PickPrep {
    static constexpr auto launch = &moe_decode_prep_launch<type>;
};

// One block: the pairs of each expert counted, scanned into bounds, then placed. The experts with
// pairs are listed in ascending order.
constexpr int kSortThreads = 1024;

__launch_bounds__(kSortThreads) __global__
    void moe_sort_kernel(const std::int32_t* __restrict__ ids, int pairs, int experts,
                         std::int32_t* __restrict__ bounds, std::int32_t* __restrict__ sorted,
                         std::int32_t* __restrict__ active,
                         std::int32_t* __restrict__ active_count) {
    extern __shared__ int shared[];
    int* count  = shared;
    int* cursor = shared + experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { count[e] = 0; }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < experts) { atomicAdd(count + e, 1); }
    }
    __syncthreads();
    // Each thread owns a run of consecutive experts; two block scans give every run its first
    // pair and its first active slot.
    using Scan = cub::BlockScan<int, kSortThreads>;
    __shared__ typename Scan::TempStorage scan;
    const int per   = (experts + kSortThreads - 1) / kSortThreads;
    const int first = threadIdx.x * per;
    int owned = 0, used = 0;
    for (int e = first; e < min(first + per, experts); ++e) {
        owned += count[e];
        used += count[e] != 0;
    }
    int run = 0, listed = 0, listed_total = 0;
    Scan(scan).ExclusiveSum(owned, run);
    __syncthreads();
    Scan(scan).ExclusiveSum(used, listed, listed_total);
    for (int e = first; e < min(first + per, experts); ++e) {
        bounds[e] = run;
        cursor[e] = run;
        run += count[e];
        if (count[e] != 0) { active[listed++] = e; }
    }
    if (threadIdx.x == kSortThreads - 1) {
        bounds[experts] = run;
        *active_count   = listed_total;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < experts) { sorted[atomicAdd(cursor + e, 1)] = p; }
    }
}

VectorLaunch vector_launch(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return vec_launch<GGML_TYPE_Q8_0>;
    case GgmlType::Q4_0:
        return vec_launch<GGML_TYPE_Q4_0>;
    case GgmlType::Q5_0:
        return vec_launch<GGML_TYPE_Q5_0>;
    case GgmlType::Q2_0:
        return vec_launch<GGML_TYPE_Q2_0>;
    case GgmlType::Q2_K: return vec_launch<GGML_TYPE_Q2_K>;
    case GgmlType::Q3_K: return vec_launch<GGML_TYPE_Q3_K>;
    case GgmlType::Q4_K: return vec_launch<GGML_TYPE_Q4_K>;
    case GgmlType::Q5_K: return vec_launch<GGML_TYPE_Q5_K>;
    case GgmlType::Q6_K: return vec_launch<GGML_TYPE_Q6_K>;
    case GgmlType::IQ2_XXS: return vec_launch<GGML_TYPE_IQ2_XXS>;
    case GgmlType::IQ2_XS: return vec_launch<GGML_TYPE_IQ2_XS>;
    case GgmlType::IQ2_S: return vec_launch<GGML_TYPE_IQ2_S>;
    case GgmlType::IQ3_XXS: return vec_launch<GGML_TYPE_IQ3_XXS>;
    case GgmlType::IQ3_S: return vec_launch<GGML_TYPE_IQ3_S>;
    case GgmlType::IQ1_S: return vec_launch<GGML_TYPE_IQ1_S>;
    case GgmlType::IQ1_M: return vec_launch<GGML_TYPE_IQ1_M>;
    case GgmlType::IQ4_NL: return vec_launch<GGML_TYPE_IQ4_NL>;
    case GgmlType::IQ4_XS: return vec_launch<GGML_TYPE_IQ4_XS>;
    }
    throw std::invalid_argument("gguf vector product: unsupported block type");
}

// The kernels read a slice's 32 activation bytes as one 256-bit load (vec::load_slice).
void check_vector_activation(const void* activation) {
    if (reinterpret_cast<std::uintptr_t>(activation) % 32 != 0) {
        throw std::invalid_argument("gguf vector product: the activation must be 32-byte aligned");
    }
}

VecArgs vector_args(const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VecStore& out) {
    check_vector_activation(activation);
    const auto* base = static_cast<const std::uint8_t*>(activation);
    return VecArgs{static_cast<const std::uint8_t*>(weight),
                   row_bytes,
                   rows,
                   k,
                   reinterpret_cast<const std::int8_t*>(base),
                   reinterpret_cast<const half2*>(base + vec_scales_offset(k, columns)),
                   out};
}

constexpr int kMatrixQuantizeThreads = 128;

// Output column c reads x column column_map[c] / column_group (c without a map), its values
// gathered through input_columns when given; values from k up to k_padded are zero.
template <mmq_q8_1_ds_layout ds_layout>
__global__ void quantize_matrix_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::int32_t* __restrict__ input_columns,
                                       const std::int32_t* __restrict__ column_map,
                                       int column_group, block_q8_1_mmq* __restrict__ y, int k,
                                       int k_padded, int columns) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const std::int64_t i0 = (std::int64_t(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= k_padded) { return; }
    const int column             = blockIdx.x;
    const int source_column = column_map != nullptr ? column_map[column] / column_group : column;
    const __nv_bfloat16* source = x + std::int64_t(source_column) * k;
    float4 xi;
    if (i0 >= k) {
        xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if (input_columns != nullptr) {
        xi = make_float4(__bfloat162float(source[input_columns[i0 + 0]]),
                         __bfloat162float(source[input_columns[i0 + 1]]),
                         __bfloat162float(source[input_columns[i0 + 2]]),
                         __bfloat162float(source[input_columns[i0 + 3]]));
    } else {
        const uint2 words = *reinterpret_cast<const uint2*>(source + i0);
        const float2 lo   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.x));
        const float2 hi   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.y));
        xi                = make_float4(lo.x, lo.y, hi.x, hi.y);
    }

    const std::int64_t k_block = i0 / QK8_1_MMQ;
    const std::int64_t iqs     = i0 % QK8_1_MMQ;

    float amax = fabsf(xi.x);
    amax       = fmaxf(amax, fabsf(xi.y));
    amax       = fmaxf(amax, fabsf(xi.z));
    amax       = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x           = roundf(xi.x * d_inv);
    q.y           = roundf(xi.y * d_inv);
    q.z           = roundf(xi.z * d_inv);
    q.w           = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;

    const std::int64_t ib = k_block * columns + column;
    reinterpret_cast<char4*>(y[ib].qs)[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) { y[ib].d2s6[iqs / 64] = d; }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

__global__ void store_plane_kernel(const float* __restrict__ in, std::int64_t in_column_stride,
                                   int rows, VecStore out) {
    const int row    = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (row >= rows) { return; }
    vec_store(out, row, column, in[column * in_column_stride + row]);
}

constexpr int kTileQuantizeThreads = 128;
constexpr int kTileColumns         = 128;

// x BF16 [columns][k] -> the tile kernel's activation: quantize_matrix_kernel's D4 numbers, the
// values [columns][k] and d [k / 32][columns_pad], d = 0 for the padding columns. K is a whole
// number of 256-value steps, so every warp is wholly in or out.
__global__ void quantize_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                     const std::int32_t* __restrict__ input_columns,
                                     std::int8_t* __restrict__ qs, float* __restrict__ d, int k,
                                     int columns, int columns_pad) {
    const int i0     = (blockDim.x * blockIdx.y + threadIdx.x) * 4;
    const int column = blockIdx.x;
    if (i0 >= k) { return; }
    if (column >= columns) {
        if (i0 % 32 == 0) { d[std::int64_t(i0 / 32) * columns_pad + column] = 0.0f; }
        return;
    }
    const __nv_bfloat16* source = x + std::int64_t(column) * k;
    float4 xi;
    if (input_columns != nullptr) {
        xi = make_float4(__bfloat162float(source[input_columns[i0 + 0]]),
                         __bfloat162float(source[input_columns[i0 + 1]]),
                         __bfloat162float(source[input_columns[i0 + 2]]),
                         __bfloat162float(source[input_columns[i0 + 3]]));
    } else {
        const uint2 words = *reinterpret_cast<const uint2*>(source + i0);
        const float2 lo   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.x));
        const float2 hi   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.y));
        xi                = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float amax = fabsf(xi.x);
    amax       = fmaxf(amax, fabsf(xi.y));
    amax       = fmaxf(amax, fabsf(xi.z));
    amax       = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    *reinterpret_cast<char4*>(qs + std::int64_t(column) * k + i0) = q;
    if (i0 % 32 == 0) { d[std::int64_t(i0 / 32) * columns_pad + column] = 1.0f / d_inv; }
}

#if defined(NINFER_SM120_FP8)
// x BF16 [columns][k] -> an MXFP8 activation: per 32 values d = 2^E with max|x| / d in (224, 448],
// so no value saturates, and each value x / d rounded to e4m3 (RN); d = 1 for the padding columns.
// The tile kernel's form (kLt false) stores d as FP32 [k / 32][columns_pad], the two-pass
// product's (kLt) the biased exponent E at lt_scale_offset(column, i0 / 32, k / 32).
template <bool kLt>
__global__ void quantize_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::int32_t* __restrict__ input_columns,
                                    std::uint8_t* __restrict__ qs, void* __restrict__ scales, int k,
                                    int columns, int columns_pad) {
    const int i0     = (blockDim.x * blockIdx.y + threadIdx.x) * 4;
    const int column = blockIdx.x;
    if (i0 >= k) { return; }
    const auto put = [&](unsigned e) {
        if constexpr (kLt) {
            static_cast<std::uint8_t*>(scales)[lt_scale_offset(column, i0 / 32, k / 32)] =
                static_cast<std::uint8_t>(e);
        } else {
            static_cast<float*>(scales)[std::int64_t(i0 / 32) * columns_pad + column] =
                __uint_as_float(e << 23);
        }
    };
    if (column >= columns) {
        if (i0 % 32 == 0) { put(127u); }
        return;
    }
    const __nv_bfloat16* source = x + std::int64_t(column) * k;
    float4 xi;
    if (input_columns != nullptr) {
        xi = make_float4(__bfloat162float(source[input_columns[i0 + 0]]),
                         __bfloat162float(source[input_columns[i0 + 1]]),
                         __bfloat162float(source[input_columns[i0 + 2]]),
                         __bfloat162float(source[input_columns[i0 + 3]]));
    } else {
        const uint2 words = *reinterpret_cast<const uint2*>(source + i0);
        const float2 lo   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.x));
        const float2 hi   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.y));
        xi                = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float amax = fabsf(xi.x);
    amax       = fmaxf(amax, fabsf(xi.y));
    amax       = fmaxf(amax, fabsf(xi.z));
    amax       = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    // Scale 2^E with amax * 2^-E in (224, 448]: no value saturates.
    const unsigned bits = __float_as_uint(amax * (1.0f / 448.0f));
    const unsigned e =
        amax == 0.0f ? 127u : ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1u : 0u);
    const float unscale = __uint_as_float((254u - e) << 23);
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(lo)
        : "f"(xi.y * unscale), "f"(xi.x * unscale));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(hi)
        : "f"(xi.w * unscale), "f"(xi.z * unscale));
    *reinterpret_cast<unsigned*>(qs + std::int64_t(column) * k + i0) =
        unsigned(lo) | (unsigned(hi) << 16);
    if (i0 % 32 == 0) { put(e); }
}

// Values 16 group .. 16 group + 15 of column `column` of BF16 x [columns][k], element i read from
// x[column][input_columns[i]] when a gather is given.

// x BF16 [columns][k] (input columns gathered when given) -> lt_fp4_encode_group's activation, a
// thread a group of 16 values; max |x| over all groups goes to the zeroed slot.
// lt_fp4_scale_kernel then takes the scales to the global.
__launch_bounds__(kFp4EncodeThreads) __global__
    void lt_fp4_encode_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::int32_t* __restrict__ input_columns,
                              std::uint8_t* __restrict__ codes, std::uint8_t* __restrict__ scales,
                              std::int8_t* __restrict__ shifts, int k, int columns,
                              unsigned* __restrict__ slot) {
    const int groups     = k / 16;
    const std::int64_t i = blockIdx.x * std::int64_t(kFp4EncodeThreads) + threadIdx.x;
    float amax           = 0.0f;
    if (i < std::int64_t(columns) * groups) {
        const int column = int(i / groups);
        const int group  = int(i - std::int64_t(column) * groups);
        float v[16];
        lt_fp4_load16(x, input_columns, k, column, group, v);
#pragma unroll
        for (int j = 0; j < 16; ++j) { amax = fmaxf(amax, fabsf(v[j])); }
        lt_fp4_encode_group(v, amax, codes, scales, shifts, k, column, group);
    }
    lt_fp4_block_max(amax, slot);
}

// lt_fp4_encode_kernel's activation at its global g = 2^G (lt_fp4_global of the slot), a thread
// four scale bytes: the four groups of one column in a 128x4 scale tile, one aligned word of scales
// and one of shifts. fp4_encode16 is invariant under power-of-two scaling while every candidate
// scale stays a normal, unsaturated E4M3 value at both powers: a group at 2^n has its candidates in
// [0x5B, 0x69] (max |x| 2^n in [128, 256)), which moved by m = G - n octaves stay in [0x0B, 0x79]
// for m in [-10, 2], so its code moves by m octaves and its e2m1 codes stand. Any other group,
// among them the one holding max |x| (m 3 or 4), is encoded here from x g. Zero scales for the
// padding of the last 128-column tile and of the last 4-group tile.
__global__ void lt_fp4_scale_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::int32_t* __restrict__ input_columns,
                                    std::uint8_t* __restrict__ codes,
                                    std::uint8_t* __restrict__ scales,
                                    const std::int8_t* __restrict__ shifts, int k, int columns,
                                    std::int64_t scale_bytes, const unsigned* __restrict__ slot) {
    const std::int64_t at = 4 * (blockIdx.x * std::int64_t(blockDim.x) + threadIdx.x);
    if (at >= scale_bytes) { return; }
    // The inverse of lt_scale_offset; the word's bytes are groups group0 .. group0 + 3.
    const int groups      = k / 16;
    const int tile_groups = (groups + 3) / 4;
    const std::int64_t tile = at / 512;
    const int within        = int(at % 512);
    const int column = int(tile / tile_groups) * 128 + (within % 16) / 4 * 32 + within / 16;
    const int group0 = int(tile % tile_groups) * 4;
    auto* word       = reinterpret_cast<unsigned*>(scales + at);
    if (column >= columns) {
        *word = 0u;
        return;
    }
    const unsigned scale4 = *word;
    const unsigned shift4 = *reinterpret_cast<const unsigned*>(shifts + at);
    const float global    = lt_fp4_global(__uint_as_float(*slot));
    const int power       = int((__float_as_uint(global) >> 23) & 0xFFu) - 127;
    unsigned moved        = 0u;
    unsigned reencode     = 0u; // bit b: group group0 + b is encoded from x
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int scale = int((scale4 >> (8 * b)) & 0xFFu);
        const int shift = int(static_cast<std::int8_t>((shift4 >> (8 * b)) & 0xFFu));
        if (group0 + b >= groups || (scale == 0 && shift == 0)) { continue; } // zero at any global
        const int move = power - shift;
        if (shift != kFp4Reencode && move >= -10 && move <= 2) {
            moved |= unsigned(scale + 8 * move) << (8 * b);
        } else {
            reencode |= 1u << b;
        }
    }
#pragma unroll 1
    for (int b = 0; b < 4; ++b) {
        if ((reencode >> b & 1u) == 0u) { continue; }
        float v[16];
        lt_fp4_load16(x, input_columns, k, column, group0 + b, v);
#pragma unroll
        for (int j = 0; j < 16; ++j) { v[j] *= global; }
        uint2 c;
        const unsigned short s = fp4_encode16(v, c);
        *reinterpret_cast<uint2*>(codes + std::int64_t(column) * (k / 2) + 8 * (group0 + b)) = c;
        moved |= unsigned(s & 0xFFu) << (8 * b);
    }
    *word = moved;
}
#endif

// out[c, r] = act(plane[c, r]) * plane[c, up + r] for a gated two-pass slab: its BF16 products,
// gate rows from 0 and up rows from `up`, columns `ld` apart. A thread takes eight rows, in 16-byte
// accesses where all eight lie in the slab: the plane, `ld`, `up`, out and its column stride keep
// every group of eight rows aligned.
__global__ void lt_gate_kernel(const __nv_bfloat16* __restrict__ plane, std::int64_t ld, int up,
                               int rows, __nv_bfloat16* __restrict__ out,
                               std::int64_t out_column_stride, int activation) {
    const int row0   = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int column = blockIdx.y;
    if (row0 >= rows) { return; }
    const __nv_bfloat16* c = plane + column * ld;
    __nv_bfloat16* o       = out + column * out_column_stride;
    const auto product     = [&](__nv_bfloat16 gate, __nv_bfloat16 value) {
        return __float2bfloat16(vec_gate(__bfloat162float(gate), activation) *
                                __bfloat162float(value));
    };
    if (row0 + 8 <= rows) {
        const uint4 gate  = *reinterpret_cast<const uint4*>(c + row0);
        const uint4 value = *reinterpret_cast<const uint4*>(c + up + row0);
        const auto* g     = reinterpret_cast<const __nv_bfloat16*>(&gate);
        const auto* v     = reinterpret_cast<const __nv_bfloat16*>(&value);
        uint4 result;
        auto* r = reinterpret_cast<__nv_bfloat16*>(&result);
#pragma unroll
        for (int j = 0; j < 8; ++j) { r[j] = product(g[j], v[j]); }
        *reinterpret_cast<uint4*>(o + row0) = result;
        return;
    }
    for (int row = row0; row < rows; ++row) { o[row] = product(c[row], c[up + row]); }
}

#if defined(NINFER_SM120_FP8)
// lt_gate_kernel's products, a thread sixteen rows, for a pass whose rows start at row `first` of
// out, the BF16 [columns][k] input of the next product: they go to out and, encoded from their
// BF16 values, to that input's NVFP4 activation in lt_fp4_encode_kernel's form (codes, scales and
// shifts at their places, max |product| to the slot). `rows` and `first` are whole groups.
__launch_bounds__(kFp4EncodeThreads) __global__
    void lt_gate_fp4_kernel(const __nv_bfloat16* __restrict__ plane, std::int64_t ld, int up,
                            int rows, int columns, int first, int k,
                            __nv_bfloat16* __restrict__ out, int activation,
                            std::uint8_t* __restrict__ codes, std::uint8_t* __restrict__ scales,
                            std::int8_t* __restrict__ shifts, unsigned* __restrict__ slot) {
    const int pass_groups = rows / 16;
    const std::int64_t i  = blockIdx.x * std::int64_t(kFp4EncodeThreads) + threadIdx.x;
    float amax            = 0.0f;
    if (i < std::int64_t(columns) * pass_groups) {
        const int column = int(i / pass_groups);
        const int group  = int(i - std::int64_t(column) * pass_groups);
        const __nv_bfloat16* c = plane + column * ld + 16 * group;
        const uint4 gate[2]  = {reinterpret_cast<const uint4*>(c)[0],
                                reinterpret_cast<const uint4*>(c)[1]};
        const uint4 value[2] = {reinterpret_cast<const uint4*>(c + up)[0],
                                reinterpret_cast<const uint4*>(c + up)[1]};
        const auto* g        = reinterpret_cast<const __nv_bfloat16*>(gate);
        const auto* u        = reinterpret_cast<const __nv_bfloat16*>(value);
        uint4 result[2];
        auto* r = reinterpret_cast<__nv_bfloat16*>(result);
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            r[j] = __float2bfloat16(vec_gate(__bfloat162float(g[j]), activation) *
                                    __bfloat162float(u[j]));
            v[j] = __bfloat162float(r[j]);
            amax = fmaxf(amax, fabsf(v[j]));
        }
        // Evict-first: only lt_fp4_scale_kernel's rare re-encodes read the products back.
        auto* o = reinterpret_cast<uint4*>(out + std::int64_t(column) * k + first + 16 * group);
        __stcs(o, result[0]);
        __stcs(o + 1, result[1]);
        lt_fp4_encode_group(v, amax, codes, scales, shifts, k, column, first / 16 + group);
    }
    lt_fp4_block_max(amax, slot);
}
#endif

[[nodiscard]] int tile_columns_pad(int columns) {
    return (columns + kTileColumns - 1) / kTileColumns * kTileColumns;
}

__global__ void gather_columns_kernel(const __nv_bfloat16* __restrict__ x, int k,
                                      const std::int32_t* __restrict__ input_columns,
                                      __nv_bfloat16* __restrict__ out) {
    const int i      = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    out[std::int64_t(column) * k + i] = x[std::int64_t(column) * k + input_columns[i]];
}

cublasHandle_t blas_handle() {
    static std::array<cublasHandle_t, 16> handles{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    if (device < 0 || device >= int(handles.size())) {
        throw std::invalid_argument("gguf: device ordinal out of range");
    }
    // The first use can be inside a CUDA graph capture (a verify graph wide enough for the cuBLAS
    // route); relaxed capture mode lets cublasCreate allocate without invalidating it.
    std::call_once(once[device], [&] {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
        const cublasStatus_t created = cublasCreate(&handles[device]);
        check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
        if (created != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("gguf: cublasCreate failed");
        }
    });
    return handles[device];
}

// Types whose blocks are narrower than the 256-value super-block one kernel block decodes: their
// rows need only be whole blocks, so the last super-block of a row may be partial.
template <ggml_type type>
inline constexpr bool kSmallBlocks =
    type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
    type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ4_NL;

template <ggml_type type, int threads, typename dst_t>
__global__ void dequantize_rows_kernel(const char* __restrict__ weight, std::int64_t row_bytes,
                                       const std::int32_t* __restrict__ row_ids,
                                       dst_t* __restrict__ out, std::int64_t out_row_stride,
                                       int k) {
    const int row          = blockIdx.y;
    const std::int64_t src = row_ids != nullptr ? row_ids[row] : row;
    const void* x          = weight + src * row_bytes;
    dst_t* y               = out + std::int64_t(row) * out_row_stride + std::int64_t(blockIdx.x) * QK_K;
    const int count        = min(QK_K, k - int(blockIdx.x) * QK_K);
    if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0* blocks = static_cast<const block_q8_0*>(x) + blockIdx.x * (QK_K / QK8_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q8_0& b = blocks[i / QK8_0];
            y[i] = ggml_cuda_cast<dst_t>(__half2float(b.d) * b.qs[i % QK8_0]);
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const block_q4_0* blocks = static_cast<const block_q4_0*>(x) + blockIdx.x * (QK_K / QK4_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q4_0& b = blocks[i / QK4_0];
            const int j         = i % QK4_0;
            const int code      = j < QK4_0 / 2 ? (b.qs[j] & 0xF) : (b.qs[j - QK4_0 / 2] >> 4);
            y[i]                = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(code - 8));
        }
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        const block_q5_0* blocks = static_cast<const block_q5_0*>(x) + blockIdx.x * (QK_K / QK5_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q5_0& b = blocks[i / QK5_0];
            const int j         = i % QK5_0;
            std::uint32_t qh;
            memcpy(&qh, b.qh, sizeof(qh));
            const int low  = j < QK5_0 / 2 ? (b.qs[j] & 0xF) : (b.qs[j - QK5_0 / 2] >> 4);
            const int code = low | int(((qh >> j) & 1u) << 4);
            y[i]           = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(code - 16));
        }
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const block_q2_0* blocks = static_cast<const block_q2_0*>(x) + blockIdx.x * (QK_K / QK2_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q2_0& b = blocks[i / QK2_0];
            const int j         = i % QK2_0;
            const int code      = (b.qs[j / 4] >> (2 * (j % 4))) & 3;
            y[i]                = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(code - 1));
        }
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        dequantize_q2_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        dequantize_q3_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        dequantize_q4_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        dequantize_q5_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        dequantize_q6_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        dequantize_iq2_xxs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        dequantize_iq2_xs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        dequantize_iq2_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        dequantize_iq3_xxs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        dequantize_iq3_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ1_S) {
        dequantize_iq1_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ1_M) {
        dequantize_iq1_m(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl* blocks =
            static_cast<const block_iq4_nl*>(x) + blockIdx.x * (QK_K / QK4_NL);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_iq4_nl& b = blocks[i / QK4_NL];
            const int j           = i % QK4_NL;
            const int code        = j < QK4_NL / 2 ? (b.qs[j] & 0xF) : (b.qs[j - QK4_NL / 2] >> 4);
            y[i] = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(kvalues_iq4nl[code]));
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        dequantize_iq4_xs(x, blockIdx.x, y, threadIdx.x);
    }
}

template <ggml_type type, int threads, typename dst_t>
void launch_dequantize(const void* weight, std::int64_t row_bytes, int k,
                       const std::int32_t* row_ids, int rows, dst_t* out,
                       std::int64_t out_row_stride, cudaStream_t stream) {
    if (k <= 0 || k % (kSmallBlocks<type> ? ggml_cuda_type_traits<type>::qk : QK_K) != 0 ||
        rows <= 0) {
        throw std::invalid_argument("gguf dequantize: K must be whole blocks of the type");
    }
    // The slice decoders write whole 16-byte groups; this kernel stays for the rest.
    if constexpr (kSliceDequantize<type>) {
        if (out_row_stride % 8 == 0 && reinterpret_cast<std::uintptr_t>(out) % 16 == 0) {
            slice_dequantize<type>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
            return;
        }
    }
    dequantize_rows_kernel<type, threads, dst_t>
        <<<dim3((k + QK_K - 1) / QK_K, rows, 1), threads, 0, stream>>>(
            static_cast<const char*>(weight), row_bytes, row_ids, out, out_row_stride, k);
    check(cudaGetLastError(), "dequantize launch");
}

template <typename dst_t>
void dispatch_dequantize(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                         const std::int32_t* row_ids, int rows, dst_t* out,
                         std::int64_t out_row_stride, cudaStream_t stream) {
    switch (type) {
    case GgmlType::Q8_0: return launch_dequantize<GGML_TYPE_Q8_0, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q4_0:
        return launch_dequantize<GGML_TYPE_Q4_0, 64>(weight, row_bytes, k, row_ids, rows, out,
                                                     out_row_stride, stream);
    case GgmlType::Q5_0:
        return launch_dequantize<GGML_TYPE_Q5_0, 64>(weight, row_bytes, k, row_ids, rows, out,
                                                     out_row_stride, stream);
    case GgmlType::Q2_0:
        return launch_dequantize<GGML_TYPE_Q2_0, 64>(weight, row_bytes, k, row_ids, rows, out,
                                                     out_row_stride, stream);
    case GgmlType::Q2_K: return launch_dequantize<GGML_TYPE_Q2_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q3_K: return launch_dequantize<GGML_TYPE_Q3_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q4_K: return launch_dequantize<GGML_TYPE_Q4_K, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q5_K: return launch_dequantize<GGML_TYPE_Q5_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q6_K: return launch_dequantize<GGML_TYPE_Q6_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_XXS: return launch_dequantize<GGML_TYPE_IQ2_XXS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_XS: return launch_dequantize<GGML_TYPE_IQ2_XS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_S: return launch_dequantize<GGML_TYPE_IQ2_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ3_XXS: return launch_dequantize<GGML_TYPE_IQ3_XXS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ3_S: return launch_dequantize<GGML_TYPE_IQ3_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ1_S: return launch_dequantize<GGML_TYPE_IQ1_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ1_M: return launch_dequantize<GGML_TYPE_IQ1_M, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ4_NL: return launch_dequantize<GGML_TYPE_IQ4_NL, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ4_XS: return launch_dequantize<GGML_TYPE_IQ4_XS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    }
    throw std::invalid_argument("gguf dequantize: unsupported block type");
}

} // namespace
} // namespace detail

BlockShape block_shape(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return {32, 34};
    case GgmlType::Q4_0:
        return {32, 18};
    case GgmlType::Q5_0:
        return {32, 22};
    case GgmlType::Q2_0:
        return {64, 18};
    case GgmlType::Q2_K: return {256, 84};
    case GgmlType::Q3_K: return {256, 110};
    case GgmlType::Q4_K: return {256, 144};
    case GgmlType::Q5_K: return {256, 176};
    case GgmlType::Q6_K: return {256, 210};
    case GgmlType::IQ2_XXS: return {256, 66};
    case GgmlType::IQ2_XS: return {256, 74};
    case GgmlType::IQ3_XXS: return {256, 98};
    case GgmlType::IQ1_S: return {256, 50};
    case GgmlType::IQ4_NL: return {32, 18};
    case GgmlType::IQ3_S: return {256, 110};
    case GgmlType::IQ2_S: return {256, 82};
    case GgmlType::IQ4_XS: return {256, 136};
    case GgmlType::IQ1_M: return {256, 56};
    }
    throw std::invalid_argument("gguf: unsupported block type");
}

bool has_matrix_kernel(GgmlType type) { return type != GgmlType::IQ1_M; }

int multiprocessor_count() { return detail::device_facts().sm_count; }

std::size_t vector_activation_scales_offset(int k, int columns) {
    return detail::vec_scales_offset(k, columns);
}

std::size_t vector_activation_bytes(int k, int columns) {
    return detail::vec_scales_offset(k, columns) + std::size_t(columns) * (k / QK8_1) * sizeof(half2);
}

void quantize_vector_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out,
                                cudaStream_t stream) {
    // Whole warps of 32 values: a warp past K leaves as one, before its reductions.
    if (k <= 0 || k % QK8_1 != 0 || columns <= 0 || columns > 65535) {
        throw std::invalid_argument("gguf vector activation: K must be whole 32-value blocks");
    }
    auto* base = static_cast<std::uint8_t*>(out);
    const dim3 blocks((k + detail::kQuantizeThreads - 1) / detail::kQuantizeThreads, columns, 1);
    detail::quantize_vector_kernel<<<blocks, detail::kQuantizeThreads, 0, stream>>>(
        x, input_columns, reinterpret_cast<std::int8_t*>(base),
        reinterpret_cast<half2*>(base + detail::vec_scales_offset(k, columns)), k);
    detail::check(cudaGetLastError(), "vector activation launch");
}

void vector_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr)) {
        throw std::invalid_argument("gguf vector product: invalid output");
    }
    const detail::VecStore store{out.bf16,     out.f32,  out.column_stride,
                                 out.epilogue, out.gate, out.gate_column_stride,
                                 static_cast<int>(out.activation)};
    detail::vector_launch(type)(
        detail::vector_args(weight, row_bytes, rows, k, activation, columns, store), columns,
        false, stream);
}

void vector_swiglu(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                   const void* activation, int columns, __nv_bfloat16* out,
                   std::int64_t out_column_stride, cudaStream_t stream,
                   GateActivation gate_activation) {
    const detail::VecStore store{out,     nullptr, out_column_stride,
                                 Epilogue::Store, nullptr, 0,
                                 static_cast<int>(gate_activation)};
    detail::vector_launch(type)(
        detail::vector_args(weight, row_bytes, rows, k, activation, columns, store), columns,
        true, stream);
}

bool has_narrow_kernel(GgmlType type, int k) {
    switch (type) {
    case GgmlType::IQ3_XXS:
    case GgmlType::IQ3_S:
    case GgmlType::IQ2_XXS:
    case GgmlType::IQ2_XS:
    case GgmlType::IQ2_S:
    case GgmlType::Q2_K: break;
    default: return false;
    }
    const int cc = detail::device_facts().cc;
    return cc >= 1200 && cc < 1300 && k > 0 && k % 1024 == 0;
}

// Below these widths the vector kernel is as fast or faster: on the RTX 5060 Ti, the start of each
// type's largest summed gain over 5-8 columns across the 27B's shapes (count-weighted; IQ2_XXS's 5
// and 7, IQ3_XXS's 5 and 6 tie within the timer's 2 us).
int narrow_min_columns(GgmlType type) {
    switch (type) {
    case GgmlType::IQ3_S:
    case GgmlType::Q2_K: return 5;
    case GgmlType::IQ2_XXS: return 7;
    default: return 6;
    }
}

bool takes_narrow_kernel(GgmlType type, int k, int columns) {
    return columns >= narrow_min_columns(type) && columns <= kMaxNarrowColumns &&
           has_narrow_kernel(type, k);
}

std::size_t narrow_activation_scales_offset(int k, int columns) {
    return (std::size_t(k) * columns + 255) / 256 * 256;
}

std::size_t narrow_activation_bytes(int k, int columns) {
    const std::size_t padded = (std::size_t(columns) + 7) / 8 * 8;
    return narrow_activation_scales_offset(k, columns) + std::size_t(k / 64) * padded * 2 * sizeof(float);
}

void quantize_narrow_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out,
                                cudaStream_t stream) {
    if (k <= 0 || k % 64 != 0 || columns <= 0 || columns > kMaxNarrowColumns ||
        reinterpret_cast<std::uintptr_t>(out) % 16 != 0) {
        throw std::invalid_argument(
            "gguf narrow activation: K must be whole 64-value pairs, 1 to 32 columns, 16-byte aligned");
    }
    auto* base            = static_cast<std::uint8_t*>(out);
    const int columns_pad = (columns + 7) / 8 * 8;
    const dim3 blocks((k + detail::kQuantizeThreads - 1) / detail::kQuantizeThreads, columns_pad, 1);
    detail::quantize_narrow_kernel<<<blocks, detail::kQuantizeThreads, 0, stream>>>(
        x, input_columns, reinterpret_cast<std::int8_t*>(base),
        reinterpret_cast<float*>(base + narrow_activation_scales_offset(k, columns)), k, columns,
        columns_pad);
    detail::check(cudaGetLastError(), "narrow activation launch");
}

void narrow_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr) || rows <= 0 ||
        columns <= 0 || columns > kMaxNarrowColumns || !has_narrow_kernel(type, k) ||
        reinterpret_cast<std::uintptr_t>(weight) % 2 != 0 || row_bytes % 2 != 0 ||
        row_bytes > (std::int64_t(1) << 24) ||
        reinterpret_cast<std::uintptr_t>(activation) % 16 != 0) {
        throw std::invalid_argument("gguf narrow product: invalid product");
    }
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const detail::NarrowArgs args{
        static_cast<const std::uint8_t*>(weight),
        row_bytes,
        rows,
        k,
        columns,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const float*>(base + narrow_activation_scales_offset(k, columns)),
        detail::VecStore{out.bf16, out.f32, out.column_stride, out.epilogue, out.gate,
                         out.gate_column_stride, static_cast<int>(out.activation)}};
    switch (type) {
    case GgmlType::IQ3_XXS: return detail::narrow_launch<GGML_TYPE_IQ3_XXS>(args, stream);
    case GgmlType::IQ3_S: return detail::narrow_launch<GGML_TYPE_IQ3_S>(args, stream);
    case GgmlType::IQ2_XXS: return detail::narrow_launch<GGML_TYPE_IQ2_XXS>(args, stream);
    case GgmlType::IQ2_XS: return detail::narrow_launch<GGML_TYPE_IQ2_XS>(args, stream);
    case GgmlType::IQ2_S: return detail::narrow_launch<GGML_TYPE_IQ2_S>(args, stream);
    case GgmlType::Q2_K: return detail::narrow_launch<GGML_TYPE_Q2_K>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf narrow product: unsupported block type");
}

int narrow_row_warps(GgmlType type, int rows, int columns) {
    int block_bytes = 0;
    switch (type) {
    case GgmlType::IQ3_XXS: block_bytes = sizeof(block_iq3_xxs); break;
    case GgmlType::IQ3_S: block_bytes = sizeof(block_iq3_s); break;
    case GgmlType::IQ2_XXS: block_bytes = sizeof(block_iq2_xxs); break;
    case GgmlType::IQ2_XS: block_bytes = sizeof(block_iq2_xs); break;
    case GgmlType::IQ2_S: block_bytes = sizeof(block_iq2_s); break;
    case GgmlType::Q2_K: block_bytes = sizeof(block_q2_K); break;
    default: throw std::invalid_argument("gguf narrow product: unsupported block type");
    }
    return detail::narrow_row_warps(block_bytes, rows, columns, detail::device_facts().sm_count);
}

std::size_t moe_routing_bytes(int experts, int pairs) {
    return (std::size_t(experts) + 1 + std::size_t(pairs) + std::size_t(experts) + 1) *
               sizeof(std::int32_t) +
           256;
}

MoeRouting moe_sort_routes(const std::int32_t* ids, int pairs, int experts, void* workspace,
                           cudaStream_t stream) {
    if (pairs <= 0 || experts <= 0 || experts > 8192 || workspace == nullptr) {
        throw std::invalid_argument("gguf moe routing: invalid pairs or experts");
    }
    auto* base                 = static_cast<std::int32_t*>(workspace);
    std::int32_t* bounds       = base;
    std::int32_t* sorted       = bounds + experts + 1;
    std::int32_t* active       = sorted + pairs;
    std::int32_t* active_count = active + experts;
    detail::moe_sort_kernel<<<1, detail::kSortThreads, 2 * experts * sizeof(int), stream>>>(
        ids, pairs, experts, bounds, sorted, active, active_count);
    detail::check(cudaGetLastError(), "moe routing launch");
    return MoeRouting{bounds, sorted, active, active_count};
}

namespace {

detail::MoeVecArgs moe_args(const MoeTable& first, const MoeTable* second,
                            const MoeRouting& routing, int column_group, int per_token,
                            const void* activation, int columns) {
    if (first.experts == nullptr || first.rows <= 0 || first.k <= 0 || column_group <= 0 ||
        per_token <= 0 || columns <= 0 || activation == nullptr || routing.bounds == nullptr ||
        (second != nullptr && (second->experts == nullptr || second->rows != first.rows ||
                               second->k != first.k || second->row_bytes != first.row_bytes))) {
        throw std::invalid_argument("gguf moe product: invalid tables or routing");
    }
    detail::check_vector_activation(activation);
    const auto* base = static_cast<const std::uint8_t*>(activation);
    detail::MoeVecArgs args{};
    args.first = reinterpret_cast<const std::uint8_t* const*>(first.experts);
    args.second =
        second != nullptr ? reinterpret_cast<const std::uint8_t* const*>(second->experts) : nullptr;
    args.row_bytes    = first.row_bytes;
    args.rows         = first.rows;
    args.k            = first.k;
    args.bounds       = routing.bounds;
    args.sorted       = routing.sorted;
    args.active       = routing.active;
    args.active_count = routing.active_count;
    args.column_group = column_group;
    args.per_token    = per_token;
    args.qs           = reinterpret_cast<const std::int8_t*>(base);
    args.ds = reinterpret_cast<const half2*>(base + detail::vec_scales_offset(first.k, columns));
    return args;
}

} // namespace

void moe_vector_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int column_group, int per_token, const void* activation,
                        int columns, const MoeOutput& out, int chunk, cudaStream_t stream) {
    const int targets =
        int(out.bf16 != nullptr) + int(out.f32 != nullptr) + int(out.weighted != nullptr);
    if (targets != 1 || (out.weighted != nullptr && out.weights == nullptr)) {
        throw std::invalid_argument("gguf moe product: invalid output");
    }
    auto args     = moe_args(table, nullptr, routing, column_group, per_token, activation, columns);
    args.out_bf16 = out.bf16;
    args.out_f32  = out.f32;
    args.weighted = out.weighted;
    args.weights    = out.weights;
    args.gate       = out.gate;
    args.activation = static_cast<int>(out.activation);
    detail::moe_launcher(type)(args, max_active, chunk, false, stream);
}

void moe_vector_swiglu(GgmlType type, const MoeTable& gate, const MoeTable& up,
                       const MoeRouting& routing, int max_active, int column_group,
                       const void* activation, int columns, __nv_bfloat16* out, int chunk,
                       cudaStream_t stream, GateActivation gate_activation) {
    if (out == nullptr) { throw std::invalid_argument("gguf moe swiglu: invalid output"); }
    auto args       = moe_args(gate, &up, routing, column_group, 1, activation, columns);
    args.out_bf16   = out;
    args.activation = static_cast<int>(gate_activation);
    detail::moe_launcher(type)(args, max_active, chunk, true, stream);
}

void moe_decode_up(GgmlType type, const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream) {
    if (args.m == nullptr || args.gate == nullptr || args.up == nullptr || args.middle == nullptr) {
        throw std::invalid_argument("gguf moe decode: invalid up arguments");
    }
    detail::decode_launcher<detail::DecodeLaunch<MoeDecodeUpArgs>, detail::PickUp>(type)(args, chunk,
                                                                                      stream);
}

void moe_decode_down(GgmlType type, const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream) {
    if (args.table_a == nullptr || args.middle == nullptr || args.weights == nullptr ||
        args.fixed == nullptr || (args.y != nullptr && args.counter == nullptr) ||
        (args.ids_b != nullptr && args.table_b == nullptr)) {
        throw std::invalid_argument("gguf moe decode: invalid down arguments");
    }
    detail::decode_launcher<detail::DecodeLaunch<MoeDecodeDownArgs>, detail::PickDown>(type)(
        args, chunk, stream);
}

void moe_decode_prep(GgmlType type, const MoeDecodePrepArgs& args, cudaStream_t stream) {
    if (args.m == nullptr || args.ids == nullptr || args.ds == nullptr || args.qsum == nullptr) {
        throw std::invalid_argument("gguf moe decode: invalid preparation arguments");
    }
    detail::decode_launcher<void (*)(const MoeDecodePrepArgs&, cudaStream_t), detail::PickPrep>(
        type)(args, stream);
}

bool moe_decode_transposed(GgmlType type) {
    // Decoder<type>::kTransposed of ggml_bridge_vec.cuh.
    return type == GgmlType::Q2_0;
}


int matrix_activation_layout(GgmlType type) {
    return static_cast<int>(mmq_get_q8_1_ds_layout(detail::to_ggml(type)));
}

std::size_t matrix_activation_bytes(int k, int columns) {
    return std::size_t(columns) * (k / QK8_1_MMQ) * sizeof(block_q8_1_mmq) +
           detail::kMatrixActivationSlack;
}

namespace {

// k_padded is a whole number of 128-value blocks, so every warp of the grid is wholly in or out.
void quantize_matrix(GgmlType type, const __nv_bfloat16* x, int k, int k_padded, int columns,
                     const std::int32_t* input_columns, const std::int32_t* column_map,
                     int column_group, void* out, cudaStream_t stream) {
    const dim3 blocks(columns,
                      (k_padded + 4 * detail::kMatrixQuantizeThreads - 1) /
                          (4 * detail::kMatrixQuantizeThreads),
                      1);
    auto* y = static_cast<block_q8_1_mmq*>(out);
    switch (mmq_get_q8_1_ds_layout(detail::to_ggml(type))) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, input_columns, column_map, column_group, y, k, k_padded, columns);
        break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, input_columns, column_map, column_group, y, k, k_padded, columns);
        break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, input_columns, column_map, column_group, y, k, k_padded, columns);
        break;
    }
    detail::check(cudaGetLastError(), "matrix activation launch");
}

} // namespace

void quantize_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream) {
    if (k <= 0 || k % (4 * detail::kMatrixQuantizeThreads) != 0 || columns <= 0) {
        throw std::invalid_argument("gguf matrix activation: K must be whole 512-value groups");
    }
    quantize_matrix(type, x, k, k, columns, input_columns, nullptr, 1, out, stream);
}

namespace {

// The matrix kernel's K step: an activation covers whole steps.
constexpr int kMatrixStep = 256;

int moe_padded_k(int k) { return (k + kMatrixStep - 1) / kMatrixStep * kMatrixStep; }

} // namespace

std::size_t moe_matrix_activation_bytes(int k, int pairs) {
    return matrix_activation_bytes(moe_padded_k(k), pairs);
}

int matrix_padded_k(int k) { return moe_padded_k(k); }

void quantize_matrix_activation_padded(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                       const std::int32_t* input_columns, void* out,
                                       cudaStream_t stream) {
    if (k <= 0 || k % 32 != 0 || columns <= 0) {
        throw std::invalid_argument("gguf padded matrix activation: K must be whole 32-value blocks");
    }
    quantize_matrix(type, x, k, moe_padded_k(k), columns, input_columns, nullptr, 1, out, stream);
}

void quantize_moe_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k,
                                    const MoeRouting& routing, int pairs, int column_group,
                                    void* out, cudaStream_t stream) {
    if (routing.sorted == nullptr || column_group <= 0 || k <= 0 || k % 32 != 0 || pairs <= 0) {
        throw std::invalid_argument("gguf moe matrix activation: invalid routing or K");
    }
    quantize_matrix(type, x, k, moe_padded_k(k), pairs, nullptr, routing.sorted, column_group, out,
                    stream);
}

std::size_t matrix_fixup_bytes(GgmlType type, int rows, int columns) {
    switch (type) {
    case GgmlType::Q8_0: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q8_0>(rows, columns);
    case GgmlType::Q4_0:
        return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q4_0>(rows, columns);
    case GgmlType::Q5_0:
        return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q5_0>(rows, columns);
    case GgmlType::Q2_0:
        return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q2_0>(rows, columns);
    case GgmlType::Q2_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q2_K>(rows, columns);
    case GgmlType::Q3_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q3_K>(rows, columns);
    case GgmlType::Q4_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q4_K>(rows, columns);
    case GgmlType::Q5_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q5_K>(rows, columns);
    case GgmlType::Q6_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q6_K>(rows, columns);
    case GgmlType::IQ2_XXS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_XXS>(rows, columns);
    case GgmlType::IQ2_XS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_XS>(rows, columns);
    case GgmlType::IQ2_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_S>(rows, columns);
    case GgmlType::IQ3_XXS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ3_XXS>(rows, columns);
    case GgmlType::IQ3_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ3_S>(rows, columns);
    case GgmlType::IQ1_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ1_S>(rows, columns);
    case GgmlType::IQ4_NL: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ4_NL>(rows, columns);
    case GgmlType::IQ4_XS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ4_XS>(rows, columns);
    case GgmlType::IQ1_M: break;
    }
    throw std::invalid_argument("gguf matrix product: no integer kernel for this block type");
}

void matrix_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, float* out,
                    std::int64_t out_column_stride, void* fixup, cudaStream_t stream, bool tail) {
#define NINFER_GGUF_MATRIX_CASE(NAME)                                                              \
    case GgmlType::NAME:                                                                           \
        detail::matrix_product_impl<GGML_TYPE_##NAME>(weight, row_bytes, rows, k, activation,      \
                                                      columns, out, out_column_stride, fixup,      \
                                                      stream, tail);                               \
        return
    switch (type) {
        NINFER_GGUF_MATRIX_CASE(Q8_0);
        NINFER_GGUF_MATRIX_CASE(Q4_0);
        NINFER_GGUF_MATRIX_CASE(Q5_0);
        NINFER_GGUF_MATRIX_CASE(Q2_0);
        NINFER_GGUF_MATRIX_CASE(Q2_K);
        NINFER_GGUF_MATRIX_CASE(Q3_K);
        NINFER_GGUF_MATRIX_CASE(Q4_K);
        NINFER_GGUF_MATRIX_CASE(Q5_K);
        NINFER_GGUF_MATRIX_CASE(Q6_K);
        NINFER_GGUF_MATRIX_CASE(IQ2_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ2_XS);
        NINFER_GGUF_MATRIX_CASE(IQ2_S);
        NINFER_GGUF_MATRIX_CASE(IQ3_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ3_S);
        NINFER_GGUF_MATRIX_CASE(IQ1_S);
        NINFER_GGUF_MATRIX_CASE(IQ4_NL);
        NINFER_GGUF_MATRIX_CASE(IQ4_XS);
    case GgmlType::IQ1_M:
        break;
    }
#undef NINFER_GGUF_MATRIX_CASE
    throw std::invalid_argument("gguf matrix product: no integer kernel for this block type");
}

bool moe_matrix_supported(GgmlType type, int rows, int k, bool tail) {
#define NINFER_GGUF_MOE_FITS_CASE(NAME)                                                            \
    case GgmlType::NAME:                                                                           \
        return detail::moe_matrix_fits_impl<GGML_TYPE_##NAME>(rows, k, tail)
    switch (type) {
        NINFER_GGUF_MOE_FITS_CASE(Q8_0);
        NINFER_GGUF_MOE_FITS_CASE(Q4_0);
        NINFER_GGUF_MOE_FITS_CASE(Q5_0);
        NINFER_GGUF_MOE_FITS_CASE(Q2_0);
        NINFER_GGUF_MOE_FITS_CASE(Q2_K);
        NINFER_GGUF_MOE_FITS_CASE(Q3_K);
        NINFER_GGUF_MOE_FITS_CASE(Q4_K);
        NINFER_GGUF_MOE_FITS_CASE(Q5_K);
        NINFER_GGUF_MOE_FITS_CASE(Q6_K);
        NINFER_GGUF_MOE_FITS_CASE(IQ2_XXS);
        NINFER_GGUF_MOE_FITS_CASE(IQ2_XS);
        NINFER_GGUF_MOE_FITS_CASE(IQ2_S);
        NINFER_GGUF_MOE_FITS_CASE(IQ3_XXS);
        NINFER_GGUF_MOE_FITS_CASE(IQ3_S);
        NINFER_GGUF_MOE_FITS_CASE(IQ1_S);
        NINFER_GGUF_MOE_FITS_CASE(IQ4_NL);
        NINFER_GGUF_MOE_FITS_CASE(IQ4_XS);
    case GgmlType::IQ1_M:
        return false;
    }
#undef NINFER_GGUF_MOE_FITS_CASE
    return false;
}

void moe_matrix_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int pairs, int max_columns, bool tail,
                        const void* activation, float* out, cudaStream_t stream) {
    if (table.experts == nullptr || routing.bounds == nullptr || activation == nullptr ||
        out == nullptr) {
        throw std::invalid_argument("gguf moe matrix product: invalid tables or routing");
    }
#define NINFER_GGUF_MOE_MATRIX_CASE(NAME)                                                          \
    case GgmlType::NAME:                                                                           \
        detail::moe_matrix_product_impl<GGML_TYPE_##NAME>(                                         \
            table.experts, table.row_bytes, table.rows, table.k, routing, max_active, pairs,       \
            max_columns, tail, activation, out, stream);                                           \
        return
    switch (type) {
        NINFER_GGUF_MOE_MATRIX_CASE(Q8_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q4_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q5_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q2_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q2_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q3_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q4_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q5_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q6_K);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ2_XXS);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ2_XS);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ2_S);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ3_XXS);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ3_S);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ1_S);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ4_NL);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ4_XS);
    case GgmlType::IQ1_M:
        break;
    }
#undef NINFER_GGUF_MOE_MATRIX_CASE
    throw std::invalid_argument("gguf moe matrix product: no integer kernel for this block type");
}

bool has_tile_kernel(GgmlType type, int k, bool mxfp8) {
    if (mxfp8 && !kTileMxfp8Built) { return false; }
    switch (type) {
    case GgmlType::IQ3_XXS:
    case GgmlType::IQ3_S:
    case GgmlType::IQ2_XXS:
    case GgmlType::IQ4_XS:
    case GgmlType::IQ2_XS:
    case GgmlType::IQ2_S: break;
    case GgmlType::Q2_K:
    case GgmlType::Q4_K:
        if (!mxfp8) { return false; }
        break;
    default: return false;
    }
    const int cc = detail::device_facts().cc;
    return cc >= 1200 && cc < 1300 && k > 0 && k % 256 == 0;
}

bool takes_tile_kernel(GgmlType type, int rows, int k, int columns, bool mxfp8) {
    if (columns < kTileMinColumns || !has_tile_kernel(type, k, mxfp8)) { return false; }
    if (rows % 128 != 0) { return true; }
    // The share of its waves' CTA slots (one CTA per SM) the tile grid must fill, in percent: on
    // the RTX 5060 Ti (ninfer_linear_bench over the 27B's shapes, T 128-1024) the tile kernel ran
    // 1.21-1.45x the matrix kernel's rate times that fill, so every type lost at a fill of 0.74 and
    // won from 0.83 except IQ4_XS (1.07-1.24x), even at 0.89 and winning from 0.94. The fills were
    // measured for the q8_1 form; the MXFP8 one, faster still, keeps them.
    const int min_fill = type == GgmlType::IQ4_XS ? 90 : 80;
    const std::int64_t sms   = detail::device_facts().sm_count;
    const std::int64_t tiles = std::int64_t(rows / 128) * ((columns + 127) / 128);
    const std::int64_t waves = (tiles + sms - 1) / sms;
    return tiles * 100 >= waves * sms * min_fill;
}

std::size_t tile_activation_scales_offset(int k, int columns) {
    return (std::size_t(k) * columns + 255) / 256 * 256;
}

std::size_t tile_activation_bytes(int k, int columns) {
    return tile_activation_scales_offset(k, columns) +
           std::size_t(k / 32) * detail::tile_columns_pad(columns) * sizeof(float);
}

void quantize_tile_activation(const __nv_bfloat16* x, int k, int columns,
                              const std::int32_t* input_columns, bool mxfp8, void* out,
                              cudaStream_t stream) {
    if (k <= 0 || k % 256 != 0 || columns <= 0) {
        throw std::invalid_argument("gguf tile activation: K must be whole 256-value steps");
    }
    if (mxfp8 && !kTileMxfp8Built) {
        throw std::invalid_argument("gguf tile activation: this build has no MXFP8 form");
    }
    auto* base = static_cast<std::uint8_t*>(out);
    const int columns_pad = detail::tile_columns_pad(columns);
    const dim3 blocks(columns_pad,
                      (k + 4 * detail::kTileQuantizeThreads - 1) /
                          (4 * detail::kTileQuantizeThreads),
                      1);
    if (mxfp8) {
#if defined(NINFER_SM120_FP8)
        detail::quantize_fp8_kernel<false><<<blocks, detail::kTileQuantizeThreads, 0, stream>>>(
            x, input_columns, base, base + tile_activation_scales_offset(k, columns), k, columns,
            columns_pad);
#endif
    } else {
        detail::quantize_tile_kernel<<<blocks, detail::kTileQuantizeThreads, 0, stream>>>(
            x, input_columns, reinterpret_cast<std::int8_t*>(base),
            reinterpret_cast<float*>(base + tile_activation_scales_offset(k, columns)), k, columns,
            columns_pad);
    }
    detail::check(cudaGetLastError(), "tile activation launch");
}

void tile_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                  const void* activation, bool mxfp8, int columns, const VectorOutput& out,
                  cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr) || rows <= 0 ||
        columns <= 0 || !has_tile_kernel(type, k, mxfp8)) {
        throw std::invalid_argument("gguf tile product: invalid product");
    }
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const detail::TileArgs args{
        static_cast<const std::uint8_t*>(weight),
        row_bytes,
        rows,
        k,
        columns,
        detail::tile_columns_pad(columns),
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const float*>(base + tile_activation_scales_offset(k, columns)),
        detail::VecStore{out.bf16, out.f32, out.column_stride, out.epilogue, out.gate,
                         out.gate_column_stride, static_cast<int>(out.activation)},
        mxfp8};
    switch (type) {
    case GgmlType::IQ3_XXS: return detail::tile_launch<GGML_TYPE_IQ3_XXS>(args, stream);
    case GgmlType::IQ3_S: return detail::tile_launch<GGML_TYPE_IQ3_S>(args, stream);
    case GgmlType::IQ2_XXS: return detail::tile_launch<GGML_TYPE_IQ2_XXS>(args, stream);
    case GgmlType::IQ4_XS: return detail::tile_launch<GGML_TYPE_IQ4_XS>(args, stream);
    case GgmlType::IQ2_XS: return detail::tile_launch<GGML_TYPE_IQ2_XS>(args, stream);
    case GgmlType::IQ2_S: return detail::tile_launch<GGML_TYPE_IQ2_S>(args, stream);
    case GgmlType::Q2_K: return detail::tile_launch<GGML_TYPE_Q2_K>(args, stream);
    case GgmlType::Q4_K: return detail::tile_launch<GGML_TYPE_Q4_K>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf tile product: no tile kernel for this block type");
}

namespace detail {
namespace {

constexpr std::size_t kLtAlign = 256;

// Bytes of an image or activation row of k values, and its scale groups.
std::size_t lt_row_bytes(int k, LtForm form) {
    return form == LtForm::Nvfp4 ? std::size_t(k) / 2 : std::size_t(k);
}
int lt_groups(int k, LtForm form) { return form == LtForm::Nvfp4 ? k / 16 : k / 32; }
// Nvfp4: the slot after an activation's scales (its max |x| as FP32 bits) and after a pass's
// ({alpha, 0, 1} of its gate or only half, then the up half's).
constexpr std::size_t kLtFp4Slot = 256;

std::size_t lt_round(std::size_t bytes) { return (bytes + kLtAlign - 1) / kLtAlign * kLtAlign; }

// Scale bytes of `rows` rows of `groups` groups: whole 128-row x 4-group tiles.
std::size_t lt_scale_bytes(int rows, int groups) {
    return std::size_t((rows + 127) / 128) * 128 * std::size_t((groups + 3) / 4) * 4;
}

// Rows per pass of a product of `rows` rows over `columns` columns, at most `most`: the GEMM
// (cuBLASLt's 128 x 128-tile block-scaled kernel, one CTA per SM) takes ceil(tiles / SMs) waves, so
// the pass is the multiple of 128 that runs the product in the fewest waves, weighed against the
// passes' own cost (an image launch and a GEMM ramp each, about 1024 / k of a wave), the largest on
// a tie. `stacked`: a pass is one GEMM over its gate and up rows. On the RTX 5060 Ti (36 SMs) at
// 1152 columns, 5120 rows of K = 6144 take 1536-row passes (3 + 3 + 3 + 1 waves) instead of
// 1792-row ones (4 + 4 + 3), 5.5% faster.
int lt_wave_pass(int rows, int columns, int k, int most, bool stacked) {
    const std::int64_t sms          = std::max(1, device_facts().sm_count);
    const std::int64_t column_tiles = (columns + 127) / 128;
    const auto waves                = [&](int count) {
        const std::int64_t row_tiles = (count + 127) / 128 * (stacked ? 2 : 1);
        return (row_tiles * column_tiles + sms - 1) / sms;
    };
    const double pass_cost = 1024.0 / double(std::max(k, 1024));
    int best               = most;
    double best_cost       = 0.0;
    for (int pass = most; pass >= 128; pass -= 128) {
        const int full = rows / pass, rest = rows % pass;
        const int passes = full + (rest != 0 ? 1 : 0);
        const double cost =
            double(full * waves(pass) + (rest != 0 ? waves(rest) : 0)) + pass_cost * passes;
        if (pass == most || cost < best_cost - 1e-9) {
            best      = pass;
            best_cost = cost;
        }
    }
    return best;
}

void check_lt(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("gguf cuBLASLt ") + what + ": status " +
                                 std::to_string(int(status)));
    }
}

cublasLtHandle_t lt_handle() {
    static std::array<cublasLtHandle_t, 16> handles{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    if (device < 0 || device >= int(handles.size())) {
        throw std::invalid_argument("gguf: device ordinal out of range");
    }
    // As blas_handle: the first use may be inside a CUDA graph capture.
    std::call_once(once[device], [&] {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
        const cublasStatus_t created = cublasLtCreate(&handles[device]);
        check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
        check_lt(created, "create");
    });
    return handles[device];
}

// D[m, n] (BF16, column-major, columns ld apart) = image[m, k] x activation[n, k]^T, plus D when
// accumulating, FP32 accumulation, no workspace. Mxfp8: e4m3 operands under VEC32_UE8M0 scales;
// Nvfp4 (`fp4_slots` {alpha, 0, 1} on device): e2m1 under VEC16_UE4M3 scales, times alpha.
struct LtKey {
    int device;
    int m;
    int n;
    int k;
    std::int64_t ld;
    bool accumulate;
    bool fp4;
    auto operator<=>(const LtKey&) const = default;
};

struct LtPlan {
    cublasLtMatmulDesc_t op  = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t d = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

void lt_gemm(int m, int n, int k, const std::uint8_t* image, const std::uint8_t* image_scales,
             const std::uint8_t* x, const std::uint8_t* x_scales, __nv_bfloat16* out,
             std::int64_t ld, bool accumulate, const float* fp4_slots, cudaStream_t stream) {
    const bool fp4 = fp4_slots != nullptr;
    static std::mutex mutex;
    static std::map<LtKey, LtPlan> plans;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    const cublasLtHandle_t handle = lt_handle();
    const std::lock_guard lock(mutex);
    const auto set_scales = [&](cublasLtMatmulDesc_t op) {
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &image_scales, sizeof image_scales),
                 "image scales");
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &x_scales, sizeof x_scales),
                 "activation scales");
    };
    auto [found, inserted] = plans.try_emplace(LtKey{device, m, n, k, ld, accumulate, fp4});
    LtPlan& plan           = found->second;
    if (inserted) {
        try {
            check_lt(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
            const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
            check_lt(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                                    sizeof ta),
                     "transa");
            check_lt(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                                    sizeof tb),
                     "transb");
            const cublasLtMatmulMatrixScale_t mode =
                fp4 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
                    : CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
            if (fp4) {
                const cublasLtPointerMode_t pointers = CUBLASLT_POINTER_MODE_DEVICE;
                check_lt(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                        &pointers, sizeof pointers),
                         "pointer mode");
            }
            check_lt(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                    &mode, sizeof mode),
                     "image scale mode");
            check_lt(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                    &mode, sizeof mode),
                     "activation scale mode");
            // The heuristic requires the scale pointers set (it finds no algorithm without them).
            set_scales(plan.op);
            const cudaDataType_t in_type = fp4 ? CUDA_R_4F_E2M1 : CUDA_R_8F_E4M3;
            check_lt(cublasLtMatrixLayoutCreate(&plan.a, in_type, k, m, k), "image layout");
            check_lt(cublasLtMatrixLayoutCreate(&plan.b, in_type, k, n, k), "x layout");
            check_lt(cublasLtMatrixLayoutCreate(&plan.d, CUDA_R_16BF, m, n, ld), "out layout");
            cublasLtMatmulPreference_t preference = nullptr;
            check_lt(cublasLtMatmulPreferenceCreate(&preference), "preference");
            const std::size_t workspace = 0;
            const cublasStatus_t set   = cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof workspace);
            cublasLtMatmulHeuristicResult_t results[8]{};
            int count                  = 0;
            const cublasStatus_t heuristic =
                set == CUBLAS_STATUS_SUCCESS
                    ? cublasLtMatmulAlgoGetHeuristic(handle, plan.op, plan.a, plan.b, plan.d,
                                                     plan.d, preference, 8, results, &count)
                    : set;
            cublasLtMatmulPreferenceDestroy(preference);
            check_lt(heuristic, "heuristic");
            if (count == 0) {
                throw std::runtime_error("gguf cuBLASLt: no block-scaled algorithm for m=" +
                                         std::to_string(m) + " n=" + std::to_string(n) +
                                         " k=" + std::to_string(k));
            }
            // The candidates share one kernel and differ in split-K and cluster shape. On the RTX
            // 5060 Ti (cuBLASLt 13.1) NVFP4's split-K ones ran at half the rate at K = 17408 or
            // 17408 rows and 3% slower at 2048 x 5120, the cluster shapes alike, and the
            // heuristic can rank split-K first: take the first without it.
            plan.algo = results[0].algo;
            for (int c = 0; c < count; ++c) {
                std::uint32_t splits = 0;
                std::size_t written  = 0;
                if (cublasLtMatmulAlgoConfigGetAttribute(&results[c].algo,
                                                         CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits,
                                                         sizeof splits, &written) ==
                        CUBLAS_STATUS_SUCCESS &&
                    written == sizeof splits && splits == 1) {
                    plan.algo = results[c].algo;
                    break;
                }
            }
        } catch (...) {
            if (plan.d != nullptr) { cublasLtMatrixLayoutDestroy(plan.d); }
            if (plan.b != nullptr) { cublasLtMatrixLayoutDestroy(plan.b); }
            if (plan.a != nullptr) { cublasLtMatrixLayoutDestroy(plan.a); }
            if (plan.op != nullptr) { cublasLtMatmulDescDestroy(plan.op); }
            plans.erase(found);
            throw;
        }
    } else {
        set_scales(plan.op);
    }
    const float alpha = 1.0f;
    const float beta  = accumulate ? 1.0f : 0.0f;
    const float* alpha_pointer = fp4 ? fp4_slots : &alpha;
    const float* beta_pointer  = fp4 ? fp4_slots + (accumulate ? 2 : 1) : &beta;
    check_lt(cublasLtMatmul(handle, plan.op, alpha_pointer, image, plan.a, x, plan.b, beta_pointer,
                            out, plan.d, out, plan.d, &plan.algo, nullptr, 0, stream),
             "matmul");
}

void image(GgmlType type, const ImageArgs& args, cudaStream_t stream) {
    switch (type) {
    case GgmlType::IQ3_XXS: return image_launch<GGML_TYPE_IQ3_XXS>(args, stream);
    case GgmlType::IQ3_S: return image_launch<GGML_TYPE_IQ3_S>(args, stream);
    case GgmlType::IQ2_XXS: return image_launch<GGML_TYPE_IQ2_XXS>(args, stream);
    case GgmlType::IQ4_XS: return image_launch<GGML_TYPE_IQ4_XS>(args, stream);
    case GgmlType::IQ2_XS: return image_launch<GGML_TYPE_IQ2_XS>(args, stream);
    case GgmlType::IQ2_S: return image_launch<GGML_TYPE_IQ2_S>(args, stream);
    case GgmlType::Q2_K: return image_launch<GGML_TYPE_Q2_K>(args, stream);
    case GgmlType::Q4_K: return image_launch<GGML_TYPE_Q4_K>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf MXFP8 image: no MXFP8 form for this block type");
}

// The NVFP4 image of `rows` rows under the weight's global, taking the rows' choice table where
// given; pass slots {alpha, 0, 1} for the activation whose max |x| is at x_amax.
void image_fp4(GgmlType type, const std::uint8_t* weight, std::int64_t row_bytes, int rows, int k,
               float global, const std::uint8_t* choices, std::uint8_t* image,
               std::uint8_t* scales, const unsigned* x_amax, float* slots, cudaStream_t stream) {
    const Fp4ImageArgs args{weight, row_bytes, rows, k,     image,
                            scales, global,    x_amax, slots, choices};
    switch (type) {
    case GgmlType::IQ3_XXS: return image_fp4_launch<GGML_TYPE_IQ3_XXS>(args, stream);
    case GgmlType::IQ3_S: return image_fp4_launch<GGML_TYPE_IQ3_S>(args, stream);
    case GgmlType::IQ2_XXS: return image_fp4_launch<GGML_TYPE_IQ2_XXS>(args, stream);
    case GgmlType::IQ4_XS: return image_fp4_launch<GGML_TYPE_IQ4_XS>(args, stream);
    case GgmlType::IQ2_XS: return image_fp4_launch<GGML_TYPE_IQ2_XS>(args, stream);
    case GgmlType::IQ2_S: return image_fp4_launch<GGML_TYPE_IQ2_S>(args, stream);
    case GgmlType::Q2_K: return image_fp4_launch<GGML_TYPE_Q2_K>(args, stream);
    case GgmlType::Q4_K: return image_fp4_launch<GGML_TYPE_Q4_K>(args, stream);
    case GgmlType::IQ1_S: return image_fp4_launch<GGML_TYPE_IQ1_S>(args, stream);
    case GgmlType::IQ1_M: return image_fp4_launch<GGML_TYPE_IQ1_M>(args, stream);
    case GgmlType::Q6_K: return image_fp4_launch<GGML_TYPE_Q6_K>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf NVFP4 image: no NVFP4 form for this block type");
}

bool has_fp4_image(GgmlType type) {
    switch (type) {
    case GgmlType::IQ3_XXS:
    case GgmlType::IQ3_S:
    case GgmlType::IQ2_XXS:
    case GgmlType::IQ4_XS:
    case GgmlType::IQ2_XS:
    case GgmlType::IQ2_S:
    case GgmlType::Q2_K:
    case GgmlType::Q4_K:
    case GgmlType::IQ1_S:
    case GgmlType::IQ1_M:
    case GgmlType::Q6_K: return true;
    default: return false;
    }
}

} // namespace
} // namespace detail

bool has_lt_product(GgmlType type, int k, LtForm form) {
    if (!kTileMxfp8Built) { return false; }
    if (form == LtForm::Mxfp8) {
        if (!has_tile_kernel(type, k, true)) { return false; }
    } else {
        // The activation quantizer stages a BF16 row of k values in shared memory.
        const detail::DeviceFacts& facts = detail::device_facts();
        if (!detail::has_fp4_image(type) || facts.cc < 1200 || facts.cc >= 1300 || k <= 0 ||
            k % 256 != 0 || std::size_t(k) * sizeof(__nv_bfloat16) > facts.shared_per_block_optin) {
            return false;
        }
    }
    static const bool block_scaled = cublasLtGetVersion() >= 120800;
    return block_scaled;
}

bool lt_takes_output(const VectorOutput& out) {
    return out.bf16 != nullptr && out.f32 == nullptr && out.epilogue != Epilogue::GateProduct &&
           reinterpret_cast<std::uintptr_t>(out.bf16) % 16 == 0 && out.column_stride % 8 == 0;
}

std::size_t lt_activation_scales_offset(int k, int columns, LtForm form) {
    return detail::lt_round(detail::lt_row_bytes(k, form) * columns);
}

std::size_t lt_activation_bytes(int k, int columns, LtForm form) {
    const std::size_t scales =
        detail::lt_round(detail::lt_scale_bytes(columns, detail::lt_groups(k, form)));
    return lt_activation_scales_offset(k, columns, form) + scales +
           (form == LtForm::Nvfp4 ? detail::kLtFp4Slot + scales : 0);
}

namespace {
// An Nvfp4 activation's max |x| slot.
const unsigned* lt_fp4_activation_slot(const void* activation, int k, int columns) {
    return reinterpret_cast<const unsigned*>(
        static_cast<const std::uint8_t*>(activation) +
        lt_activation_scales_offset(k, columns, LtForm::Nvfp4) +
        detail::lt_round(detail::lt_scale_bytes(columns, detail::lt_groups(k, LtForm::Nvfp4))));
}

#if defined(NINFER_SM120_FP8)
LtFp4Parts fp4_parts(void* activation, int k, int columns) {
    auto* base = static_cast<std::uint8_t*>(activation);
    auto* slot = const_cast<unsigned*>(lt_fp4_activation_slot(activation, k, columns));
    return {base, base + lt_activation_scales_offset(k, columns, LtForm::Nvfp4),
            reinterpret_cast<std::int8_t*>(slot) + detail::kLtFp4Slot, slot};
}

// The scale pass after the encode of x [columns][k] into `parts` (see lt_fp4_scale_kernel).
void fp4_scale_pass(const __nv_bfloat16* x, const std::int32_t* input_columns,
                    const LtFp4Parts& parts, int k, int columns, cudaStream_t stream) {
    const auto scale_bytes = std::int64_t(detail::lt_scale_bytes(columns, k / 16));
    detail::lt_fp4_scale_kernel<<<int((scale_bytes / 4 + 255) / 256), 256, 0, stream>>>(
        x, input_columns, parts.codes, parts.scales, parts.shifts, k, columns, scale_bytes,
        parts.slot);
}
#endif
} // namespace

void quantize_lt_activation(const __nv_bfloat16* x, int k, int columns,
                            const std::int32_t* input_columns, LtForm form, void* out,
                            cudaStream_t stream) {
    if (k <= 0 || k % 256 != 0 || columns <= 0) {
        throw std::invalid_argument("gguf two-pass activation: K must be whole 256-value steps");
    }
#if defined(NINFER_SM120_FP8)
    auto* base = static_cast<std::uint8_t*>(out);
    if (form == LtForm::Nvfp4) {
        // One pass encodes every group at its own power of two and finds max |x|; the scale pass
        // then takes the scales to the global (see lt_fp4_scale_kernel).
        const LtFp4Parts parts = fp4_parts(out, k, columns);
        detail::check(cudaMemsetAsync(parts.slot, 0, sizeof(unsigned), stream), "NVFP4 amax reset");
        const std::int64_t total = std::int64_t(columns) * (k / 16);
        const auto blocks = int((total + detail::kFp4EncodeThreads - 1) / detail::kFp4EncodeThreads);
        detail::lt_fp4_encode_kernel<<<blocks, detail::kFp4EncodeThreads, 0, stream>>>(
            x, input_columns, parts.codes, parts.scales, parts.shifts, k, columns, parts.slot);
        fp4_scale_pass(x, input_columns, parts, k, columns, stream);
        detail::check(cudaGetLastError(), "NVFP4 activation launch");
        return;
    }
    const dim3 blocks(detail::tile_columns_pad(columns),
                      (k + 4 * detail::kTileQuantizeThreads - 1) /
                          (4 * detail::kTileQuantizeThreads),
                      1);
    detail::quantize_fp8_kernel<true><<<blocks, detail::kTileQuantizeThreads, 0, stream>>>(
        x, input_columns, base, base + lt_activation_scales_offset(k, columns, form), k, columns,
        detail::tile_columns_pad(columns));
    detail::check(cudaGetLastError(), "MXFP8 activation launch");
#else
    (void)x;
    (void)input_columns;
    (void)form;
    (void)out;
    (void)stream;
    throw std::invalid_argument("gguf two-pass activation: this build has no block-scaled form");
#endif
}

LtFp4Parts begin_lt_fp4_activation(void* activation, int k, int columns, cudaStream_t stream) {
    if (activation == nullptr || k <= 0 || k % 256 != 0 || columns <= 0) {
        throw std::invalid_argument("gguf NVFP4 activation: K must be whole 256-value steps");
    }
#if defined(NINFER_SM120_FP8)
    const LtFp4Parts parts = fp4_parts(activation, k, columns);
    detail::check(cudaMemsetAsync(parts.slot, 0, sizeof(unsigned), stream), "NVFP4 amax reset");
    return parts;
#else
    (void)stream;
    throw std::invalid_argument("gguf NVFP4 activation: this build has no NVFP4 form");
#endif
}

void finish_lt_fp4_activation(const __nv_bfloat16* x, const std::int32_t* input_columns, int k,
                              int columns, const LtFp4Parts& parts, cudaStream_t stream) {
#if defined(NINFER_SM120_FP8)
    fp4_scale_pass(x, input_columns, parts, k, columns, stream);
    detail::check(cudaGetLastError(), "NVFP4 activation scale pass");
#else
    (void)x;
    (void)input_columns;
    (void)k;
    (void)columns;
    (void)parts;
    (void)stream;
    throw std::invalid_argument("gguf NVFP4 activation: this build has no NVFP4 form");
#endif
}

std::size_t lt_scratch_bytes(int k, int rows, int columns, bool gated, LtForm form) {
    const int image_rows = gated ? 2 * rows : rows;
    std::size_t bytes =
        detail::lt_round(std::size_t(image_rows) * detail::lt_row_bytes(k, form)) +
        detail::lt_round(detail::lt_scale_bytes(image_rows, detail::lt_groups(k, form)));
    if (gated) { bytes += detail::lt_round(std::size_t(image_rows) * columns * 2); }
    if (form == LtForm::Nvfp4) { bytes += detail::kLtFp4Slot; }
    return bytes;
}

namespace {

// The table an Nvfp4 product of weight global `global` reads (Fp4Choices), or null.
const std::uint8_t* product_choices(const Fp4Choices& choices, float global) {
    if (choices.table == nullptr || reinterpret_cast<std::uintptr_t>(choices.table) % 128 != 0 ||
        !(choices.weight_amax > 0.0f) || detail::lt_fp4_global(choices.weight_amax) != global) {
        return nullptr;
    }
    return choices.table;
}

} // namespace

std::size_t fp4_choice_bytes(int rows, int k) {
    const std::size_t bytes = (std::size_t(rows) * std::size_t(k / 32) + 1) / 2;
    return (bytes + 127) / 128 * 128;
}

const std::uint8_t* fp4_choice_rows(const std::uint8_t* table, int first, int k) {
    const std::size_t slices = std::size_t(first) * std::size_t(k / 32);
    return table == nullptr || slices % 2 != 0 ? nullptr : table + slices / 2;
}

void fp4_choices(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                 float weight_amax, std::uint8_t* table, cudaStream_t stream) {
    if (!detail::has_fp4_image(type) || rows <= 0 || k <= 0 || k % 256 != 0 ||
        !(weight_amax > 0.0f) || table == nullptr ||
        reinterpret_cast<std::uintptr_t>(table) % 128 != 0) {
        throw std::invalid_argument("gguf NVFP4 choice table: invalid matrix or table");
    }
    const detail::Fp4ImageArgs args{static_cast<const std::uint8_t*>(weight),
                                    row_bytes,
                                    rows,
                                    k,
                                    nullptr,
                                    nullptr,
                                    detail::lt_fp4_global(weight_amax),
                                    nullptr,
                                    nullptr};
    switch (type) {
    case GgmlType::IQ3_XXS: return detail::fp4_choices_launch<GGML_TYPE_IQ3_XXS>(args, table, stream);
    case GgmlType::IQ3_S: return detail::fp4_choices_launch<GGML_TYPE_IQ3_S>(args, table, stream);
    case GgmlType::IQ2_XXS: return detail::fp4_choices_launch<GGML_TYPE_IQ2_XXS>(args, table, stream);
    case GgmlType::IQ4_XS: return detail::fp4_choices_launch<GGML_TYPE_IQ4_XS>(args, table, stream);
    case GgmlType::IQ2_XS: return detail::fp4_choices_launch<GGML_TYPE_IQ2_XS>(args, table, stream);
    case GgmlType::IQ2_S: return detail::fp4_choices_launch<GGML_TYPE_IQ2_S>(args, table, stream);
    case GgmlType::Q2_K: return detail::fp4_choices_launch<GGML_TYPE_Q2_K>(args, table, stream);
    case GgmlType::Q4_K: return detail::fp4_choices_launch<GGML_TYPE_Q4_K>(args, table, stream);
    case GgmlType::IQ1_S: return detail::fp4_choices_launch<GGML_TYPE_IQ1_S>(args, table, stream);
    case GgmlType::IQ1_M: return detail::fp4_choices_launch<GGML_TYPE_IQ1_M>(args, table, stream);
    case GgmlType::Q6_K: return detail::fp4_choices_launch<GGML_TYPE_Q6_K>(args, table, stream);
    default: break;
    }
}

int lt_rows_per_pass(int k, int columns, bool gated, LtForm form, std::size_t scratch_bytes) {
    int rows = 0;
    while (rows < (1 << 20) &&
           lt_scratch_bytes(k, rows + 128, columns, gated, form) <= scratch_bytes) {
        rows += 128;
    }
    return rows;
}

std::size_t lt_planned_scratch_bytes(int k, int rows, int columns, bool gated, LtForm form) {
    // Rows of an MXFP8 slab of kLtImageBytes, in either form (an Nvfp4 slab holds half the bytes).
    const int image_rows = std::max(128, int(kLtImageBytes / std::size_t(k)) / 128 * 128);
    const int pass       = std::min(gated ? std::max(128, image_rows / 256 * 128) : image_rows,
                                    (rows + 127) / 128 * 128);
    return lt_scratch_bytes(k, pass, columns, gated, form);
}

void lt_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                LtForm form, float weight_amax, const Fp4Choices& choices,
                const void* activation, int columns, const VectorOutput& out, void* scratch,
                std::size_t scratch_bytes, cudaStream_t stream) {
    const bool fp4 = form == LtForm::Nvfp4;
    if (!lt_takes_output(out) || rows <= 0 || columns <= 0 || !has_lt_product(type, k, form) ||
        (fp4 && !(weight_amax > 0.0f))) {
        throw std::invalid_argument("gguf two-pass product: invalid product");
    }
    const int most = lt_rows_per_pass(k, columns, false, form, scratch_bytes);
    if (most <= 0) { throw std::invalid_argument("gguf two-pass product: scratch too small"); }
    const int pass       = detail::lt_wave_pass(rows, columns, k, most, false);
    auto* image          = static_cast<std::uint8_t*>(scratch);
    auto* image_scales =
        image + detail::lt_round(std::size_t(pass) * detail::lt_row_bytes(k, form));
    auto* slots          = reinterpret_cast<float*>(
        image_scales + detail::lt_round(detail::lt_scale_bytes(pass, detail::lt_groups(k, form))));
    const auto* x        = static_cast<const std::uint8_t*>(activation);
    const auto* x_scales = x + lt_activation_scales_offset(k, columns, form);
    const float global   = fp4 ? detail::lt_fp4_global(weight_amax) : 1.0f;
    const std::uint8_t* table = fp4 ? product_choices(choices, global) : nullptr;
    for (int first = 0; first < rows; first += pass) {
        const int count = std::min(pass, rows - first);
        const auto* rows_weight = static_cast<const std::uint8_t*>(weight) + first * row_bytes;
        if (fp4) {
            detail::image_fp4(type, rows_weight, row_bytes, count, k, global,
                              fp4_choice_rows(table, first, k), image, image_scales,
                              lt_fp4_activation_slot(activation, k, columns), slots, stream);
        } else {
            detail::image(type, {rows_weight, row_bytes, count, k, image, image_scales}, stream);
        }
        detail::lt_gemm(count, columns, k, image, image_scales, x, x_scales, out.bf16 + first,
                        out.column_stride, out.epilogue == Epilogue::Accumulate,
                        fp4 ? slots : nullptr, stream);
    }
}

void lt_gated_product(GgmlType gate_type, const void* gate, std::int64_t gate_row_bytes,
                      GgmlType up_type, const void* up, std::int64_t up_row_bytes, int rows, int k,
                      LtForm form, float weight_amax, const Fp4Choices& gate_choices,
                      const Fp4Choices& up_choices, const void* activation,
                      const void* up_activation, int columns, __nv_bfloat16* out,
                      std::int64_t out_column_stride, GateActivation gate_activation,
                      void* out_activation, void* scratch, std::size_t scratch_bytes,
                      cudaStream_t stream) {
    const bool fp4 = form == LtForm::Nvfp4;
    if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 16 != 0 ||
        out_column_stride % 8 != 0 || rows <= 0 || columns <= 0 ||
        !has_lt_product(gate_type, k, form) || !has_lt_product(up_type, k, form) ||
        (fp4 && !(weight_amax > 0.0f)) ||
        (out_activation != nullptr && (!fp4 || out_column_stride != rows || rows % 256 != 0))) {
        throw std::invalid_argument("gguf two-pass gated product: invalid product");
    }
#if defined(NINFER_SM120_FP8)
    LtFp4Parts encoded{};
    if (out_activation != nullptr) {
        encoded = fp4_parts(out_activation, rows, columns);
        detail::check(cudaMemsetAsync(encoded.slot, 0, sizeof(unsigned), stream),
                      "NVFP4 amax reset");
    }
#else
    if (out_activation != nullptr) {
        throw std::invalid_argument("gguf two-pass gated product: this build has no NVFP4 form");
    }
#endif
    const int most = lt_rows_per_pass(k, columns, true, form, scratch_bytes);
    if (most <= 0) {
        throw std::invalid_argument("gguf two-pass gated product: scratch too small");
    }
    const bool stacked = up_activation == nullptr;
    const int pass     = detail::lt_wave_pass(rows, columns, k, most, stacked);
    // A pass of `count` rows: gate rows at image rows [0, count), up rows from `offset` (whole
    // 128-row scale tiles), products in a BF16 plane of 2 * offset rows a column. Nvfp4: gate and
    // up share the pair's global, so over one activation their stacked GEMM takes one alpha.
    const std::size_t row_image = detail::lt_row_bytes(k, form);
    const int groups            = detail::lt_groups(k, form);
    auto* image                 = static_cast<std::uint8_t*>(scratch);
    auto* image_scales          = image + detail::lt_round(std::size_t(2) * pass * row_image);
    auto* plane                 = reinterpret_cast<__nv_bfloat16*>(
        image_scales + detail::lt_round(detail::lt_scale_bytes(2 * pass, groups)));
    auto* gate_slots = reinterpret_cast<float*>(
        reinterpret_cast<std::uint8_t*>(plane) +
        detail::lt_round(std::size_t(2) * pass * columns * 2));
    auto* up_slots   = gate_slots + 4;
    const auto* x        = static_cast<const std::uint8_t*>(activation);
    const auto* x_scales = x + lt_activation_scales_offset(k, columns, form);
    const auto* ux       = stacked ? x : static_cast<const std::uint8_t*>(up_activation);
    const float global   = fp4 ? detail::lt_fp4_global(weight_amax) : 1.0f;
    const std::uint8_t* gate_table = fp4 ? product_choices(gate_choices, global) : nullptr;
    const std::uint8_t* up_table   = fp4 ? product_choices(up_choices, global) : nullptr;
    for (int first = 0; first < rows; first += pass) {
        const int count  = std::min(pass, rows - first);
        const int offset = (count + 127) / 128 * 128;
        const auto* gate_weight = static_cast<const std::uint8_t*>(gate) + first * gate_row_bytes;
        const auto* up_weight   = static_cast<const std::uint8_t*>(up) + first * up_row_bytes;
        std::uint8_t* up_image  = image + std::size_t(offset) * row_image;
        std::uint8_t* up_scales = image_scales + detail::lt_scale_bytes(offset, groups);
        if (fp4) {
            detail::image_fp4(gate_type, gate_weight, gate_row_bytes, count, k, global,
                              fp4_choice_rows(gate_table, first, k), image, image_scales,
                              lt_fp4_activation_slot(activation, k, columns), gate_slots, stream);
            detail::image_fp4(up_type, up_weight, up_row_bytes, count, k, global,
                              fp4_choice_rows(up_table, first, k), up_image, up_scales,
                              lt_fp4_activation_slot(ux, k, columns), up_slots, stream);
        } else {
            detail::image(gate_type, {gate_weight, gate_row_bytes, count, k, image, image_scales},
                          stream);
            detail::image(up_type, {up_weight, up_row_bytes, count, k, up_image, up_scales},
                          stream);
        }
        if (stacked) {
            detail::lt_gemm(offset + count, columns, k, image, image_scales, x, x_scales, plane,
                            2 * offset, false, fp4 ? gate_slots : nullptr, stream);
        } else {
            detail::lt_gemm(count, columns, k, image, image_scales, x, x_scales, plane, 2 * offset,
                            false, fp4 ? gate_slots : nullptr, stream);
            detail::lt_gemm(count, columns, k, up_image, up_scales, ux,
                            ux + lt_activation_scales_offset(k, columns, form), plane + offset,
                            2 * offset, false, fp4 ? up_slots : nullptr, stream);
        }
        // Passes start at whole 128-row tiles, so out + first keeps out's 16-byte alignment.
#if defined(NINFER_SM120_FP8)
        if (out_activation != nullptr) {
            const std::int64_t total = std::int64_t(columns) * (count / 16);
            detail::lt_gate_fp4_kernel<<<int((total + detail::kFp4EncodeThreads - 1) /
                                             detail::kFp4EncodeThreads),
                                         detail::kFp4EncodeThreads, 0, stream>>>(
                plane, 2 * offset, offset, count, columns, first, rows, out,
                static_cast<int>(gate_activation), encoded.codes, encoded.scales, encoded.shifts,
                encoded.slot);
            detail::check(cudaGetLastError(), "two-pass gate launch");
            continue;
        }
#endif
        detail::lt_gate_kernel<<<dim3((count + 1023) / 1024, columns, 1), 128, 0, stream>>>(
            plane, 2 * offset, offset, count, out + first, out_column_stride,
            static_cast<int>(gate_activation));
        detail::check(cudaGetLastError(), "two-pass gate launch");
    }
#if defined(NINFER_SM120_FP8)
    if (out_activation != nullptr) {
        fp4_scale_pass(out, nullptr, encoded, rows, columns, stream);
        detail::check(cudaGetLastError(), "NVFP4 activation launch");
    }
#endif
}

void matrix_amax(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                 unsigned* amax, cudaStream_t stream) {
    if (!detail::has_fp4_image(type) || rows <= 0 || k <= 0 || k % 256 != 0 || amax == nullptr) {
        throw std::invalid_argument("gguf matrix amax: invalid matrix");
    }
    detail::check(cudaMemsetAsync(amax, 0, sizeof(unsigned), stream), "amax reset");
    const detail::AmaxArgs args{static_cast<const std::uint8_t*>(weight), row_bytes, rows, k, amax};
    switch (type) {
    case GgmlType::IQ3_XXS: return detail::amax_launch<GGML_TYPE_IQ3_XXS>(args, stream);
    case GgmlType::IQ3_S: return detail::amax_launch<GGML_TYPE_IQ3_S>(args, stream);
    case GgmlType::IQ2_XXS: return detail::amax_launch<GGML_TYPE_IQ2_XXS>(args, stream);
    case GgmlType::IQ4_XS: return detail::amax_launch<GGML_TYPE_IQ4_XS>(args, stream);
    case GgmlType::IQ2_XS: return detail::amax_launch<GGML_TYPE_IQ2_XS>(args, stream);
    case GgmlType::IQ2_S: return detail::amax_launch<GGML_TYPE_IQ2_S>(args, stream);
    case GgmlType::Q2_K: return detail::amax_launch<GGML_TYPE_Q2_K>(args, stream);
    case GgmlType::Q4_K: return detail::amax_launch<GGML_TYPE_Q4_K>(args, stream);
    case GgmlType::IQ1_S: return detail::amax_launch<GGML_TYPE_IQ1_S>(args, stream);
    case GgmlType::IQ1_M: return detail::amax_launch<GGML_TYPE_IQ1_M>(args, stream);
    case GgmlType::Q6_K: return detail::amax_launch<GGML_TYPE_Q6_K>(args, stream);
    default: break;
    }
}

void store_plane(const float* in, std::int64_t in_column_stride, int rows, int columns,
                 const VectorOutput& out, cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr) || rows <= 0 ||
        columns <= 0) {
        throw std::invalid_argument("gguf store plane: invalid output");
    }
    const detail::VecStore store{out.bf16,     out.f32,  out.column_stride,
                                 out.epilogue, out.gate, out.gate_column_stride,
                                 static_cast<int>(out.activation)};
    detail::store_plane_kernel<<<dim3((rows + 255) / 256, columns, 1), 256, 0, stream>>>(
        in, in_column_stride, rows, store);
    detail::check(cudaGetLastError(), "store plane launch");
}

void gather_columns(const __nv_bfloat16* x, int k, int columns, const std::int32_t* input_columns,
                    __nv_bfloat16* out, cudaStream_t stream) {
    detail::gather_columns_kernel<<<dim3((k + 255) / 256, columns, 1), 256, 0, stream>>>(
        x, k, input_columns, out);
    detail::check(cudaGetLastError(), "gather columns launch");
}

int dequantized_rows_per_pass(int k, std::size_t scratch_bytes) {
    const std::size_t row = std::size_t(k) * sizeof(__nv_bfloat16);
    return static_cast<int>(std::min<std::size_t>(scratch_bytes / row, 1U << 20) / 128 * 128);
}

void dequantized_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                         const __nv_bfloat16* x, int columns, float* out,
                         std::int64_t out_column_stride, void* scratch, std::size_t scratch_bytes,
                         cudaStream_t stream) {
    const int pass = dequantized_rows_per_pass(k, scratch_bytes);
    if (pass <= 0) { throw std::invalid_argument("gguf dequantized product: scratch too small"); }
    cublasHandle_t blas = detail::blas_handle();
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("gguf: cublasSetStream failed");
    }
    auto* rows_bf16    = static_cast<__nv_bfloat16*>(scratch);
    const float alpha  = 1.0f;
    const float beta   = 0.0f;
    for (int first = 0; first < rows; first += pass) {
        const int count = std::min(pass, rows - first);
        dequantize_rows(type, static_cast<const char*>(weight) + first * row_bytes, row_bytes, k,
                        nullptr, count, rows_bf16, k, stream);
        const cublasStatus_t status = cublasGemmEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, count, columns, k, &alpha, rows_bf16, CUDA_R_16BF, k,
            x, CUDA_R_16BF, k, &beta, out + first, CUDA_R_32F,
            static_cast<int>(out_column_stride), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("gguf dequantized product: cublasGemmEx failed");
        }
    }
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __nv_bfloat16* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, float* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __half* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

} // namespace ninfer::ops::gguf
