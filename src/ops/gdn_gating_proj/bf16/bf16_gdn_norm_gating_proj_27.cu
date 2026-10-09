#include "core/weight.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/hadamard_transform.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {
// Each head computes both control dots and the complete norm. RMS scaling can be
// applied after the dots; h is independently rounded from the full normalized input.
// Rotate writes h already in the rotated basis of the input projection that consumes it: the head
// CTAs of the first D / 1024 heads each transform one 1024-block of every token in their tile from
// the same BF16 values the unrotated route stores.
// The dots and the norm sum in the order of OrderThreads threads: each one accumulates the 8-element
// chunks tid, tid + OrderThreads, ... in turn, each warp of them reduces its lanes, and the warp sums
// add up in warp order. A CTA of Threads physical threads runs OrderThreads / Threads of those
// threads each, so the order, and every result bit, is independent of the CTA size.
template <int Tile, int Threads, int OrderThreads, bool Rotate>
__global__ __launch_bounds__(Threads) void gdn_norm_gating_27_simt(
    const __nv_bfloat16* x, const __nv_bfloat16* nw, const __nv_bfloat16* aw,
    const __nv_bfloat16* bw, const float* alog, const float* bias, const __nv_bfloat16* signs,
    __nv_bfloat16* h, float* g, float* beta, int tokens, float eps) {
    pdl::enter();
    constexpr int D = 5120, H = 48, Warps = Threads / 32;
    constexpr int Passes = OrderThreads / Threads, OrderWarps = OrderThreads / 32;
    static_assert(OrderThreads % Threads == 0);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, head = blockIdx.x,
              first = blockIdx.y * Tile;
    float aa[Passes][Tile]{}, bb[Passes][Tile]{}, ss[Passes][Tile]{};
#pragma unroll
    for (int o = 0; o < Passes; ++o) {
        for (int base = (tid + o * Threads) * 8; base < D; base += OrderThreads * 8) {
            const int4 av = load_vec<int4>(aw + head * D + base),
                       bv = load_vec<int4>(bw + head * D + base), nv = load_vec<int4>(nw + base);
            const auto* ap = reinterpret_cast<const unsigned*>(&av);
            const auto* bp = reinterpret_cast<const unsigned*>(&bv);
            const auto* np = reinterpret_cast<const unsigned*>(&nv);
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                if (first + t >= tokens) continue;
                const int4 xv  = load_vec<int4>(x + std::int64_t(first + t) * D + base);
                const auto* xp = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 a = bf16x2_bits_to_float2(ap[p]), b = bf16x2_bits_to_float2(bp[p]);
                    const float2 n = bf16x2_bits_to_float2(np[p]), v = bf16x2_bits_to_float2(xp[p]);
                    const float z0 = v.x * (1.0f + n.x), z1 = v.y * (1.0f + n.y);
                    aa[o][t] = fmaf(a.x, z0, aa[o][t]);
                    aa[o][t] = fmaf(a.y, z1, aa[o][t]);
                    bb[o][t] = fmaf(b.x, z0, bb[o][t]);
                    bb[o][t] = fmaf(b.y, z1, bb[o][t]);
                    ss[o][t] = fmaf(v.x, v.x, ss[o][t]);
                    ss[o][t] = fmaf(v.y, v.y, ss[o][t]);
                }
            }
        }
    }
    __shared__ float partial[3][Tile][OrderWarps], inverse[Tile];
#pragma unroll
    for (int o = 0; o < Passes; ++o) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            const float a = warp_reduce_sum(aa[o][t]);
            const float b = warp_reduce_sum(bb[o][t]);
            const float s = warp_reduce_sum(ss[o][t]);
            if (lane == 0) {
                partial[0][t][o * Warps + warp] = a;
                partial[1][t][o * Warps + warp] = b;
                partial[2][t][o * Warps + warp] = s;
            }
        }
    }
    __syncthreads();
    if (tid < Tile) {
        float a = 0, b = 0, s = 0;
#pragma unroll
        for (int w = 0; w < OrderWarps; ++w) {
            a += partial[0][tid][w];
            b += partial[1][tid][w];
            s += partial[2][tid][w];
        }
        const float inv = rsqrtf(s / D + eps);
        inverse[tid]    = inv;
        if (first + tid < tokens) {
            const auto i = std::int64_t(first + tid) * H + head;
            g[i]         = -expf(alog[head]) * softplus(a * inv + bias[head]);
            beta[i]      = sigmoid(b * inv);
        }
    }
    __syncthreads();
    if constexpr (Rotate) {
        // Four warps per (token, 1024-block): the tile's tokens in rounds of Warps / 4, every warp
        // entering the quarter transform's barriers in each round.
        constexpr int kGroups = Warps / kHadamardQuarterWarps;
        __shared__ HadamardQuarterShared transform[kGroups];
        const int group   = warp / kHadamardQuarterWarps;
        const int quarter = warp % kHadamardQuarterWarps;
        if (head >= D / kHadamardTransformBlock) { return; }
        const int block_base = head * kHadamardTransformBlock;
#pragma unroll
        for (int round = 0; round < (Tile + kGroups - 1) / kGroups; ++round) {
            const int t = round * kGroups + group;
            if (t < Tile && first + t < tokens) {
                const std::int64_t row = std::int64_t(first + t) * D;
                const int offset       = block_base + quarter * kWarpSize * 8 + lane * 8;
                float xv[8];
                float nv[8];
                hadamard_unpack8(load_vec<uint4>(x + row + offset), xv);
                hadamard_unpack8(load_vec<uint4>(nw + offset), nv);
                float v[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    v[j] = __bfloat162float(__float2bfloat16_rn(xv[j] * inverse[t] * (1 + nv[j])));
                }
                hadamard_quarter_forward_store(v, signs + block_base, h + row + block_base,
                                               transform[group], quarter, lane,
                                               [] { __syncthreads(); });
            } else {
                __syncthreads();
                __syncthreads();
            }
        }
        return;
    }
    // Disjoint, pair-aligned h slices across the 48 heads avoid a separate norm kernel.
    constexpr int PairsPerHead = (D / 2 + H - 1) / H;
    const int pair             = head * PairsPerHead + tid;
    if (tid < PairsPerHead && pair < D / 2) {
        const float2 n = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(nw)[pair]);
#pragma unroll
        for (int t = 0; t < Tile; ++t)
            if (first + t < tokens) {
                const auto i   = std::int64_t(first + t) * (D / 2) + pair;
                const float2 v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x)[i]);
                reinterpret_cast<__nv_bfloat162*>(h)[i] = __floats2bfloat162_rn(
                    v.x * inverse[t] * (1 + n.x), v.y * inverse[t] * (1 + n.y));
            }
    }
}
} // namespace

void bf16_gdn_norm_gating_proj_27_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                         Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                         const Tensor& alog, const Tensor& bias, Tensor& g,
                                         Tensor& beta, const Tensor* signs, cudaStream_t stream) {
    const auto launch = [&]<int T, int Threads, int Order>() {
        const auto run = [&]<bool Rotate>() {
            CUDA_CHECK(pdl::launch_consumer(
                {dim3(48, (x.ne[1] + T - 1) / T), dim3(Threads), 0, stream},
                gdn_norm_gating_27_simt<T, Threads, Order, Rotate>,
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(norm_weight.data),
                static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(alog.data), static_cast<const float*>(bias.data),
                Rotate ? static_cast<const __nv_bfloat16*>(signs->data) : nullptr,
                static_cast<__nv_bfloat16*>(h.data), static_cast<float*>(g.data),
                static_cast<float*>(beta.data), x.ne[1], eps));
        };
        if (signs != nullptr) {
            run.template operator()<true>();
        } else {
            run.template operator()<false>();
        }
    };
    // Through 14 tokens the sums keep the order of 512 threads, beyond it that of 256. Measured on the
    // RTX 5060 Ti (36 SMs) with the weights out of L2, as a decode graph meets them: 256 physical
    // threads run one- and two-token tiles 0-2.5 us faster than 512 (T=8: 15.4 -> 13.0 us), and
    // one-token tiles win through 4 tokens.
    const int tokens = x.ne[1];
    if (tokens <= 4)
        launch.template operator()<1, 256, 512>();
    else if (tokens <= 14)
        launch.template operator()<2, 256, 512>();
    else if (tokens <= 28)
        launch.template operator()<2, 256, 256>();
    else
        launch.template operator()<4, 256, 256>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
