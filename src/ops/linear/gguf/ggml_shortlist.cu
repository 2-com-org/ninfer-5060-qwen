// shortlist_argmax (ggml_bridge.h): the coarse logits' top candidates per column, exact rescoring of
// their rows by the vector kernel's arithmetic (ggml_bridge_vec.cuh), and the argmax.
//
// Selection runs in two exact stages over 16-bit order keys of the BF16 logits: each tile of
// kShortlistTileRows rows keeps its `candidates` largest (a tie to the lower row), which contain
// every global winner of the tile; one block per column then selects the same number among the
// tiles' survivors, kept in row order. A block selects by bisecting the key of the last winner (16
// counting rounds), then takes the keys above it and the earliest of the ties.
#include "ggml_bridge_internal.cuh"

#include <cub/block/block_scan.cuh>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::gguf {
namespace detail {
namespace {

constexpr int kTileThreads = 256;
constexpr int kItems       = 8;
static_assert(kTileThreads * kItems == kShortlistTileRows);
static_assert(1024 * kItems == kMaxShortlistMerged);

// take[j]: item j of this thread is among the `want` largest keys of the block's items (Items
// consecutive positions a thread, threads in order), a tie to the earlier position; slot[j] its
// rank in position order among the taken items. Keys are 16-bit; the block holds at least `want`
// items.
template <int Threads>
__device__ void select_largest(const std::uint32_t (&key)[kItems], int want, bool (&take)[kItems],
                               int (&slot)[kItems]) {
    using Scan = cub::BlockScan<int, Threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int partial[2][Threads / 32];
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    int buffer     = 0;
    // Items with a key of at least `bound` over the block. Alternate buffers let one barrier serve:
    // a buffer is rewritten only after the next round's barrier, when every read of it is done.
    const auto count_from = [&](std::uint32_t bound) {
        int n = 0;
#pragma unroll
        for (int j = 0; j < kItems; ++j) { n += key[j] >= bound ? 1 : 0; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { n += __shfl_xor_sync(0xFFFFFFFFu, n, o); }
        if (lane == 0) { partial[buffer][warp] = n; }
        __syncthreads();
        int total = 0;
#pragma unroll
        for (int w = 0; w < Threads / 32; ++w) { total += partial[buffer][w]; }
        buffer ^= 1;
        return total;
    };
    // The largest key `last` with at least `want` items from it: the last winner's key.
    std::uint32_t last = 0, beyond = 0x10000u;
    while (beyond - last > 1) {
        const std::uint32_t mid = (last + beyond) / 2;
        if (count_from(mid) >= want) {
            last = mid;
        } else {
            beyond = mid;
        }
    }
    const int above = count_from(last + 1);
    int ties = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) { ties += key[j] == last ? 1 : 0; }
    int tie = 0;
    Scan(scan).ExclusiveSum(ties, tie);
    __syncthreads();
    int chosen = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        bool t = key[j] > last;
        if (key[j] == last) {
            t = above + tie < want;
            ++tie;
        }
        take[j] = t;
        chosen += t ? 1 : 0;
    }
    int before = 0;
    Scan(scan).ExclusiveSum(chosen, before);
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        slot[j] = before;
        before += take[j] ? 1 : 0;
    }
}

// Block (tile, column): the tile's `want` largest logits, rows in order, into the column's run of
// tiles * want survivors; rows past the matrix pad with key 0 and row -1.
__launch_bounds__(kTileThreads) __global__
    void shortlist_tile_kernel(const __nv_bfloat16* coarse, std::int64_t column_stride, int rows,
                               int want, std::uint32_t* keys, std::int32_t* survivors) {
    const int column = int(blockIdx.y);
    const int tile   = int(blockIdx.x);
    const int first  = tile * kShortlistTileRows + int(threadIdx.x) * kItems;
    const auto* src  = reinterpret_cast<const std::uint16_t*>(coarse + column * column_stride);
    std::uint32_t key[kItems];
    if (first + kItems <= rows && reinterpret_cast<std::uintptr_t>(src + first) % 16 == 0) {
        const uint4 v               = *reinterpret_cast<const uint4*>(src + first);
        const std::uint32_t w[4]    = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            key[2 * j]     = bf16_order_key(std::uint16_t(w[j] & 0xFFFFu));
            key[2 * j + 1] = bf16_order_key(std::uint16_t(w[j] >> 16));
        }
    } else {
#pragma unroll
        for (int j = 0; j < kItems; ++j) {
            key[j] = first + j < rows ? bf16_order_key(src[first + j]) : 0u;
        }
    }
    bool take[kItems];
    int slot[kItems];
    select_largest<kTileThreads>(key, want, take, slot);
    const std::int64_t base = (std::int64_t(column) * gridDim.x + tile) * want;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        if (take[j]) {
            keys[base + slot[j]]      = key[j];
            survivors[base + slot[j]] = first + j < rows ? first + j : -1;
        }
    }
}

// Block `column`: the `want` largest of the column's `merged` survivors (tile-major, so in row
// order), their rows into candidates [column][want]; zeroes the column's rescoring key.
template <int Threads>
__launch_bounds__(Threads) __global__
    void shortlist_merge_kernel(const std::uint32_t* keys, const std::int32_t* survivors,
                                int merged, int want, std::int32_t* candidates,
                                unsigned long long* best) {
    const int column        = int(blockIdx.x);
    const std::int64_t base = std::int64_t(column) * merged;
    const int first         = int(threadIdx.x) * kItems;
    std::uint32_t key[kItems];
#pragma unroll
    for (int j = 0; j < kItems; ++j) { key[j] = first + j < merged ? keys[base + first + j] : 0u; }
    bool take[kItems];
    int slot[kItems];
    select_largest<Threads>(key, want, take, slot);
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        if (take[j]) {
            candidates[std::int64_t(column) * want + slot[j]] = survivors[base + first + j];
        }
    }
    if (threadIdx.x == 0) { best[column] = 0; }
}

__global__ void shortlist_finish_kernel(const unsigned long long* best, const std::int32_t* ids,
                                        std::int32_t* out, int columns) {
    const int column = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= columns) { return; }
    const auto row = std::int32_t(0xFFFFFFFFu - std::uint32_t(best[column] & 0xFFFFFFFFull));
    out[column]    = ids != nullptr ? ids[row] : row;
}

using ShortlistLaunch = void (*)(const ShortlistRescoreArgs&, int, cudaStream_t);

ShortlistLaunch shortlist_launch(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return shortlist_rescore_launch<GGML_TYPE_Q8_0>;
    case GgmlType::Q4_0: return shortlist_rescore_launch<GGML_TYPE_Q4_0>;
    case GgmlType::Q5_0: return shortlist_rescore_launch<GGML_TYPE_Q5_0>;
    case GgmlType::Q2_0: return shortlist_rescore_launch<GGML_TYPE_Q2_0>;
    case GgmlType::Q2_K: return shortlist_rescore_launch<GGML_TYPE_Q2_K>;
    case GgmlType::Q3_K: return shortlist_rescore_launch<GGML_TYPE_Q3_K>;
    case GgmlType::Q4_K: return shortlist_rescore_launch<GGML_TYPE_Q4_K>;
    case GgmlType::Q5_K: return shortlist_rescore_launch<GGML_TYPE_Q5_K>;
    case GgmlType::Q6_K: return shortlist_rescore_launch<GGML_TYPE_Q6_K>;
    case GgmlType::IQ2_XXS: return shortlist_rescore_launch<GGML_TYPE_IQ2_XXS>;
    case GgmlType::IQ2_XS: return shortlist_rescore_launch<GGML_TYPE_IQ2_XS>;
    case GgmlType::IQ2_S: return shortlist_rescore_launch<GGML_TYPE_IQ2_S>;
    case GgmlType::IQ3_XXS: return shortlist_rescore_launch<GGML_TYPE_IQ3_XXS>;
    case GgmlType::IQ3_S: return shortlist_rescore_launch<GGML_TYPE_IQ3_S>;
    case GgmlType::IQ1_S: return shortlist_rescore_launch<GGML_TYPE_IQ1_S>;
    case GgmlType::IQ1_M: return shortlist_rescore_launch<GGML_TYPE_IQ1_M>;
    case GgmlType::IQ4_NL: return shortlist_rescore_launch<GGML_TYPE_IQ4_NL>;
    case GgmlType::IQ4_XS: return shortlist_rescore_launch<GGML_TYPE_IQ4_XS>;
    }
    throw std::invalid_argument("gguf shortlist argmax: unsupported block type");
}

constexpr std::size_t align256(std::size_t n) { return (n + 255) / 256 * 256; }

struct ShortlistLayout {
    int tiles = 0, merged = 0;
    std::size_t keys = 0, survivors = 0, candidates = 0, best = 0, total = 0;
};

ShortlistLayout shortlist_layout(int rows, int k, int columns, int candidates) {
    ShortlistLayout l;
    l.tiles       = (rows + kShortlistTileRows - 1) / kShortlistTileRows;
    l.merged      = l.tiles * candidates;
    const auto cs = std::size_t(columns);
    l.keys        = align256(vector_activation_bytes(k, columns));
    l.survivors   = l.keys + align256(cs * l.merged * sizeof(std::uint32_t));
    l.candidates  = l.survivors + align256(cs * l.merged * sizeof(std::int32_t));
    l.best        = l.candidates + align256(cs * candidates * sizeof(std::int32_t));
    l.total       = l.best + align256(cs * sizeof(unsigned long long));
    return l;
}

void check_shortlist(int rows, int k, int columns, int candidates) {
    if (candidates < 1 || candidates > kMaxShortlistCandidates || rows < candidates || k <= 0 ||
        k % 32 != 0 || columns < 1 || columns > 65535) {
        throw std::invalid_argument("gguf shortlist argmax: unsupported shape");
    }
    if (std::int64_t((rows + kShortlistTileRows - 1) / kShortlistTileRows) * candidates >
        kMaxShortlistMerged) {
        throw std::invalid_argument("gguf shortlist argmax: too many rows for the candidates");
    }
}

} // namespace
} // namespace detail

std::size_t shortlist_argmax_bytes(int rows, int k, int columns, int candidates) {
    detail::check_shortlist(rows, k, columns, candidates);
    return detail::shortlist_layout(rows, k, columns, candidates).total;
}

void shortlist_argmax(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                      const __nv_bfloat16* coarse, std::int64_t coarse_column_stride,
                      const __nv_bfloat16* x, int columns, int candidates,
                      const std::int32_t* ids, std::int32_t* out, void* workspace,
                      cudaStream_t stream) {
    detail::check_shortlist(rows, k, columns, candidates);
    if (weight == nullptr || coarse == nullptr || x == nullptr || out == nullptr ||
        workspace == nullptr || coarse_column_stride < rows ||
        reinterpret_cast<std::uintptr_t>(workspace) % 256 != 0) {
        throw std::invalid_argument("gguf shortlist argmax: invalid operands");
    }
    const auto launch = detail::shortlist_launch(type);
    const auto l      = detail::shortlist_layout(rows, k, columns, candidates);
    auto* base        = static_cast<std::uint8_t*>(workspace);
    auto* keys        = reinterpret_cast<std::uint32_t*>(base + l.keys);
    auto* survivors   = reinterpret_cast<std::int32_t*>(base + l.survivors);
    auto* chosen      = reinterpret_cast<std::int32_t*>(base + l.candidates);
    auto* best        = reinterpret_cast<unsigned long long*>(base + l.best);
    quantize_vector_activation(x, k, columns, nullptr, base, stream);
    detail::shortlist_tile_kernel<<<dim3(l.tiles, columns), detail::kTileThreads, 0, stream>>>(
        coarse, coarse_column_stride, rows, candidates, keys, survivors);
    detail::check(cudaGetLastError(), "shortlist tile launch");
    if (l.merged <= detail::kTileThreads * detail::kItems) {
        detail::shortlist_merge_kernel<detail::kTileThreads>
            <<<columns, detail::kTileThreads, 0, stream>>>(keys, survivors, l.merged, candidates,
                                                          chosen, best);
    } else {
        detail::shortlist_merge_kernel<1024>
            <<<columns, 1024, 0, stream>>>(keys, survivors, l.merged, candidates, chosen, best);
    }
    detail::check(cudaGetLastError(), "shortlist merge launch");
    const detail::ShortlistRescoreArgs args{
        static_cast<const std::uint8_t*>(weight),
        row_bytes,
        k,
        candidates,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + detail::vec_scales_offset(k, columns)),
        chosen,
        best};
    launch(args, columns, stream);
    detail::shortlist_finish_kernel<<<(columns + 127) / 128, 128, 0, stream>>>(best, ids, out,
                                                                              columns);
    detail::check(cudaGetLastError(), "shortlist finish launch");
}

} // namespace ninfer::ops::gguf
