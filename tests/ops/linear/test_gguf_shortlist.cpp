// shortlist_argmax: the argmax of a GGUF matrix over the top rows of coarse logits.
//
// Oracle: the candidate set is the exact top `candidates` of the coarse BF16 values (a tie to the
// lower row), computed on the host; the FP64 dot of ggml's dequantized rows with x bounds the
// chosen row's value from below by the q8_1 activation's rounding error. The contract's exact form
// is checked too: the chosen row is the argmax over the candidates of linear()'s BF16 output at
// T <= 8 (the vector product), the lower row on a tie.
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/shortlist_argmax.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

struct TypeCase {
    QType qtype;
    const char* name;
    std::vector<int> half_fields; // byte offsets of a block's binary16 multipliers
};

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

std::vector<std::uint8_t> random_blocks(const TypeCase& c, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng) {
    const auto block         = gguf_block_shape(c.qtype);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(bytes.data() + i * block.bytes + offset, &bits, sizeof(bits));
        }
    }
    return bytes;
}

Weight gguf_weight(QType qtype, const void* data, std::int32_t rows, std::int32_t k) {
    Weight w;
    w.qtype   = qtype;
    w.layout  = QuantLayout::GgufBlocks;
    w.qdata   = data;
    w.payload = data;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim       = 2;
    w.group_size = static_cast<std::uint32_t>(gguf_block_shape(qtype).elements);
    w.group      = gguf_block_shape(qtype).elements;
    return w;
}

enum class Coarse {
    Random, // distinct-ish values
    Ties,   // four distinct values: ties at every selection boundary, across tiles
    Flat,   // one value: the candidates are rows [0, candidates)
};

struct Case {
    const TypeCase* type;
    std::int32_t rows, k;
    std::vector<std::int32_t> widths;
    std::vector<std::int32_t> candidates;
    Coarse coarse;
    bool duplicate_rows; // odd rows repeat the even row before them: ties in the exact values
    bool mapped;         // the exact matrix in mapped host memory
    bool ids;
};

std::string label(const Case& c, std::int32_t t, std::int32_t candidates) {
    static const char* coarse[] = {"random", "ties", "flat"};
    return std::string(c.type->name) + " R=" + std::to_string(c.rows) + " K=" +
           std::to_string(c.k) + " T=" + std::to_string(t) + " C=" + std::to_string(candidates) +
           " " + coarse[int(c.coarse)] + (c.duplicate_rows ? " dup" : "") +
           (c.mapped ? " mapped" : "") + (c.ids ? " ids" : "");
}

// The `count` largest of values (BF16 bits as float), a tie to the lower index, in index order.
std::vector<std::int32_t> top_rows(const std::vector<float>& values, std::int32_t count) {
    std::vector<std::int32_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::int32_t a, std::int32_t b) { return values[a] > values[b]; });
    order.resize(count);
    std::sort(order.begin(), order.end());
    return order;
}

int run(const Case& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const TypeCase& type      = *c.type;
    const auto block          = gguf_block_shape(type.qtype);
    const std::int64_t stride = std::int64_t(c.k / block.elements) * block.bytes;
    auto host_blocks          = random_blocks(type, c.rows, c.k, rng);
    if (c.duplicate_rows) {
        for (std::int32_t r = 1; r < c.rows; r += 2) {
            std::memcpy(host_blocks.data() + r * stride, host_blocks.data() + (r - 1) * stride,
                        std::size_t(stride));
        }
    }
    DeviceBuffer device_blocks(host_blocks.size());
    device_blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    void* mapped_host     = nullptr;
    const void* exact_ptr = device_blocks.p;
    if (c.mapped) {
        check(cudaHostAlloc(&mapped_host, host_blocks.size(), cudaHostAllocMapped), "host alloc");
        std::memcpy(mapped_host, host_blocks.data(), host_blocks.size());
        void* device_view = nullptr;
        check(cudaHostGetDevicePointer(&device_view, mapped_host, 0), "mapped pointer");
        exact_ptr = device_view;
    }
    const Weight exact      = gguf_weight(type.qtype, exact_ptr, c.rows, c.k);
    const Weight on_device  = gguf_weight(type.qtype, device_blocks.p, c.rows, c.k);

    // ggml's dequantized rows, the oracle's weights, decoded as the checks reach them.
    DeviceBuffer decoded_device(std::size_t(c.k) * sizeof(float));
    std::map<std::int32_t, std::vector<float>> decoded;
    const auto decoded_row = [&](std::int32_t r) -> const std::vector<float>& {
        auto& row = decoded[r];
        if (row.empty()) {
            ops::gguf::dequantize_rows(ops::detail::gguf_type(type.qtype),
                                       static_cast<const std::uint8_t*>(device_blocks.p) +
                                           r * stride,
                                       stride, c.k, nullptr, 1,
                                       static_cast<float*>(decoded_device.p), c.k, nullptr);
            check(cudaDeviceSynchronize(), "dequantize");
            row.resize(c.k);
            decoded_device.copy_to_host(row.data(), row.size() * sizeof(float));
        }
        return row;
    };

    const std::int32_t max_t = *std::max_element(c.widths.begin(), c.widths.end());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(c.k) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    std::vector<__nv_bfloat16> coarse_host(std::size_t(c.rows) * max_t);
    std::uniform_int_distribution<int> level(0, 3);
    for (auto& v : coarse_host) {
        v = __float2bfloat16(c.coarse == Coarse::Random ? normal(rng)
                             : c.coarse == Coarse::Ties ? float(level(rng))
                                                        : 0.5f);
    }
    std::vector<std::int32_t> ids(c.rows);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    DeviceBuffer coarse_device(coarse_host.size() * sizeof(__nv_bfloat16));
    DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));
    coarse_device.copy_from_host(coarse_host.data(), coarse_host.size() * sizeof(__nv_bfloat16));
    ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    Tensor ids_tensor(ids_device.p, DType::I32, {c.rows});

    std::size_t capacity = ops::linear_workspace_capacity_bytes(
        type.qtype, c.rows, c.k, ops::LinearPolicy::A16Only, 1, max_t);
    for (const std::int32_t candidates : c.candidates) {
        capacity = std::max(capacity, ops::shortlist_argmax_workspace_bytes(c.rows, c.k, max_t,
                                                                            candidates));
    }
    WorkspaceArena workspace(capacity + (1U << 20));

    int failures = 0;
    for (const std::int32_t t : c.widths) {
        Tensor x(x_device.p, DType::BF16, {c.k, t});
        Tensor coarse(coarse_device.p, DType::BF16, {c.rows, t});
        // linear()'s BF16 rows: the contract's exact values (vector product at T <= 8).
        std::vector<__nv_bfloat16> y_host;
        if (t <= ops::gguf::kMaxVectorColumns) {
            DeviceBuffer y_device(std::size_t(c.rows) * t * sizeof(__nv_bfloat16));
            Tensor y(y_device.p, DType::BF16, {c.rows, t});
            ops::linear(x, on_device, y, ops::LinearPolicy::A16Only, workspace, nullptr);
            check(cudaDeviceSynchronize(), "linear");
            y_host.resize(std::size_t(c.rows) * t);
            y_device.copy_to_host(y_host.data(), y_host.size() * sizeof(__nv_bfloat16));
        }
        for (const std::int32_t candidates : c.candidates) {
            DeviceBuffer out_device(std::size_t(t) * sizeof(std::int32_t));
            Tensor out(out_device.p, DType::I32, {t});
            ops::shortlist_argmax(coarse, x, exact, candidates, c.ids ? &ids_tensor : nullptr, out,
                                  workspace, nullptr);
            check(cudaDeviceSynchronize(), "shortlist_argmax");
            std::vector<std::int32_t> got(t);
            out_device.copy_to_host(got.data(), got.size() * sizeof(std::int32_t));
            for (std::int32_t col = 0; col < t; ++col) {
                std::vector<float> column(c.rows);
                for (std::int32_t r = 0; r < c.rows; ++r) {
                    column[r] = __bfloat162float(coarse_host[std::size_t(col) * c.rows + r]);
                }
                const auto set = top_rows(column, candidates);
                // The chosen row, mapped back through ids.
                std::int32_t row = got[col];
                if (c.ids) {
                    const auto at = std::find(ids.begin(), ids.end(), got[col]);
                    row = at == ids.end() ? -1 : std::int32_t(at - ids.begin());
                }
                if (!std::binary_search(set.begin(), set.end(), row)) {
                    std::cerr << "FAIL " << label(c, t, candidates) << " column " << col
                              << ": row " << row << " is not a candidate\n";
                    ++failures;
                    continue;
                }
                // FP64 oracle: the chosen value against the candidates' best, within the q8_1
                // rounding bound of both (half a step per value, d = max|x| / 127 per 32, plus the
                // BF16 rounding of the stored value).
                const float* xc = nullptr;
                std::vector<float> xf(c.k);
                for (std::int32_t i = 0; i < c.k; ++i) {
                    xf[i] = __bfloat162float(x_host[std::size_t(col) * c.k + i]);
                }
                xc = xf.data();
                std::vector<double> step(c.k / 32);
                for (std::int32_t s = 0; s < c.k / 32; ++s) {
                    double amax = 0;
                    for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::abs(double(xc[32 * s + i]))); }
                    step[s] = amax / 127.0 * 0.5;
                }
                const auto value = [&](std::int32_t r, double& bound) {
                    double acc = 0, err = 0;
                    const float* w = decoded_row(r).data();
                    for (std::int32_t i = 0; i < c.k; ++i) {
                        acc += double(w[i]) * xc[i];
                        err += std::abs(double(w[i])) * step[i / 32];
                    }
                    bound = err + std::abs(acc) * 0x1p-8 + 1e-6;
                    return acc;
                };
                double chosen_bound = 0, best_bound = 0, best = -INFINITY;
                const double chosen = value(row, chosen_bound);
                for (const std::int32_t r : set) {
                    double b = 0;
                    const double v = value(r, b);
                    if (v > best) {
                        best       = v;
                        best_bound = b;
                    }
                }
                if (chosen < best - (chosen_bound + best_bound)) {
                    std::cerr << "FAIL " << label(c, t, candidates) << " column " << col
                              << ": value " << chosen << " below the best " << best << '\n';
                    ++failures;
                }
                // Exact contract: argmax of the vector product's BF16 rows, the lower row on a tie.
                if (!y_host.empty()) {
                    std::int32_t want = set[0];
                    for (const std::int32_t r : set) {
                        const float v = __bfloat162float(y_host[std::size_t(col) * c.rows + r]);
                        const float w = __bfloat162float(y_host[std::size_t(col) * c.rows + want]);
                        if (v > w) { want = r; }
                    }
                    if (row != want) {
                        std::cerr << "FAIL " << label(c, t, candidates) << " column " << col
                                  << ": row " << row << ", the vector product's argmax is " << want
                                  << '\n';
                        ++failures;
                    }
                }
            }
        }
    }
    if (mapped_host != nullptr) { check(cudaFreeHost(mapped_host), "host free"); }
    return failures;
}

int refusals(const TypeCase& type) {
    int failures = 0;
    DeviceBuffer blocks(std::size_t(64) * 136 * 4);
    const Weight w = gguf_weight(type.qtype, blocks.p, 64, 1024);
    DeviceBuffer coarse_device(64 * 2 * sizeof(__nv_bfloat16));
    DeviceBuffer x_device(1024 * 2 * sizeof(__nv_bfloat16));
    DeviceBuffer out_device(2 * sizeof(std::int32_t));
    Tensor coarse(coarse_device.p, DType::BF16, {64, 2});
    Tensor x(x_device.p, DType::BF16, {1024, 2});
    Tensor out(out_device.p, DType::I32, {2});
    Tensor short_out(out_device.p, DType::I32, {1});
    Tensor wrong_x(x_device.p, DType::BF16, {512, 2});
    WorkspaceArena workspace(1U << 20);
    const auto refuses = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return;
        }
        std::cerr << "FAIL shortlist_argmax accepts " << what << '\n';
        ++failures;
    };
    refuses("no candidates", [&] { ops::shortlist_argmax(coarse, x, w, 0, nullptr, out, workspace, nullptr); });
    Tensor coarse32(coarse_device.p, DType::BF16, {32, 2});
    refuses("more candidates than rows", [&] {
        ops::shortlist_argmax(coarse32, x, gguf_weight(type.qtype, blocks.p, 32, 1024), 33,
                              nullptr, out, workspace, nullptr);
    });
    refuses("more than the most candidates", [&] {
        ops::shortlist_argmax(coarse, x, w, 65, nullptr, out, workspace, nullptr);
    });
    refuses("a short output", [&] { ops::shortlist_argmax(coarse, x, w, 4, nullptr, short_out, workspace, nullptr); });
    refuses("x of another width", [&] { ops::shortlist_argmax(coarse, wrong_x, w, 4, nullptr, out, workspace, nullptr); });
    Weight gathered        = w;
    gathered.input_columns = static_cast<const std::int32_t*>(x_device.p);
    refuses("an input gather", [&] { ops::shortlist_argmax(coarse, x, gathered, 4, nullptr, out, workspace, nullptr); });
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const TypeCase iq4_xs{QType::GGUF_IQ4_XS, "iq4_xs", {0}};
        const TypeCase iq2_xxs{QType::GGUF_IQ2_XXS, "iq2_xxs", {0}};
        const TypeCase q3_k{QType::GGUF_Q3_K, "q3_k", {108}};
        const TypeCase q8_0{QType::GGUF_Q8_0, "q8_0", {0}};
        const std::vector<Case> cases{
            // The 27B's proposal shape: 131072 rows of K = 5120, drafting widths.
            {&iq4_xs, 131072, 5120, {1, 3}, {1, 32, 64}, Coarse::Random, false, false, true},
            {&iq4_xs, 131072, 5120, {1}, {32}, Coarse::Random, false, true, true},
            // A partial last tile; ties in the coarse logits across tiles and in the exact values.
            {&iq4_xs, 5124, 5120, {1, 2, 8, 9}, {1, 7, 32, 64}, Coarse::Ties, true, false, false},
            {&iq4_xs, 5124, 5120, {8}, {1, 32}, Coarse::Flat, true, true, true},
            // Rows of 1320 bytes (4-byte staging) with the grid table; rows of 330 bytes (2-byte
            // staging); fewer rows than a tile, and as many candidates as rows.
            {&iq2_xxs, 4100, 5120, {1, 4}, {32}, Coarse::Random, false, true, false},
            {&q3_k, 3000, 768, {1, 5}, {16, 64}, Coarse::Ties, true, false, true},
            {&q8_0, 64, 1024, {1, 2}, {64, 63}, Coarse::Random, false, false, false},
            // More survivors than one 256-thread block holds (the 1024-thread merge).
            {&q8_0, 300000, 256, {1, 2}, {32}, Coarse::Ties, false, false, true},
        };
        int failures       = 0;
        std::uint32_t seed = 4242;
        for (const auto& c : cases) {
            const int f = run(c, seed++);
            std::cout << (f == 0 ? "OK   " : "FAIL ") << label(c, c.widths.back(), c.candidates.back())
                      << '\n';
            failures += f;
        }
        failures += refusals(iq4_xs);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF shortlist argmax\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF shortlist argmax: " << error.what() << '\n';
        return 1;
    }
}
