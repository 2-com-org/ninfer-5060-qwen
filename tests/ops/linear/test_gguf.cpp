// GGUF block matrices through the public linear, linear_add and linear_swiglu Ops and the embedding
// gather, against an FP64 product of the exactly dequantized weights. The products quantize the
// activation to q8_1 as llama.cpp does, so they are held to that arithmetic's error, not to A16's.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
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
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

struct TypeCase {
    QType qtype;
    const char* name;
    // Byte offsets of the binary16 multipliers inside one block, set to finite values.
    std::vector<int> half_fields;
};

std::vector<TypeCase> cases() {
    return {
        {QType::GGUF_Q8_0, "q8_0", {0}},       {QType::GGUF_Q2_K, "q2_k", {80, 82}},
        {QType::GGUF_Q3_K, "q3_k", {108}},     {QType::GGUF_Q4_K, "q4_k", {0, 2}},
        {QType::GGUF_Q5_K, "q5_k", {0, 2}},    {QType::GGUF_Q6_K, "q6_k", {208}},
        {QType::GGUF_IQ2_XXS, "iq2_xxs", {0}}, {QType::GGUF_IQ2_XS, "iq2_xs", {0}},
        {QType::GGUF_IQ2_S, "iq2_s", {0}},     {QType::GGUF_IQ3_XXS, "iq3_xxs", {0}},
        {QType::GGUF_IQ3_S, "iq3_s", {0}},     {QType::GGUF_IQ1_S, "iq1_s", {0}},
        {QType::GGUF_IQ1_M, "iq1_m", {}},      {QType::GGUF_IQ4_NL, "iq4_nl", {0}},
        {QType::GGUF_IQ4_XS, "iq4_xs", {0}},   {QType::GGUF_Q4_0, "q4_0", {0}},
        {QType::GGUF_Q5_0, "q5_0", {0}},       {QType::GGUF_Q2_0, "q2_0", {0}},
    };
}

float half_value(const std::uint8_t* p) {
    __half h;
    std::memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

// An independent host decode of the formats simple enough to state here, each value d * (code -
// offset): Q4_0 (low nibbles first, then high), Q5_0 (Q4_0 plus bit j of qh as bit 4 of value j)
// and Q2_0 (two bits per value, low bits first). Empty for every other type.
std::vector<float> host_decode(QType qtype, const std::vector<std::uint8_t>& bytes) {
    std::vector<float> out;
    const auto block = gguf_block_shape(qtype);
    if (qtype != QType::GGUF_Q4_0 && qtype != QType::GGUF_Q5_0 && qtype != QType::GGUF_Q2_0) {
        return out;
    }
    const std::size_t blocks = bytes.size() / block.bytes;
    out.reserve(blocks * block.elements);
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::uint8_t* b = bytes.data() + i * block.bytes;
        const float d         = half_value(b);
        for (int j = 0; j < block.elements; ++j) {
            int code = 0;
            if (qtype == QType::GGUF_Q2_0) {
                code = ((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1;
            } else {
                const std::uint8_t* qs = b + (qtype == QType::GGUF_Q5_0 ? 6 : 2);
                code                   = j < 16 ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
                if (qtype == QType::GGUF_Q5_0) {
                    std::uint32_t qh;
                    std::memcpy(&qh, b + 2, sizeof(qh));
                    code = (code | int(((qh >> j) & 1U) << 4)) - 16;
                } else {
                    code -= 8;
                }
            }
            out.push_back(d * float(code));
        }
    }
    return out;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Random blocks with every binary16 multiplier finite; IQ1_M spreads its multiplier over the top
// nibbles of its four scale words.
std::vector<std::uint8_t> random_blocks(const TypeCase& c, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng) {
    const auto block = gguf_block_shape(c.qtype);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (c.qtype == QType::GGUF_IQ1_M) {
            const std::uint16_t d = half_bits(scale(rng));
            auto* sc              = reinterpret_cast<std::uint16_t*>(base + 48);
            for (int j = 0; j < 4; ++j) {
                sc[j] = static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
            }
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

std::vector<float> to_float(const std::vector<__nv_bfloat16>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = __bfloat162float(values[i]); }
    return out;
}

struct Error {
    double relative_rms = 0;
    double relative_max = 0;
};

Error compare(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0, den = 0, worst = 0, scale = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = double(got[i]) - want[i];
        num += d * d;
        den += want[i] * want[i];
        worst = std::max(worst, std::abs(d));
        scale = std::max(scale, std::abs(want[i]));
    }
    return {std::sqrt(num / std::max(den, 1e-30)), worst / std::max(scale, 1e-30)};
}

// K = 1536 and K = 2560, Qwen3.8-Flash-Next's hidden width, whose 80 slices leave the vector
// products' last pass over a row with half of a warp's lanes idle; and Gemma 4's widths 2816, 2112
// and 704, which are no whole number of the matrix kernel's 512-value activation steps (2112 and
// 704 not even of its 256-value weight steps). The 27B-sized shapes (4100 and 5124 rows of K =
// 5120) take the vector kernels' static-K and four-row forms, with a partial last block of rows.
int run_type(const TypeCase& c, std::uint32_t seed, std::int32_t k, std::int32_t row_count,
             const std::vector<std::int32_t>& widths) {
    const std::int32_t kRows = row_count;
    const std::int32_t kK    = k;
    const std::int32_t max_t = *std::max_element(widths.begin(), widths.end());
    std::mt19937 rng(seed);
    int failures = 0;

    const auto host_blocks = random_blocks(c, 2 * kRows, kK, rng);
    // Zero bytes after the last row, as a bound device tail keeps them: a row whose K is no whole
    // number of the matrix kernel's 256-value steps reads its last step into the next row.
    DeviceBuffer blocks(host_blocks.size() + 4096);
    blocks.fill(0);
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const auto block        = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(kK / block.elements) * block.bytes;
    const Weight gate = gguf_weight(c.qtype, blocks.p, kRows, kK);
    Weight up         = gate;
    up.qdata          = static_cast<const std::byte*>(blocks.p) + kRows * row_bytes;
    const Weight both = gguf_weight(c.qtype, blocks.p, 2 * kRows, kK);

    // Exact weights.
    DeviceBuffer exact_device(std::size_t(2) * kRows * kK * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, kK,
                               nullptr, 2 * kRows, static_cast<float*>(exact_device.p), kK,
                               nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * kK);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));
    for (const float v : exact) {
        if (!std::isfinite(v)) {
            std::cerr << c.name << ": non-finite dequantized weight\n";
            return 1;
        }
    }
    if (const auto host = host_decode(c.qtype, host_blocks); !host.empty() && host != exact) {
        std::cerr << "FAIL " << c.name << ": dequantization differs from the host decode\n";
        return 1;
    }

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(kK) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    const auto x_float = to_float(x_host);
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    std::vector<std::int32_t> columns(kK);
    for (std::int32_t i = 0; i < kK; ++i) { columns[i] = i; }
    std::shuffle(columns.begin(), columns.end(), rng);
    DeviceBuffer columns_device(columns.size() * sizeof(std::int32_t));
    columns_device.copy_from_host(columns.data(), columns.size() * sizeof(std::int32_t));
    Weight gathered        = gate;
    gathered.input_columns = static_cast<const std::int32_t*>(columns_device.p);

    const auto product = [&](std::int32_t row0, std::int32_t t, bool permuted) {
        std::vector<double> out(std::size_t(kRows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < kRows; ++r) {
                double acc = 0;
                const float* w = exact.data() + std::size_t(row0 + r) * kK;
                const float* x = x_float.data() + std::size_t(col) * kK;
                for (std::int32_t i = 0; i < kK; ++i) {
                    acc += double(w[i]) * x[permuted ? columns[i] : i];
                }
                out[std::size_t(col) * kRows + r] = acc;
            }
        }
        return out;
    };

    const std::size_t capacity = std::max(
        {ops::linear_workspace_capacity_bytes(c.qtype, kRows, kK, ops::LinearPolicy::A16Only, 1,
                                              max_t),
         ops::linear_swiglu_workspace_capacity_bytes(c.qtype, 2 * kRows, kK,
                                                     ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, kK,
                                                          ops::LinearPolicy::A16Only, 1, max_t)});
    WorkspaceArena workspace(capacity + (1U << 20));

    // q8_1 activations cost one product about 1 % RMS against the exact weights; a SwiGLU output
    // carries two such products, so it gets twice the budget.
    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want) {
        const Error e     = compare(got, want);
        const double gain = label.starts_with("swiglu") ? 2.0 : 1.0;
        const bool ok     = e.relative_rms < 2e-2 * gain && e.relative_max < 8e-2 * gain;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " " << label << " rms=" << e.relative_rms
                      << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {kK, t});
        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);

        ops::linear(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        const auto plain = product(0, t, false);
        failures += run_check("linear T=" + std::to_string(t), to_float(host), plain);

        ops::linear(x, gathered, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "gathered linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("gathered T=" + std::to_string(t), to_float(host),
                              product(0, t, true));

        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        ops::linear_add(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_add");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }
        failures += run_check("linear_add T=" + std::to_string(t), to_float(host), sum);

        const auto upper = product(kRows, t, false);
        std::vector<double> swiglu(plain.size());
        for (std::size_t i = 0; i < swiglu.size(); ++i) {
            swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
        }
        ops::linear_swiglu(x, both, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu T=" + std::to_string(t), to_float(host), swiglu);

        ops::linear_swiglu(x, gate, up, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu pair");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu pair T=" + std::to_string(t), to_float(host), swiglu);
    }
    // The planned capacity covers every call; the slack above it only keeps a shortfall from
    // aborting the numerical checks before this one reports it.
    if (workspace.peak_used() > capacity) {
        std::cerr << "FAIL " << c.name << " workspace peak " << workspace.peak_used()
                  << " bytes over the planned " << capacity << '\n';
        ++failures;
    }

    // The token-table gather is the exact weights rounded to BF16.
    const std::vector<std::int32_t> ids{3, 0, 255, 17, 3};
    DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    DeviceBuffer rows_device(ids.size() * kK * sizeof(__nv_bfloat16));
    Tensor ids_tensor(ids_device.p, DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor rows(rows_device.p, DType::BF16, {kK, static_cast<std::int32_t>(ids.size())});
    ops::embedding(ids_tensor, gate, rows, nullptr);
    check(cudaDeviceSynchronize(), "embedding");
    std::vector<__nv_bfloat16> gathered_rows(ids.size() * kK);
    rows_device.copy_to_host(gathered_rows.data(), gathered_rows.size() * sizeof(__nv_bfloat16));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::int32_t j = 0; j < kK; ++j) {
            const float want = __bfloat162float(__float2bfloat16(exact[std::size_t(ids[i]) * kK + j]));
            if (__bfloat162float(gathered_rows[i * kK + j]) != want) {
                std::cerr << "FAIL " << c.name << " embedding row " << ids[i] << " column " << j
                          << '\n';
                ++failures;
                i = ids.size();
                break;
            }
        }
    }
    return failures;
}

// The wide products of the six i-quants the tile kernel serves (their matrix-kernel activation also
// carries an FP32 d per 32 values) against their own arithmetic: x quantized on the host as the
// activation is (d = max|x| / 127 in FP32, each value rounded to x / d) times the exactly
// dequantized weights, in FP64. The matrix and tile kernels then differ from it by FP32
// accumulation only: the FP32 plane is held to 2e-5 of the largest output, the BF16 forms to BF16
// rounding. The Ops take whichever route gguf::takes_tile_kernel picks; the tile kernel is also
// run directly in every epilogue form. 27B widths (K = 5120, 6144 and 17408 take the two-scale
// types' static-K forms, 2560 the general one), a partial last tile of columns, and a zero
// 32-value group (d = 0).
int run_q8_oracle(const TypeCase& c, std::uint32_t seed, std::int32_t k, std::int32_t kRows,
                  const std::vector<std::int32_t>& widths) {
    const std::int32_t max_t = *std::max_element(widths.begin(), widths.end());
    std::mt19937 rng(seed);
    int failures = 0;

    const auto host_blocks = random_blocks(c, 2 * kRows, k, rng);
    DeviceBuffer blocks(host_blocks.size() + 4096);
    blocks.fill(0);
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const auto block             = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const Weight gate            = gguf_weight(c.qtype, blocks.p, kRows, k);
    Weight up                    = gate;
    up.qdata = static_cast<const std::byte*>(blocks.p) + kRows * row_bytes;

    DeviceBuffer exact_device(std::size_t(2) * kRows * k * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, k, nullptr,
                               2 * kRows, static_cast<float*>(exact_device.p), k, nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * k);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(k) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    for (int i = 0; i < 32; ++i) { x_host[std::size_t(k) + 64 + i] = __float2bfloat16(0.0f); }
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    std::vector<std::int32_t> order(k);
    for (std::int32_t i = 0; i < k; ++i) { order[i] = i; }
    std::shuffle(order.begin(), order.end(), rng);
    DeviceBuffer order_device(order.size() * sizeof(std::int32_t));
    order_device.copy_from_host(order.data(), order.size() * sizeof(std::int32_t));
    Weight gathered        = gate;
    gathered.input_columns = static_cast<const std::int32_t*>(order_device.p);

    // out[col][r] over rows [row0, row0 + count) for every column, x gathered through `order`.
    const auto oracle = [&](std::int32_t row0, std::int32_t count, bool permuted) {
        std::vector<double> xq(std::size_t(k) * max_t);
        for (std::int32_t col = 0; col < max_t; ++col) {
            for (std::int32_t g = 0; g < k; g += 32) {
                float v[32];
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const std::int32_t src = permuted ? order[g + i] : g + i;
                    v[i] = __bfloat162float(x_host[std::size_t(col) * k + src]);
                    amax = std::max(amax, std::fabs(v[i]));
                }
                const float d_inv = 127.0f / amax;
                const float d     = 1.0f / d_inv;
                for (int i = 0; i < 32; ++i) {
                    const float q = amax == 0.0f ? 0.0f : std::round(v[i] * d_inv);
                    xq[std::size_t(col) * k + g + i] = double(q) * double(d);
                }
            }
        }
        std::vector<double> out(std::size_t(count) * max_t);
        for (std::int32_t col = 0; col < max_t; ++col) {
            const double* x = xq.data() + std::size_t(col) * k;
            for (std::int32_t r = 0; r < count; ++r) {
                const float* w = exact.data() + std::size_t(row0 + r) * k;
                double acc     = 0;
                for (std::int32_t i = 0; i < k; ++i) { acc += double(w[i]) * x[i]; }
                out[std::size_t(col) * count + r] = acc;
            }
        }
        return out;
    };
    const auto both     = oracle(0, 2 * kRows, false); // [col][gate rows; up rows]
    const auto permuted = oracle(0, kRows, true);
    const auto slice    = [&](const std::vector<double>& all, std::int32_t stride,
                           std::int32_t row0, std::int32_t t) {
        std::vector<double> out(std::size_t(kRows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < kRows; ++r) {
                out[std::size_t(col) * kRows + r] = all[std::size_t(col) * stride + row0 + r];
            }
        }
        return out;
    };

    const ops::detail::GgufShape shape{c.qtype, kRows, k};
    const std::size_t capacity = std::max(
        {ops::detail::gguf_project_workspace_bytes({&shape, 1}, 1, max_t),
         ops::linear_workspace_capacity_bytes(c.qtype, kRows, k, ops::LinearPolicy::A16Only, 1,
                                              max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, k,
                                                          ops::LinearPolicy::A16Only, 1, max_t)});
    WorkspaceArena workspace(capacity + (1U << 20));

    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want, bool bf16) {
        const Error e = compare(got, want);
        const bool ok = bf16 ? e.relative_rms < 3e-3 && e.relative_max < 4e-3
                             : e.relative_max < 2e-5;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " K=" << k << " q8 " << label
                      << " rms=" << e.relative_rms << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {k, t});
        const auto plain = slice(both, 2 * kRows, 0, t);
        const auto upper = slice(both, 2 * kRows, kRows, t);
        const std::string at = "T=" + std::to_string(t);

        DeviceBuffer plane_device(std::size_t(kRows) * t * sizeof(float));
        std::vector<float> plane(std::size_t(kRows) * t);
        ops::detail::GgufProduct product;
        product.f32      = static_cast<float*>(plane_device.p);
        product.f32_rows = kRows;
        for (const bool gather : {false, true}) {
            product.weight = gather ? &gathered : &gate;
            ops::detail::gguf_project(x, {&product, 1}, workspace, nullptr);
            check(cudaDeviceSynchronize(), "gguf_project");
            plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
            failures += run_check((gather ? "gathered fp32 " : "fp32 ") + at, plane,
                                  gather ? slice(permuted, kRows, 0, t) : plain, false);
        }

        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);
        ops::linear(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("linear " + at, to_float(host), plain, true);

        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        ops::linear_add(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_add");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }
        failures += run_check("linear_add " + at, to_float(host), sum, true);

        std::vector<double> swiglu(plain.size());
        for (std::size_t i = 0; i < swiglu.size(); ++i) {
            swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
        }
        ops::linear_swiglu(x, gate, up, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu pair");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu pair " + at, to_float(host), swiglu, true);

        // The tile kernel itself, whichever route the products above took, in each epilogue form
        // it compiles: FP32 and BF16 Store, BF16 Accumulate, BF16 SiLU GateProduct, and the
        // general one (FP32 Accumulate, GELU GateProduct).
        const auto type = ops::detail::gguf_type(c.qtype);
        if (!ops::gguf::has_tile_kernel(type, k)) { continue; }
        DeviceBuffer activation(ops::gguf::tile_activation_bytes(k, t));
        DeviceBuffer gathered_activation(ops::gguf::tile_activation_bytes(k, t));
        const auto* xb = static_cast<const __nv_bfloat16*>(x_device.p);
        ops::gguf::quantize_tile_activation(xb, k, t, nullptr, activation.p, nullptr);
        ops::gguf::quantize_tile_activation(xb, k, t, gathered.input_columns,
                                            gathered_activation.p, nullptr);
        const auto tile = [&](const void* weight, const DeviceBuffer& act,
                              const ops::gguf::VectorOutput& o) {
            ops::gguf::tile_product(type, weight, row_bytes, kRows, k, act.p, t, o, nullptr);
            check(cudaDeviceSynchronize(), "tile_product");
        };
        ops::gguf::VectorOutput f32_out;
        f32_out.f32           = static_cast<float*>(plane_device.p);
        f32_out.column_stride = kRows;
        for (const bool gather : {false, true}) {
            tile(blocks.p, gather ? gathered_activation : activation, f32_out);
            plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
            failures += run_check((gather ? "tile gathered fp32 " : "tile fp32 ") + at, plane,
                                  gather ? slice(permuted, kRows, 0, t) : plain, false);
        }
        std::vector<float> base(plane.size());
        for (auto& v : base) { v = normal(rng); }
        plane_device.copy_from_host(base.data(), base.size() * sizeof(float));
        f32_out.epilogue = ops::gguf::Epilogue::Accumulate;
        tile(blocks.p, activation, f32_out);
        plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
        std::vector<double> accumulated = plain;
        for (std::size_t i = 0; i < accumulated.size(); ++i) { accumulated[i] += base[i]; }
        failures += run_check("tile fp32 accumulate " + at, plane, accumulated, false);

        ops::gguf::VectorOutput bf16_out;
        bf16_out.bf16          = static_cast<__nv_bfloat16*>(out_device.p);
        bf16_out.column_stride = kRows;
        tile(blocks.p, activation, bf16_out);
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("tile bf16 " + at, to_float(host), plain, true);

        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        bf16_out.epilogue = ops::gguf::Epilogue::Accumulate;
        tile(blocks.p, activation, bf16_out);
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("tile bf16 accumulate " + at, to_float(host), sum, true);

        // The gate plane holds the gate rows' products; the up rows multiply by act(gate).
        std::vector<float> gate_values(plain.size());
        for (std::size_t i = 0; i < gate_values.size(); ++i) { gate_values[i] = float(plain[i]); }
        plane_device.copy_from_host(gate_values.data(), gate_values.size() * sizeof(float));
        bf16_out.epilogue           = ops::gguf::Epilogue::GateProduct;
        bf16_out.gate               = static_cast<const float*>(plane_device.p);
        bf16_out.gate_column_stride = kRows;
        for (const auto act : {ops::gguf::GateActivation::Silu, ops::gguf::GateActivation::GeluTanh}) {
            bf16_out.activation = act;
            tile(up.qdata, activation, bf16_out);
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
            std::vector<double> gated(plain.size());
            for (std::size_t i = 0; i < gated.size(); ++i) {
                const double g = gate_values[i];
                const double a =
                    act == ops::gguf::GateActivation::Silu
                        ? g / (1.0 + std::exp(-g))
                        : 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
                gated[i] = a * upper[i];
            }
            failures += run_check(std::string("tile ") +
                                      (act == ops::gguf::GateActivation::Silu ? "silu " : "gelu ") +
                                      at,
                                  to_float(host), gated, true);
        }
    }
    if (workspace.peak_used() > capacity) {
        std::cerr << "FAIL " << c.name << " K=" << k << " q8 workspace peak "
                  << workspace.peak_used() << " bytes over the planned " << capacity << '\n';
        ++failures;
    }
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
        int failures      = 0;
        std::uint32_t seed = 9001;
        const std::vector<std::int32_t> widths{1, 2, 3, 4, 5, 8, 9, 17, 40, 70, 130};
        for (const auto& c : cases()) {
            for (const std::int32_t k : {1536, 2560, 2816, 2112, 704}) {
                if (k % gguf_block_shape(c.qtype).elements != 0) { continue; }
                const int f = run_type(c, seed++, k, 256, widths);
                std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                          << '\n';
                failures += f;
            }
            // Both classes of the four-row route (N < K and N >= K), every width it may take.
            for (const std::int32_t rows : {4100, 5124}) {
                const int f = run_type(c, seed++, 5120, rows, {1, 2, 3, 4, 5, 6, 7, 8});
                std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                          << " K=5120 rows=" << rows << '\n';
                failures += f;
            }
        }
        for (const auto& c : cases()) {
            const auto type = ops::detail::gguf_type(c.qtype);
            if (c.qtype != QType::GGUF_IQ3_XXS && c.qtype != QType::GGUF_IQ3_S &&
                c.qtype != QType::GGUF_IQ2_XXS && c.qtype != QType::GGUF_IQ4_XS &&
                c.qtype != QType::GGUF_IQ2_XS && c.qtype != QType::GGUF_IQ2_S) {
                continue;
            }
            for (const std::int32_t k : {2560, 5120, 6144, 17408}) {
                const bool tile = ops::gguf::has_tile_kernel(type, k);
                const int f     = run_q8_oracle(c, seed++, k, 256, {100, 128, 200});
                std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                          << " q8_1 oracle" << (tile ? " (and the tile kernel)" : "") << '\n';
                failures += f;
                // Rows the matrix kernel cannot take (168 = 128 + 40) take the tile kernel.
                if (tile && k == 5120) {
                    const int g = run_q8_oracle(c, seed++, k, 168, {129, 200});
                    std::cout << (g == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                              << " rows=168 q8_1 oracle (tile kernel)\n";
                    failures += g;
                }
            }
            // A grid that fills whole waves of the device takes the tile kernel through the Ops:
            // their activation sharing, input gather and workspace plan.
            const int rows = ops::gguf::multiprocessor_count() * 128;
            if (ops::gguf::takes_tile_kernel(type, rows, 2560, 128)) {
                const int f = run_q8_oracle(c, seed++, 2560, rows, {128, 200});
                std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                          << " K=2560 rows=" << rows << " q8_1 oracle (tile route)\n";
                failures += f;
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF Linear: " << error.what() << '\n';
        return 1;
    }
}
