// GGUF block matrices through the public linear, linear_add and linear_swiglu Ops and the embedding
// gather, against an FP64 product of the exactly dequantized weights. The products quantize the
// activation to q8_1 as llama.cpp does, so they are held to that arithmetic's error, not to A16's;
// under an A8 policy the tile kernel's MXFP8 form is held to its own criterion.

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
#include <stdexcept>
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

        // The vector kernel's general epilogue, which the products above do not reach: FP32
        // Accumulate, the GELU GateProduct and the fused GELU product (Gemma 4's GeGLU).
        if (t <= ops::gguf::kMaxVectorColumns) {
            const auto type = ops::detail::gguf_type(c.qtype);
            DeviceBuffer activation(ops::gguf::vector_activation_bytes(kK, t));
            ops::gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x_device.p),
                                                  kK, t, nullptr, activation.p, nullptr);
            std::mt19937 plane_rng(seed + std::uint32_t(t));
            std::vector<float> plane(std::size_t(kRows) * t);
            for (auto& v : plane) { v = normal(plane_rng); }
            std::vector<double> accumulated = plain;
            for (std::size_t i = 0; i < plane.size(); ++i) { accumulated[i] += plane[i]; }
            DeviceBuffer plane_device(plane.size() * sizeof(float));
            plane_device.copy_from_host(plane.data(), plane.size() * sizeof(float));
            ops::gguf::VectorOutput f32_out;
            f32_out.f32           = static_cast<float*>(plane_device.p);
            f32_out.column_stride = kRows;
            f32_out.epilogue      = ops::gguf::Epilogue::Accumulate;
            ops::gguf::vector_product(type, blocks.p, row_bytes, kRows, kK, activation.p, t,
                                      f32_out, nullptr);
            check(cudaDeviceSynchronize(), "vector fp32 accumulate");
            plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
            failures += run_check("vector fp32 accumulate T=" + std::to_string(t), plane,
                                  accumulated);

            const auto gelu = [](double g) {
                return 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
            };
            std::vector<float> gate_values(plain.begin(), plain.end());
            plane_device.copy_from_host(gate_values.data(), gate_values.size() * sizeof(float));
            std::vector<double> gated(plain.size());
            for (std::size_t i = 0; i < gated.size(); ++i) {
                gated[i] = gelu(gate_values[i]) * upper[i];
            }
            ops::gguf::VectorOutput gelu_out;
            gelu_out.bf16               = static_cast<__nv_bfloat16*>(out_device.p);
            gelu_out.column_stride      = kRows;
            gelu_out.epilogue           = ops::gguf::Epilogue::GateProduct;
            gelu_out.gate               = static_cast<const float*>(plane_device.p);
            gelu_out.gate_column_stride = kRows;
            gelu_out.activation         = ops::gguf::GateActivation::GeluTanh;
            ops::gguf::vector_product(type, up.qdata, row_bytes, kRows, kK, activation.p, t,
                                      gelu_out, nullptr);
            check(cudaDeviceSynchronize(), "vector gelu");
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
            failures += run_check("vector gelu T=" + std::to_string(t), to_float(host), gated);

            std::vector<double> geglu(plain.size());
            for (std::size_t i = 0; i < geglu.size(); ++i) { geglu[i] = gelu(plain[i]) * upper[i]; }
            ops::gguf::vector_swiglu(type, blocks.p, row_bytes, kRows, kK, activation.p, t,
                                     static_cast<__nv_bfloat16*>(out_device.p), kRows, nullptr,
                                     ops::gguf::GateActivation::GeluTanh);
            check(cudaDeviceSynchronize(), "vector geglu");
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
            failures += run_check("swiglu gelu T=" + std::to_string(t), to_float(host), geglu);

            // The kernels load a block's 32 values at once: a misaligned activation is refused.
            bool refused = false;
            try {
                ops::gguf::vector_product(type, blocks.p, row_bytes, kRows, kK,
                                          static_cast<const std::uint8_t*>(activation.p) + 16, t,
                                          f32_out, nullptr);
            } catch (const std::invalid_argument&) { refused = true; }
            if (!refused) {
                std::cerr << "FAIL " << c.name << " vector product with a 16-byte aligned activation"
                          << " T=" << t << " was not refused\n";
                ++failures;
            }
        }
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

// The MXFP8 criterion. e4m3 keeps 3 mantissa bits, so rounding an operand errs by at most 2^-4
// relative (about 2.6% RMS over uniformly spread mantissas) and a product of two rounded operands
// by about 3.7% relative RMS (measured 3.4-3.8% at the 27B's shapes). The bound allows the worst
// mantissa distribution (5.1%) with margin; the largest single error is capped at 15% of the
// largest output. A non-finite output fails (its RMS is not finite).
constexpr double kMxfp8RelativeRms = 6e-2;
constexpr double kMxfp8RelativeMax = 1.5e-1;

// The NVFP4 criterion. e2m1 keeps one mantissa bit: under its E4M3 scale per 16 values (the least
// squared error of four candidates) rounding an operand errs about 10% RMS and a product of two
// rounded operands about 14%; measured 0.10-0.18 relative RMS over the 27B's shapes and the
// eleven types, the largest single error 0.21 of the largest output.
constexpr double kNvfp4RelativeRms = 2.5e-1;
constexpr double kNvfp4RelativeMax = 3.5e-1;

// The arithmetic a check holds a product to.
enum class Form { Q8, Mxfp8, Nvfp4 };

// The wide products of the types the tile kernel serves, in either of its arithmetics.
//
// q8_1 (the six i-quants, whose matrix-kernel activation also carries an FP32 d per 32 values)
// against its own arithmetic: x quantized on the host as the activation is (d = max|x| / 127 in
// FP32, each value rounded to x / d) times the exactly dequantized weights, in FP64. The matrix and
// tile kernels then differ from it by FP32 accumulation only: the FP32 plane is held to 2e-5 of the
// largest output, the BF16 forms to BF16 rounding.
//
// MXFP8 (the i-quants, Q2_K and Q4_K under an A8 policy, a 120a build) against the FP64 product of
// the BF16 activation and the exactly dequantized weights, under the MXFP8 criterion: the form
// rounds both operands to e4m3 under power-of-two scales per 32 values.
//
// NVFP4 (the eleven types with the two-pass NVFP4 form under an A4 policy) against the same FP64
// product under the NVFP4 criterion: the two-pass product directly at every width, and from
// gguf::lt_min_columns the Ops' BF16 products, which take it. gguf_max_abs, whose max |w| the form's
// weight global needs, is checked exactly against the dequantized rows, and a matrix without it
// refused.
//
// The Ops take whichever route gguf::takes_tile_kernel picks; the tile kernel is also run directly
// in every epilogue form. 27B widths (K = 5120, 6144 and 17408 take the two-scale types' static-K
// q8_1 forms, 2560 the general one), a partial last tile of columns, and a zero 32-value group
// (d = 0, or 2^0 in MXFP8).
int run_tile_oracle(const TypeCase& c, std::uint32_t seed, std::int32_t k, std::int32_t kRows,
                    const std::vector<std::int32_t>& widths, Form form) {
    const std::int32_t max_t = *std::max_element(widths.begin(), widths.end());
    std::mt19937 rng(seed);
    int failures = 0;
    // An A4 policy admits the MXFP8 tile form too.
    const bool mxfp8  = form != Form::Q8;
    const auto policy = form == Form::Nvfp4   ? ops::LinearPolicy::AllowA4
                        : form == Form::Mxfp8 ? ops::LinearPolicy::AllowA8
                                              : ops::LinearPolicy::A16Only;
    const char* label_form = form == Form::Nvfp4 ? "nvfp4 " : mxfp8 ? "mxfp8 " : "q8 ";
    const auto type         = ops::detail::gguf_type(c.qtype);

    const auto host_blocks = random_blocks(c, 2 * kRows, k, rng);
    DeviceBuffer blocks(host_blocks.size() + 4096);
    blocks.fill(0);
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const auto block             = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    Weight gate                  = gguf_weight(c.qtype, blocks.p, kRows, k);
    Weight up                    = gate;
    up.qdata = static_cast<const std::byte*>(blocks.p) + kRows * row_bytes;

    DeviceBuffer exact_device(std::size_t(2) * kRows * k * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, k, nullptr,
                               2 * kRows, static_cast<float*>(exact_device.p), k, nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * k);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));

    if (form == Form::Nvfp4) {
        DeviceBuffer words(2 * sizeof(unsigned));
        ops::gguf_max_abs(gate, static_cast<unsigned*>(words.p), nullptr);
        ops::gguf_max_abs(up, static_cast<unsigned*>(words.p) + 1, nullptr);
        check(cudaDeviceSynchronize(), "gguf_max_abs");
        float amax[2];
        words.copy_to_host(amax, sizeof amax);
        for (int half = 0; half < 2; ++half) {
            float want = 0.0f;
            for (std::size_t i = std::size_t(half) * kRows * k; i < std::size_t(half + 1) * kRows * k;
                 ++i) {
                want = std::max(want, std::fabs(exact[i]));
            }
            // The decoders may round a value's last bit differently from ggml's dequantization.
            if (!(std::fabs(amax[half] - want) <= 1e-6f * want)) {
                std::cerr << "FAIL " << c.name << " K=" << k << " gguf_max_abs " << amax[half]
                          << " against " << want << '\n';
                ++failures;
            }
        }
        gate.gguf_amax = amax[0];
        up.gguf_amax   = amax[1];
    }

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

    // A down matrix [kDownRows, kRows] over the gated rows, for linear_swiglu_add (its own random
    // stream, so the products above keep their data).
    constexpr std::int32_t kDownRows = 384;
    std::mt19937 down_rng(seed ^ 0x5a5a5a5au);
    const bool down_fits = kRows % block.elements == 0;
    const auto down_host = down_fits ? random_blocks(c, kDownRows, kRows, down_rng)
                                     : std::vector<std::uint8_t>{};
    DeviceBuffer down_blocks(down_host.size() + 4096);
    down_blocks.fill(0);
    if (down_fits) { down_blocks.copy_from_host(down_host.data(), down_host.size()); }
    Weight down = gguf_weight(c.qtype, down_blocks.p, kDownRows, down_fits ? kRows : k);
    if (form == Form::Nvfp4 && down_fits) {
        DeviceBuffer word(sizeof(unsigned));
        ops::gguf_max_abs(down, static_cast<unsigned*>(word.p), nullptr);
        check(cudaDeviceSynchronize(), "gguf_max_abs down");
        word.copy_to_host(&down.gguf_amax, sizeof(float));
    }

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
                    xq[std::size_t(col) * k + g + i] = mxfp8 ? double(v[i]) : double(q) * double(d);
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
    const std::size_t swiglu_add =
        down_fits ? ops::linear_swiglu_add_workspace_capacity_bytes(c.qtype, &c.qtype, kRows, k,
                                                                    c.qtype, kDownRows, policy,
                                                                    policy, 1, max_t)
                  : 0;
    const std::size_t capacity = std::max(
        {ops::detail::gguf_project_workspace_bytes({&shape, 1}, policy, 1, max_t),
         ops::linear_workspace_capacity_bytes(c.qtype, kRows, k, policy, 1, max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, k, policy, 1,
                                                          max_t),
         swiglu_add});
    WorkspaceArena workspace(capacity + (1U << 20));

    const auto check_as = [&](Form arithmetic, const std::string& label,
                              const std::vector<float>& got, const std::vector<double>& want,
                              bool bf16) {
        const Error e = compare(got, want);
        const bool ok = arithmetic == Form::Nvfp4 ? e.relative_rms < kNvfp4RelativeRms &&
                                                        e.relative_max < kNvfp4RelativeMax
                        : arithmetic == Form::Mxfp8 ? e.relative_rms < kMxfp8RelativeRms &&
                                                          e.relative_max < kMxfp8RelativeMax
                        : bf16 ? e.relative_rms < 3e-3 && e.relative_max < 4e-3
                               : e.relative_max < 2e-5;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " K=" << k << ' ' << label_form << label
                      << " rms=" << e.relative_rms << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };
    // The tile and matrix kernels' arithmetic under this policy.
    const Form tile_form = mxfp8 ? Form::Mxfp8 : Form::Q8;
    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want, bool bf16) {
        return check_as(tile_form, label, got, want, bf16);
    };
    const auto lt_form = form == Form::Nvfp4 ? ops::gguf::LtForm::Nvfp4 : ops::gguf::LtForm::Mxfp8;
    const bool two_pass = form != Form::Q8 && ops::gguf::has_lt_product(type, k, lt_form);

    // NVFP4 scale-choice tables (ops::gguf_fp4_choices): the gate rows' under their own max |w|
    // (in pinned host memory, which the image pass reads over the bus), the pair's under the
    // larger of both (the gated product's global), and the [gate; up] parent's, which
    // linear_swiglu reads from row kRows on for the up rows. `corrupt` is the gate table with
    // every choice replaced by another candidate.
    const float pair_amax  = std::max(gate.gguf_amax, up.gguf_amax);
    Weight parent          = gguf_weight(c.qtype, blocks.p, 2 * kRows, k);
    parent.gguf_amax       = pair_amax;
    const bool choices     = form == Form::Nvfp4 && two_pass;
    const std::size_t rows_table = choices ? ops::gguf_fp4_choice_bytes(gate, policy) : 0;
    const std::size_t pair_table = choices ? ops::gguf_fp4_choice_bytes(parent, policy) : 0;
    std::uint8_t* gate_choices   = nullptr;
    DeviceBuffer up_choices(std::max<std::size_t>(rows_table, 1));
    DeviceBuffer pair_gate_choices(std::max<std::size_t>(rows_table, 1));
    DeviceBuffer parent_choices(std::max<std::size_t>(pair_table, 1));
    DeviceBuffer corrupt_choices(std::max<std::size_t>(rows_table, 1));
    if (choices) {
        if (rows_table == 0 || rows_table != ops::gguf::fp4_choice_bytes(kRows, k) ||
            pair_table != ops::gguf::fp4_choice_bytes(2 * kRows, k)) {
            std::cerr << "FAIL " << c.name << " K=" << k << " nvfp4 choice table bytes "
                      << rows_table << " / " << pair_table << '\n';
            ++failures;
        }
        check(cudaHostAlloc(reinterpret_cast<void**>(&gate_choices), rows_table,
                            cudaHostAllocPortable),
              "pinned choice table");
        ops::gguf_fp4_choices(gate, gate.gguf_amax, gate_choices, nullptr);
        ops::gguf_fp4_choices(gate, pair_amax, static_cast<std::uint8_t*>(pair_gate_choices.p),
                              nullptr);
        ops::gguf_fp4_choices(up, pair_amax, static_cast<std::uint8_t*>(up_choices.p), nullptr);
        ops::gguf_fp4_choices(parent, pair_amax, static_cast<std::uint8_t*>(parent_choices.p),
                              nullptr);
        check(cudaDeviceSynchronize(), "gguf_fp4_choices");
        std::vector<std::uint8_t> flipped(gate_choices, gate_choices + rows_table);
        for (auto& b : flipped) { b = static_cast<std::uint8_t>(~b); }
        corrupt_choices.copy_from_host(flipped.data(), flipped.size());
    }

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {k, t});
        const auto plain = slice(both, 2 * kRows, 0, t);
        const auto upper = slice(both, 2 * kRows, kRows, t);
        const std::string at = "T=" + std::to_string(t);
        const auto* xb       = static_cast<const __nv_bfloat16*>(x_device.p);
        // NVFP4 holds the Ops to its arithmetic where they take its two-pass form, and leaves the
        // narrower widths and the FP32 plane (the MXFP8 and q8_1 forms) to those runs.
        const bool nvfp4_ops = form == Form::Nvfp4 && two_pass && t >= ops::gguf::lt_min_columns(ops::gguf::LtForm::Nvfp4);
        const bool ops_run   = form != Form::Nvfp4 || nvfp4_ops;
        const Form ops_form  = nvfp4_ops ? Form::Nvfp4 : tile_form;

        DeviceBuffer plane_device(std::size_t(kRows) * t * sizeof(float));
        std::vector<float> plane(std::size_t(kRows) * t);
        ops::detail::GgufProduct product;
        product.f32      = static_cast<float*>(plane_device.p);
        product.f32_rows = kRows;
        for (const bool gather : {false, true}) {
            if (form == Form::Nvfp4) { break; }
            product.weight = gather ? &gathered : &gate;
            ops::detail::gguf_project(x, {&product, 1}, policy, workspace, nullptr);
            check(cudaDeviceSynchronize(), "gguf_project");
            plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
            failures += run_check((gather ? "gathered fp32 " : "fp32 ") + at, plane,
                                  gather ? slice(permuted, kRows, 0, t) : plain, false);
        }

        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);
        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }
        if (ops_run) {
            ops::linear(x, gate, out, policy, workspace, nullptr);
            check(cudaDeviceSynchronize(), "linear");
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
            failures += check_as(ops_form, "linear " + at, to_float(host), plain, true);

            out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
            ops::linear_add(x, gate, out, policy, workspace, nullptr);
            check(cudaDeviceSynchronize(), "linear_add");
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
            failures += check_as(ops_form, "linear_add " + at, to_float(host), sum, true);

            std::vector<double> swiglu(plain.size());
            for (std::size_t i = 0; i < swiglu.size(); ++i) {
                swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
            }
            ops::linear_swiglu(x, gate, up, out, policy, workspace, nullptr);
            check(cudaDeviceSynchronize(), "linear_swiglu pair");
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
            failures += check_as(ops_form, "swiglu pair " + at, to_float(host), swiglu, true);
        }
        if (nvfp4_ops) {
            // A matrix allowed A4 activations without its measured max |w| is refused.
            Weight unmeasured     = gate;
            unmeasured.gguf_amax  = 0.0F;
            bool refused          = false;
            try {
                ops::linear(x, unmeasured, out, policy, workspace, nullptr);
            } catch (const std::invalid_argument&) { refused = true; }
            if (!refused) {
                std::cerr << "FAIL " << c.name << " K=" << k << " nvfp4 linear without max |w| "
                          << at << " was not refused\n";
                ++failures;
            }
        }
        if (nvfp4_ops && down_fits) {
            // linear_swiglu_add is linear_swiglu then linear_add of its BF16 output, byte for
            // byte, where the gate kernel encodes down's NVFP4 activation as well.
            DeviceBuffer h_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
            DeviceBuffer composed_device(std::size_t(kDownRows) * t * sizeof(__nv_bfloat16));
            DeviceBuffer fused_device(std::size_t(kDownRows) * t * sizeof(__nv_bfloat16));
            std::vector<__nv_bfloat16> start(std::size_t(kDownRows) * t);
            for (auto& v : start) { v = __float2bfloat16(normal(down_rng)); }
            composed_device.copy_from_host(start.data(), start.size() * sizeof(__nv_bfloat16));
            fused_device.copy_from_host(start.data(), start.size() * sizeof(__nv_bfloat16));
            Tensor h(h_device.p, DType::BF16, {kRows, t});
            Tensor composed(composed_device.p, DType::BF16, {kDownRows, t});
            Tensor fused(fused_device.p, DType::BF16, {kDownRows, t});
            ops::linear_swiglu(x, gate, up, h, policy, workspace, nullptr);
            ops::linear_add(h, down, composed, policy, workspace, nullptr);
            ops::linear_swiglu_add(x, gate, &up, down, fused, policy, policy, workspace, nullptr);
            check(cudaDeviceSynchronize(), "linear_swiglu_add");
            std::vector<__nv_bfloat16> a(start.size()), b(start.size());
            composed_device.copy_to_host(a.data(), a.size() * sizeof(__nv_bfloat16));
            fused_device.copy_to_host(b.data(), b.size() * sizeof(__nv_bfloat16));
            if (std::memcmp(a.data(), b.data(), a.size() * sizeof(__nv_bfloat16)) != 0) {
                std::cerr << "FAIL " << c.name << " K=" << k << " nvfp4 linear_swiglu_add " << at
                          << " differs from linear_swiglu + linear_add\n";
                ++failures;
            }
        }

        // The tile kernel itself, whichever route the products above took, in each epilogue form
        // it compiles: FP32 and BF16 Store, BF16 Accumulate, BF16 SiLU GateProduct, and the
        // general one (FP32 Accumulate, GELU GateProduct).
        if (form != Form::Nvfp4 && ops::gguf::has_tile_kernel(type, k, mxfp8)) {
            DeviceBuffer activation(ops::gguf::tile_activation_bytes(k, t));
            DeviceBuffer gathered_activation(ops::gguf::tile_activation_bytes(k, t));
            ops::gguf::quantize_tile_activation(xb, k, t, nullptr, mxfp8, activation.p, nullptr);
            ops::gguf::quantize_tile_activation(xb, k, t, gathered.input_columns, mxfp8,
                                                gathered_activation.p, nullptr);
            const auto tile = [&](const void* weight, const DeviceBuffer& act,
                                  const ops::gguf::VectorOutput& o) {
                ops::gguf::tile_product(type, weight, row_bytes, kRows, k, act.p, mxfp8, t, o,
                                        nullptr);
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

        // The two-pass product itself in the form under test, in passes of 128 rows (a product
        // of 168 rows ends with a partial one): BF16 Store (plain and gathered), Accumulate, and
        // the gated pair in both activations over one activation and over distinct gate and up
        // gathers.
        if (!two_pass) { continue; }
        const Form lt_arithmetic = form;
        DeviceBuffer lt_plain(ops::gguf::lt_activation_bytes(k, t, lt_form));
        DeviceBuffer lt_gathered(ops::gguf::lt_activation_bytes(k, t, lt_form));
        ops::gguf::quantize_lt_activation(xb, k, t, nullptr, lt_form, lt_plain.p, nullptr);
        ops::gguf::quantize_lt_activation(xb, k, t, gathered.input_columns, lt_form,
                                          lt_gathered.p, nullptr);
        const std::size_t gated_scratch = ops::gguf::lt_scratch_bytes(k, 128, t, true, lt_form);
        const std::size_t plain_scratch = ops::gguf::lt_scratch_bytes(k, 128, t, false, lt_form);
        DeviceBuffer scratch(gated_scratch);
        ops::gguf::VectorOutput lt_out;
        lt_out.bf16          = static_cast<__nv_bfloat16*>(out_device.p);
        lt_out.column_stride = kRows;
        const auto lt = [&](const DeviceBuffer& act) {
            ops::gguf::lt_product(type, blocks.p, row_bytes, kRows, k, lt_form, gate.gguf_amax,
                                  {}, act.p, t, lt_out, scratch.p, plain_scratch, nullptr);
            check(cudaDeviceSynchronize(), "lt_product");
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        };
        for (const bool gather : {false, true}) {
            lt(gather ? lt_gathered : lt_plain);
            failures += check_as(lt_arithmetic,
                                 (gather ? "two-pass gathered bf16 " : "two-pass bf16 ") + at,
                                 to_float(host), gather ? slice(permuted, kRows, 0, t) : plain,
                                 true);
        }
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        lt_out.epilogue = ops::gguf::Epilogue::Accumulate;
        lt(lt_plain);
        failures +=
            check_as(lt_arithmetic, "two-pass bf16 accumulate " + at, to_float(host), sum, true);
        const auto permuted_gate = slice(permuted, kRows, 0, t);
        for (const bool distinct : {false, true}) {
            for (const auto act :
                 {ops::gguf::GateActivation::Silu, ops::gguf::GateActivation::GeluTanh}) {
                // distinct: the gate rows over the gathered activation, the up rows the plain one.
                ops::gguf::lt_gated_product(type, blocks.p, row_bytes, type, up.qdata, row_bytes,
                                            kRows, k, lt_form,
                                            std::max(gate.gguf_amax, up.gguf_amax), {}, {},
                                            distinct ? lt_gathered.p : lt_plain.p,
                                            distinct ? lt_plain.p : nullptr, t,
                                            static_cast<__nv_bfloat16*>(out_device.p), kRows, act,
                                            nullptr, scratch.p, gated_scratch, nullptr);
                check(cudaDeviceSynchronize(), "lt_gated_product");
                out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
                std::vector<double> gated(plain.size());
                for (std::size_t i = 0; i < gated.size(); ++i) {
                    const double g = distinct ? permuted_gate[i] : plain[i];
                    const double a =
                        act == ops::gguf::GateActivation::Silu
                            ? g / (1.0 + std::exp(-g))
                            : 0.5 * g *
                                  (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
                    gated[i] = a * upper[i];
                }
                failures += check_as(lt_arithmetic,
                                     std::string("two-pass gated ") +
                                         (act == ops::gguf::GateActivation::Silu ? "silu "
                                                                                 : "gelu ") +
                                         (distinct ? "distinct gathers " : "") + at,
                                     to_float(host), gated, true);
            }
        }
        // The gated product's NVFP4 activation of its own output (for the next product) is that
        // output's quantize_lt_activation, byte for byte: e2m1 codes, E4M3 scales (with the
        // padding columns) and the max |x| word after them.
        if (lt_form == ops::gguf::LtForm::Nvfp4 && kRows % 256 == 0) {
            const std::size_t bytes = ops::gguf::lt_activation_bytes(kRows, t, lt_form);
            DeviceBuffer fused(bytes), separate(bytes);
            check(cudaMemset(fused.p, 0xAB, bytes), "fill");
            check(cudaMemset(separate.p, 0xCD, bytes), "fill");
            auto* gated_out = static_cast<__nv_bfloat16*>(out_device.p);
            ops::gguf::lt_gated_product(type, blocks.p, row_bytes, type, up.qdata, row_bytes, kRows,
                                        k, lt_form, std::max(gate.gguf_amax, up.gguf_amax), {},
                                        {}, lt_plain.p, nullptr, t, gated_out, kRows,
                                        ops::gguf::GateActivation::Silu, fused.p, scratch.p,
                                        gated_scratch, nullptr);
            ops::gguf::quantize_lt_activation(gated_out, kRows, t, nullptr, lt_form, separate.p,
                                              nullptr);
            check(cudaDeviceSynchronize(), "lt_gated_product activation");
            std::vector<std::uint8_t> a(bytes), b(bytes);
            fused.copy_to_host(a.data(), bytes);
            separate.copy_to_host(b.data(), bytes);
            const std::size_t codes  = std::size_t(t) * kRows / 2;
            const std::size_t offset = ops::gguf::lt_activation_scales_offset(kRows, t, lt_form);
            const std::size_t scales =
                std::size_t((t + 127) / 128 * 128) * ((kRows / 16 + 3) / 4 * 4);
            const bool same = std::memcmp(a.data(), b.data(), codes) == 0 &&
                              std::memcmp(a.data() + offset, b.data() + offset, scales + 4) == 0;
            if (!same) {
                std::cerr << "FAIL two-pass gated nvfp4 activation " << at << '\n';
                ++failures;
            }
        }
        // A product given its weight's choice table under its own global is the searching
        // product byte for byte; one under another global it ignores, and one under its global it
        // reads (the corrupted table then changes the product).
        if (choices) {
            const auto bytes_of = [&] {
                std::vector<__nv_bfloat16> v(host.size());
                out_device.copy_to_host(v.data(), v.size() * sizeof(__nv_bfloat16));
                return v;
            };
            const auto same = [](const std::vector<__nv_bfloat16>& a,
                                 const std::vector<__nv_bfloat16>& b) {
                return std::memcmp(a.data(), b.data(), a.size() * sizeof(__nv_bfloat16)) == 0;
            };
            const auto single = [&](const ops::gguf::Fp4Choices& table) {
                lt_out.epilogue = ops::gguf::Epilogue::Store;
                ops::gguf::lt_product(type, blocks.p, row_bytes, kRows, k, lt_form, gate.gguf_amax,
                                      table, lt_plain.p, t, lt_out, scratch.p, plain_scratch,
                                      nullptr);
                check(cudaDeviceSynchronize(), "lt_product choices");
                return bytes_of();
            };
            const auto* corrupt = static_cast<const std::uint8_t*>(corrupt_choices.p);
            const auto searched = single({});
            const bool table_same   = same(single({gate_choices, gate.gguf_amax}), searched);
            const bool other_same   = same(single({corrupt, 4.0f * gate.gguf_amax}), searched);
            const bool corrupt_read = !same(single({corrupt, gate.gguf_amax}), searched);
            const auto gated = [&](const ops::gguf::Fp4Choices& g, const ops::gguf::Fp4Choices& u) {
                ops::gguf::lt_gated_product(type, blocks.p, row_bytes, type, up.qdata, row_bytes,
                                            kRows, k, lt_form, pair_amax, g, u, lt_plain.p,
                                            nullptr, t, static_cast<__nv_bfloat16*>(out_device.p),
                                            kRows, ops::gguf::GateActivation::Silu, nullptr,
                                            scratch.p, gated_scratch, nullptr);
                check(cudaDeviceSynchronize(), "lt_gated_product choices");
                return bytes_of();
            };
            const bool gated_same =
                same(gated({static_cast<const std::uint8_t*>(pair_gate_choices.p), pair_amax},
                           {static_cast<const std::uint8_t*>(up_choices.p), pair_amax}),
                     gated({}, {}));
            bool parent_same = true;
            if (nvfp4_ops) {
                const auto fused = [&](const std::uint8_t* table) {
                    Weight w           = parent;
                    w.fp4_choices      = table;
                    w.fp4_choices_amax = pair_amax;
                    ops::linear_swiglu(x, w, out, policy, workspace, nullptr);
                    check(cudaDeviceSynchronize(), "linear_swiglu choices");
                    return bytes_of();
                };
                parent_same = same(fused(static_cast<const std::uint8_t*>(parent_choices.p)),
                                   fused(nullptr));
            }
            if (!table_same || !other_same || !corrupt_read || !gated_same || !parent_same) {
                std::cerr << "FAIL " << c.name << " K=" << k << " nvfp4 choice tables " << at
                          << ": table " << table_same << " other global " << other_same
                          << " corrupt read " << corrupt_read << " gated " << gated_same
                          << " parent " << parent_same << '\n';
                ++failures;
            }
        }
    }
    if (gate_choices != nullptr) { check(cudaFreeHost(gate_choices), "free choice table"); }
    if (workspace.peak_used() > capacity) {
        std::cerr << "FAIL " << c.name << " K=" << k << ' ' << label_form << "workspace peak "
                  << workspace.peak_used() << " bytes over the planned " << capacity << '\n';
        ++failures;
    }
    return failures;
}

// The narrow kernel (gguf::narrow_product, up to 32 columns of six types on sm_120) against its
// arithmetic, the vector kernel's: x quantized on the host as its activation is (d = max|x| / 127
// in FP32, each value x / d rounded, d kept in FP16) times the exactly dequantized weights, in FP64.
// The kernel differs from it by FP32 accumulation only: the FP32 plane is held to 2e-5 of the
// largest output, the BF16 forms to BF16 rounding. The kernel directly at every width (also the
// narrowest, which the Ops leave to the vector kernel) in each epilogue form and over a gathered
// input; the Ops at the widths takes_narrow_kernel gives it (gguf_project's FP32 plane, linear,
// linear_add, the gated pair and the [gate; up] parent), with their workspace plan. The 27B's K,
// rows that leave a partial unit (168), a weight that starts 2 bytes past a 16-byte boundary, and a
// zero 32-value group (d = 0).
int run_narrow_oracle(const TypeCase& c, std::uint32_t seed, std::int32_t k, std::int32_t kRows,
                      const std::vector<std::int32_t>& widths) {
    const std::int32_t max_t = *std::max_element(widths.begin(), widths.end());
    std::mt19937 rng(seed);
    int failures    = 0;
    const auto type = ops::detail::gguf_type(c.qtype);

    const auto host_blocks = random_blocks(c, 2 * kRows, k, rng);
    DeviceBuffer blocks(host_blocks.size() + 4096);
    blocks.fill(0);
    blocks.copy_from_host(host_blocks.data(), host_blocks.size(), 2);
    const auto* weight           = static_cast<const std::uint8_t*>(blocks.p) + 2;
    const auto block             = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const Weight gate            = gguf_weight(c.qtype, weight, kRows, k);
    Weight up                    = gate;
    up.qdata                     = weight + kRows * row_bytes;
    const Weight both            = gguf_weight(c.qtype, weight, 2 * kRows, k);

    // The reference dequantizes an aligned copy (the dequantizer requires alignment).
    DeviceBuffer aligned(host_blocks.size());
    aligned.copy_from_host(host_blocks.data(), host_blocks.size());
    DeviceBuffer exact_device(std::size_t(2) * kRows * k * sizeof(float));
    ops::gguf::dequantize_rows(type, static_cast<const std::uint8_t*>(aligned.p), row_bytes, k,
                               nullptr, 2 * kRows, static_cast<float*>(exact_device.p), k, nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * k);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(k) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    for (int i = 0; i < 32; ++i) { x_host[std::size_t(k) + 64 + i] = __float2bfloat16(0.0f); }
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));
    const auto* xb = static_cast<const __nv_bfloat16*>(x_device.p);

    std::vector<std::int32_t> order(k);
    for (std::int32_t i = 0; i < k; ++i) { order[i] = i; }
    std::shuffle(order.begin(), order.end(), rng);
    DeviceBuffer order_device(order.size() * sizeof(std::int32_t));
    order_device.copy_from_host(order.data(), order.size() * sizeof(std::int32_t));
    const auto* order_p    = static_cast<const std::int32_t*>(order_device.p);
    Weight gathered        = gate;
    gathered.input_columns = order_p;

    // out[col][r] over rows [0, count) for every column, x gathered through `order`.
    const auto oracle = [&](std::int32_t count, bool permuted) {
        std::vector<double> xq(std::size_t(k) * max_t);
        for (std::int32_t col = 0; col < max_t; ++col) {
            for (std::int32_t g = 0; g < k; g += 32) {
                float v[32];
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    v[i] = __bfloat162float(x_host[std::size_t(col) * k + (permuted ? order[g + i] : g + i)]);
                    amax = std::max(amax, std::fabs(v[i]));
                }
                const float d       = amax / 127.0f;
                const double scale  = double(__half2float(__float2half(d)));
                for (int i = 0; i < 32; ++i) {
                    const float q = amax == 0.0f ? 0.0f : std::round(v[i] / d);
                    xq[std::size_t(col) * k + g + i] = double(q) * scale;
                }
            }
        }
        std::vector<double> out(std::size_t(count) * max_t);
        for (std::int32_t col = 0; col < max_t; ++col) {
            const double* x = xq.data() + std::size_t(col) * k;
            for (std::int32_t r = 0; r < count; ++r) {
                const float* w = exact.data() + std::size_t(r) * k;
                double acc     = 0;
                for (std::int32_t i = 0; i < k; ++i) { acc += double(w[i]) * x[i]; }
                out[std::size_t(col) * count + r] = acc;
            }
        }
        return out;
    };
    const auto all      = oracle(2 * kRows, false); // [col][gate rows; up rows]
    const auto permuted = oracle(kRows, true);
    const auto slice    = [&](const std::vector<double>& from, std::int32_t stride,
                           std::int32_t row0, std::int32_t t) {
        std::vector<double> out(std::size_t(kRows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < kRows; ++r) {
                out[std::size_t(col) * kRows + r] = from[std::size_t(col) * stride + row0 + r];
            }
        }
        return out;
    };

    const ops::detail::GgufShape shape{c.qtype, kRows, k};
    const std::size_t capacity = std::max(
        {ops::detail::gguf_project_workspace_bytes({&shape, 1}, ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_workspace_capacity_bytes(c.qtype, kRows, k, ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_swiglu_workspace_capacity_bytes(c.qtype, 2 * kRows, k,
                                                     ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, k,
                                                          ops::LinearPolicy::A16Only, 1, max_t)});
    WorkspaceArena workspace(capacity + (1U << 20));

    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want, bool bf16) {
        const Error e = compare(got, want);
        const bool ok = bf16 ? e.relative_rms < 3e-3 && e.relative_max < 4e-3 : e.relative_max < 2e-5;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " K=" << k << " rows=" << kRows << " narrow " << label
                      << " rms=" << e.relative_rms << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {k, t});
        const auto plain     = slice(all, 2 * kRows, 0, t);
        const auto upper     = slice(all, 2 * kRows, kRows, t);
        const std::string at = "T=" + std::to_string(t);

        DeviceBuffer plane_device(std::size_t(kRows) * t * sizeof(float));
        std::vector<float> plane(std::size_t(kRows) * t);
        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);
        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }

        // The kernel itself.
        DeviceBuffer activation(ops::gguf::narrow_activation_bytes(k, t));
        DeviceBuffer gathered_activation(ops::gguf::narrow_activation_bytes(k, t));
        ops::gguf::quantize_narrow_activation(xb, k, t, nullptr, activation.p, nullptr);
        ops::gguf::quantize_narrow_activation(xb, k, t, order_p, gathered_activation.p, nullptr);
        const auto narrow = [&](const void* w, const DeviceBuffer& act,
                                const ops::gguf::VectorOutput& o) {
            ops::gguf::narrow_product(type, w, row_bytes, kRows, k, act.p, t, o, nullptr);
            check(cudaDeviceSynchronize(), "narrow_product");
        };
        ops::gguf::VectorOutput f32_out;
        f32_out.f32           = static_cast<float*>(plane_device.p);
        f32_out.column_stride = kRows;
        for (const bool gather : {false, true}) {
            narrow(weight, gather ? gathered_activation : activation, f32_out);
            plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
            failures += run_check((gather ? "gathered fp32 " : "fp32 ") + at, plane,
                                  gather ? slice(permuted, kRows, 0, t) : plain, false);
        }
        std::vector<float> base(plane.size());
        for (auto& v : base) { v = normal(rng); }
        plane_device.copy_from_host(base.data(), base.size() * sizeof(float));
        f32_out.epilogue = ops::gguf::Epilogue::Accumulate;
        narrow(weight, activation, f32_out);
        plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
        std::vector<double> accumulated = plain;
        for (std::size_t i = 0; i < accumulated.size(); ++i) { accumulated[i] += base[i]; }
        failures += run_check("fp32 accumulate " + at, plane, accumulated, false);

        ops::gguf::VectorOutput bf16_out;
        bf16_out.bf16          = static_cast<__nv_bfloat16*>(out_device.p);
        bf16_out.column_stride = kRows;
        narrow(weight, activation, bf16_out);
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("bf16 " + at, to_float(host), plain, true);
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        bf16_out.epilogue = ops::gguf::Epilogue::Accumulate;
        narrow(weight, activation, bf16_out);
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("bf16 accumulate " + at, to_float(host), sum, true);

        // The gate products over the up rows, gated by the plain products in an FP32 plane.
        std::vector<float> gate_values(plain.begin(), plain.end());
        plane_device.copy_from_host(gate_values.data(), gate_values.size() * sizeof(float));
        const auto gelu = [](double g) {
            return 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
        };
        for (const auto act : {ops::gguf::GateActivation::Silu, ops::gguf::GateActivation::GeluTanh}) {
            std::vector<double> gated(plain.size());
            for (std::size_t i = 0; i < gated.size(); ++i) {
                const double g = gate_values[i];
                gated[i] = (act == ops::gguf::GateActivation::Silu ? g / (1.0 + std::exp(-g)) : gelu(g)) * upper[i];
            }
            ops::gguf::VectorOutput gated_out = bf16_out;
            gated_out.epilogue           = ops::gguf::Epilogue::GateProduct;
            gated_out.gate               = static_cast<const float*>(plane_device.p);
            gated_out.gate_column_stride = kRows;
            gated_out.activation         = act;
            narrow(up.qdata, activation, gated_out);
            out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
            failures += run_check(std::string(act == ops::gguf::GateActivation::Silu ? "silu " : "gelu ") +
                                      "gate product " + at,
                                  to_float(host), gated, true);
        }

        // A misaligned activation is refused.
        bool refused = false;
        try {
            ops::gguf::narrow_product(type, weight, row_bytes, kRows, k,
                                      static_cast<const std::uint8_t*>(activation.p) + 8, t,
                                      f32_out, nullptr);
        } catch (const std::invalid_argument&) { refused = true; }
        if (!refused) {
            std::cerr << "FAIL " << c.name << " narrow product with an 8-byte aligned activation "
                      << at << " was not refused\n";
            ++failures;
        }

        // The Ops where they take the kernel.
        if (!ops::gguf::takes_narrow_kernel(type, k, t)) { continue; }
        ops::detail::GgufProduct product;
        product.f32      = static_cast<float*>(plane_device.p);
        product.f32_rows = kRows;
        for (const bool gather : {false, true}) {
            product.weight = gather ? &gathered : &gate;
            ops::detail::gguf_project(x, {&product, 1}, ops::LinearPolicy::A16Only, workspace, nullptr);
            check(cudaDeviceSynchronize(), "gguf_project");
            plane_device.copy_to_host(plane.data(), plane.size() * sizeof(float));
            failures += run_check((gather ? "ops gathered fp32 " : "ops fp32 ") + at, plane,
                                  gather ? slice(permuted, kRows, 0, t) : plain, false);
        }
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        ops::linear(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("linear " + at, to_float(host), plain, true);

        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        ops::linear_add(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_add");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("linear_add " + at, to_float(host), sum, true);

        std::vector<double> swiglu(plain.size());
        for (std::size_t i = 0; i < swiglu.size(); ++i) {
            swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
        }
        ops::linear_swiglu(x, gate, up, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu pair");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu pair " + at, to_float(host), swiglu, true);

        ops::linear_swiglu(x, both, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu " + at, to_float(host), swiglu, true);
    }
    if (workspace.peak_used() > capacity) {
        std::cerr << "FAIL " << c.name << " K=" << k << " narrow workspace peak "
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
        for (const Form oracle_form : {Form::Q8, Form::Mxfp8, Form::Nvfp4}) {
            const bool mxfp8 = oracle_form != Form::Q8;
            if (mxfp8 && !ops::gguf::kTileMxfp8Built) { continue; }
            const std::string form = oracle_form == Form::Nvfp4 ? " nvfp4 oracle"
                                     : mxfp8                    ? " mxfp8 oracle"
                                                                : " q8_1 oracle";
            for (const auto& c : cases()) {
                const auto type   = ops::detail::gguf_type(c.qtype);
                if (oracle_form == Form::Nvfp4) {
                    // The two-pass product directly at the 27B's K, a partial last pass of rows,
                    // and the Ops at the narrowest width that takes it and at a prompt chunk's.
                    if (!ops::gguf::has_lt_product(type, 2560, ops::gguf::LtForm::Nvfp4)) {
                        continue;
                    }
                    for (const std::int32_t k : {2560, 5120, 6144, 17408}) {
                        const int f = run_tile_oracle(c, seed++, k, 256, {100, 200}, oracle_form);
                        std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                                  << " K=" << k << form << " (two-pass product)\n";
                        failures += f;
                    }
                    const int g = run_tile_oracle(c, seed++, 5120, 168, {129}, oracle_form);
                    const int h = run_tile_oracle(c, seed++, 2560, 256, {ops::gguf::lt_min_columns(ops::gguf::LtForm::Nvfp4), 1024},
                                                  oracle_form);
                    std::cout << (g == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                              << " K=5120 rows=168" << form << " (two-pass product)\n"
                              << (h == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                              << " K=2560 T=" << ops::gguf::lt_min_columns(ops::gguf::LtForm::Nvfp4) << ",1024"
                              << form << " (two-pass route)\n";
                    failures += g + h;
                    continue;
                }
                const bool iquant =
                    c.qtype == QType::GGUF_IQ3_XXS || c.qtype == QType::GGUF_IQ3_S ||
                    c.qtype == QType::GGUF_IQ2_XXS || c.qtype == QType::GGUF_IQ4_XS ||
                    c.qtype == QType::GGUF_IQ2_XS || c.qtype == QType::GGUF_IQ2_S;
                const bool kquant = c.qtype == QType::GGUF_Q2_K || c.qtype == QType::GGUF_Q4_K;
                if (!iquant && !(mxfp8 && kquant)) { continue; }
                for (const std::int32_t k : {2560, 5120, 6144, 17408}) {
                    const bool tile = ops::gguf::has_tile_kernel(type, k, mxfp8);
                    const int f =
                        run_tile_oracle(c, seed++, k, 256, {100, 128, 200}, oracle_form);
                    std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                              << form << (tile ? " (and the tile kernel)" : "") << '\n';
                    failures += f;
                    // Rows the matrix kernel cannot take (168 = 128 + 40) take the tile kernel.
                    if (tile && k == 5120) {
                        const int g = run_tile_oracle(c, seed++, k, 168, {129, 200}, oracle_form);
                        std::cout << (g == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                                  << " K=" << k << " rows=168" << form << " (tile kernel)\n";
                        failures += g;
                    }
                }
                // From its lt_min_columns the Ops take the two-pass MXFP8 product (BF16 linear,
                // linear_add and the gated pair; the FP32 plane keeps the tile kernel).
                if (mxfp8 && ops::gguf::has_lt_product(type, 2560, ops::gguf::LtForm::Mxfp8)) {
                    const int f = run_tile_oracle(c, seed++, 2560, 256, {ops::gguf::lt_min_columns(ops::gguf::LtForm::Mxfp8)},
                                                  oracle_form);
                    std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                              << " K=2560 T=" << ops::gguf::lt_min_columns(ops::gguf::LtForm::Mxfp8) << form
                              << " (two-pass route)\n";
                    failures += f;
                }
                // A grid that fills whole waves of the device takes the tile kernel through the
                // Ops: their activation sharing, input gather and workspace plan.
                const int rows = ops::gguf::multiprocessor_count() * 128;
                if (ops::gguf::takes_tile_kernel(type, rows, 2560, ops::gguf::kTileMinColumns,
                                                 mxfp8)) {
                    const int f = run_tile_oracle(c, seed++, 2560, rows,
                                                  {ops::gguf::kTileMinColumns, 128, 200},
                                                  oracle_form);
                    std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                              << " K=2560 rows=" << rows << form << " (tile route)\n";
                    failures += f;
                }
            }
        }
        // The narrow kernel at the 27B's K, and at rows that reach each width's configuration of
        // every warp count (narrow_row_warps: rows / SMs decides it).
        const std::vector<std::int32_t> narrow_widths{1, 3, 5, 8, 9, 13, 16, 17, 24, 25, 32};
        for (const auto& c : cases()) {
            const auto type = ops::detail::gguf_type(c.qtype);
            if (!ops::gguf::has_narrow_kernel(type, 5120)) { continue; }
            std::vector<std::pair<std::int32_t, std::int32_t>> runs; // (k, rows)
            for (const std::int32_t k : {5120, 6144, 17408}) {
                for (const std::int32_t rows : {256, 168}) { runs.emplace_back(k, rows); }
            }
            bool reached[4][3] = {};
            const auto reach = [&](std::int32_t rows, bool mark) {
                bool added = false;
                for (const std::int32_t t : narrow_widths) {
                    bool& r = reached[(t + 7) / 8 - 1][ops::gguf::narrow_row_warps(type, rows, t) - 2];
                    added |= !r;
                    r |= mark;
                }
                return added;
            };
            (void)reach(256, true);
            (void)reach(168, true);
            const int sms = ops::gguf::multiprocessor_count();
            for (const int per_sm : {50, 84, 168, 336}) {
                if (reach(sms * per_sm + 8, false)) {
                    (void)reach(sms * per_sm + 8, true);
                    runs.emplace_back(5120, sms * per_sm + 8);
                }
            }
            for (int nt = 0; nt < 4; ++nt) {
                for (int tw = 0; tw < 3; ++tw) {
                    if (!reached[nt][tw]) {
                        std::cerr << "FAIL " << c.name << " narrow kernel: no tested rows reach NT "
                                  << nt + 1 << " TW " << tw + 2 << "\n";
                        ++failures;
                    }
                }
            }
            for (const auto& [k, rows] : runs) {
                const int f = run_narrow_oracle(c, seed++, k, rows, narrow_widths);
                std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                          << " rows=" << rows << " narrow kernel\n";
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
