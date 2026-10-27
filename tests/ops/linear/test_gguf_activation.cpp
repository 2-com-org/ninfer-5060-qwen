// The norms that also write a GGUF product's NVFP4 input activation (ninfer/ops/gguf_activation.h)
// against the separate path they replace: the norm's BF16 output and the activation bytes
// quantize_lt_activation forms from it (e2m1 codes, E4M3 scales with the padding columns, and the
// max |x| word after them), ungathered or under a consumer's input gather, are identical. Both norms' arithmetic is qualified by their own tests
// against an FP64 oracle; the NVFP4 encode by the GGUF Linear test.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gguf_activation.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

constexpr float kEps = 1e-6f;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// Normal values with a few large outliers per row, so groups take different scales and some
// reach the re-encode boundary of the block maximum.
std::vector<__nv_bfloat16> random_values(std::size_t count, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, sigma);
    std::uniform_int_distribution<int> outlier(0, 499);
    std::vector<__nv_bfloat16> values(count);
    for (auto& v : values) {
        const float x = normal(rng);
        v             = __float2bfloat16(outlier(rng) == 0 ? 40.0f * x : x);
    }
    return values;
}

DeviceBuffer upload(const std::vector<__nv_bfloat16>& values) {
    DeviceBuffer buffer(values.size() * sizeof(__nv_bfloat16));
    buffer.copy_from_host(values.data(), buffer.bytes);
    return buffer;
}

// The consumed bytes of two activations of a [k, t] input agree, and so do the two outputs.
int compare(const std::string& at, const DeviceBuffer& out, const DeviceBuffer& reference_out,
            const DeviceBuffer& activation, const DeviceBuffer& reference, std::int32_t k,
            std::int32_t t) {
    check(cudaDeviceSynchronize(), at.c_str());
    int failures = 0;
    std::vector<std::uint8_t> a(out.bytes), b(out.bytes);
    out.copy_to_host(a.data(), a.size());
    reference_out.copy_to_host(b.data(), b.size());
    if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
        std::cerr << "FAIL " << at << ": the norm output differs\n";
        ++failures;
    }
    const auto form = ops::gguf::LtForm::Nvfp4;
    a.resize(activation.bytes);
    b.resize(activation.bytes);
    activation.copy_to_host(a.data(), a.size());
    reference.copy_to_host(b.data(), b.size());
    const std::size_t codes  = std::size_t(t) * k / 2;
    const std::size_t offset = ops::gguf::lt_activation_scales_offset(k, t, form);
    const std::size_t scales = std::size_t((t + 127) / 128 * 128) * ((k / 16 + 3) / 4 * 4);
    if (std::memcmp(a.data(), b.data(), codes) != 0) {
        std::cerr << "FAIL " << at << ": the e2m1 codes differ\n";
        ++failures;
    }
    if (std::memcmp(a.data() + offset, b.data() + offset, scales + 4) != 0) {
        std::cerr << "FAIL " << at << ": the scales or the max |x| word differ\n";
        ++failures;
    }
    return failures;
}

// The hidden-state norm (unit offset) at the 27B's width.
int run_hidden(std::int32_t t, std::mt19937& rng) {
    const std::int32_t k = ops::kRmsNormGgufActivationWidth;
    const auto x_host    = random_values(std::size_t(k) * t, 1.0f, rng);
    const auto w_host    = random_values(std::size_t(k), 0.3f, rng);
    const DeviceBuffer x_device = upload(x_host);
    const DeviceBuffer w_device = upload(w_host);
    const std::size_t bytes     = ops::gguf_activation_bytes(k, t);
    if (bytes != ops::gguf::lt_activation_bytes(k, t, ops::gguf::LtForm::Nvfp4)) {
        std::cerr << "FAIL hidden T=" << t << ": activation bytes\n";
        return 1;
    }
    DeviceBuffer out_device(x_device.bytes), reference_out(x_device.bytes);
    DeviceBuffer activation(bytes), reference(bytes);
    activation.fill(0xAB);
    reference.fill(0xCD);
    const Tensor x(x_device.p, DType::BF16, {k, t});
    const Tensor w(w_device.p, DType::BF16, {k});
    Tensor out(out_device.p, DType::BF16, {k, t});
    Tensor ref(reference_out.p, DType::BF16, {k, t});
    ops::rmsnorm_gguf_activation(x, w, kEps, out, activation.p, nullptr);
    ops::rmsnorm(x, w, kEps, true, ref, nullptr);
    ops::gguf::quantize_lt_activation(static_cast<const __nv_bfloat16*>(reference_out.p), k, t,
                                      nullptr, ops::gguf::LtForm::Nvfp4, reference.p, nullptr);
    return compare("hidden K=" + std::to_string(k) + " T=" + std::to_string(t), out_device,
                   reference_out, activation, reference, k, t);
}

enum class Gather { None, TiledHeads, Random };

// The converter's tiled value heads (tools/convert/gguf_blocks.py tiled_input_columns, 16 key
// heads) or a permutation of single columns.
std::vector<std::int32_t> gather_columns(Gather gather, std::int32_t heads, std::mt19937& rng) {
    const std::int32_t d = ops::kGatedRmsNormGgufActivationHead;
    std::vector<std::int32_t> columns(std::size_t(d) * heads);
    for (std::int32_t c = 0; c < d * heads; ++c) {
        const std::int32_t tiled = c / d, lane = c % d, key_heads = 16, per_key = heads / key_heads;
        columns[c] = ((tiled % key_heads) * per_key + tiled / key_heads) * d + lane;
    }
    if (gather == Gather::Random) {
        for (std::int32_t c = 0; c < d * heads; ++c) { columns[c] = c; }
        std::shuffle(columns.begin(), columns.end(), rng);
    }
    return columns;
}

// The SiLU-gated head norm over the 27B's 48 value heads of 128.
int run_gated(std::int32_t heads, std::int32_t t, Gather gather, std::mt19937& rng) {
    const std::int32_t d = ops::kGatedRmsNormGgufActivationHead;
    const std::int32_t k = d * heads;
    const auto x_host    = random_values(std::size_t(k) * t, 0.5f, rng);
    const auto z_host    = random_values(std::size_t(k) * t, 1.5f, rng);
    const auto w_host    = random_values(std::size_t(d), 0.8f, rng);
    const DeviceBuffer x_device = upload(x_host);
    const DeviceBuffer z_device = upload(z_host);
    const DeviceBuffer w_device = upload(w_host);
    const std::size_t bytes     = ops::gguf_activation_bytes(k, t);
    DeviceBuffer out_device(x_device.bytes), reference_out(x_device.bytes);
    DeviceBuffer activation(bytes), reference(bytes);
    activation.fill(0xAB);
    reference.fill(0xCD);
    DeviceBuffer columns_device;
    const std::int32_t* columns = nullptr;
    if (gather != Gather::None) {
        const auto host = gather_columns(gather, heads, rng);
        columns_device  = DeviceBuffer(host.size() * sizeof(std::int32_t));
        columns_device.copy_from_host(host.data(), columns_device.bytes);
        columns = static_cast<const std::int32_t*>(columns_device.p);
    }
    const Tensor x(x_device.p, DType::BF16, {d, heads, t});
    const Tensor z(z_device.p, DType::BF16, {d, heads, t});
    const Tensor w(w_device.p, DType::BF16, {d});
    Tensor out(out_device.p, DType::BF16, {d, heads, t});
    Tensor ref(reference_out.p, DType::BF16, {d, heads, t});
    ops::gated_rmsnorm_gguf_activation(x, w, z, kEps, out, activation.p, columns, nullptr);
    ops::gated_rmsnorm(x, w, z, ops::GateActivation::Silu, kEps, ref, nullptr);
    ops::gguf::quantize_lt_activation(static_cast<const __nv_bfloat16*>(reference_out.p), k, t,
                                      columns, ops::gguf::LtForm::Nvfp4, reference.p, nullptr);
    const char* name = gather == Gather::None         ? ""
                       : gather == Gather::TiledHeads ? " tiled heads"
                                                      : " random gather";
    return compare("gated H=" + std::to_string(heads) + " T=" + std::to_string(t) + name,
                   out_device, reference_out, activation, reference, k, t);
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "device properties");
    if (!ops::gguf::kTileMxfp8Built || properties.major < 12) {
        std::cout << "SKIP: the NVFP4 activation needs the sm_120 FP8 build on an RTX 50 device\n";
        return 77;
    }
    try {
        int failures = 0;
        std::mt19937 rng(4242);
        // The narrowest width (one column past a 128-column scale tile), a partial last scale
        // tile, the 262K prompt chunk and the widest chunk.
        for (const std::int32_t t : {ops::kGgufActivationMinTokens, 1100, 3456, 4096}) {
            failures += run_hidden(t, rng);
            failures += run_gated(48, t, Gather::None, rng);
            failures += run_gated(48, t, Gather::TiledHeads, rng);
        }
        // The fewest and the most heads, a partial scale tile, and any gather.
        failures += run_gated(2, 1031, Gather::Random, rng);
        failures += run_gated(16, 1031, Gather::TiledHeads, rng);
        failures += run_gated(96, 1100, Gather::Random, rng);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF NVFP4 norm activation\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF NVFP4 norm activation: " << error.what() << '\n';
        return 1;
    }
}
