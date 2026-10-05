// FP16 recurrent-state storage against FP32 over a long decode, through the public
// gated_delta_net Op. Both chains see identical inputs; the FP16 chain rounds its stored state
// once per call. Reports the output error per step (relative L2 over all heads) and fails if it
// exceeds 2e-2 -- five bf16 rounding steps -- anywhere, or if it trends upward (the last quarter's
// mean above twice the second quarter's).
//
// Inputs mimic the 27B's layer statistics loosely: 16 query/key heads and 48 value heads,
// normalized q/k, per-head decays from short (alpha 0.9) to near-lossless memory (alpha 0.9999),
// and beta in (0, 1).
// --oracle-only qualifies output and published FP16 state directly against the independent FP64
// recurrence at the 27B geometry. Reference carry is rounded only between public Op calls.

#include "ninfer/ops/gated_delta_net.h"
#include "core/arena.h"
#include "ops/gdn_ref.h"
#include "ops/op_tester.h"
#include "ops/small_t_oracle.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::Tensor;
namespace oracle = ninfer::test::small_t_oracle;

constexpr int kQk    = 16;
constexpr int kHv    = 48;
constexpr int kDim   = 128;
constexpr int kSteps = 4096;

// Decode binary16 without CUDA conversion routines. This also preserves signed zero and
// subnormals, which can appear at an observable state boundary.
double fp16_value(std::uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31U;
    const unsigned fraction = bits & 1023U;
    double value;
    if (exponent == 31U) {
        value = fraction == 0 ? std::numeric_limits<double>::infinity()
                              : std::numeric_limits<double>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<double>(fraction), -24);
    } else {
        value = std::ldexp(static_cast<double>(1024U + fraction), static_cast<int>(exponent) - 25);
    }
    return (bits & 0x8000U) != 0 ? -value : value;
}

// Quantize a reference state directly from FP64, rather than first rounding to a private FP32
// materialization. Ties go to the even binary16 code independently of the host rounding mode.
std::uint16_t fp16_rne(double value) {
    if (!std::isfinite(value)) { throw std::invalid_argument("non-finite reference state"); }
    const auto sign = static_cast<std::uint16_t>(std::signbit(value) ? 0x8000U : 0U);
    const double magnitude = std::abs(value);
    const auto integer_rne = [](double scaled) {
        const auto floor = static_cast<unsigned>(std::floor(scaled));
        const double remainder = scaled - static_cast<double>(floor);
        return floor + static_cast<unsigned>(remainder > 0.5 ||
                                             (remainder == 0.5 && (floor & 1U) != 0));
    };
    if (magnitude < std::ldexp(1.0, -14)) {
        return static_cast<std::uint16_t>(sign | integer_rne(std::ldexp(magnitude, 24)));
    }
    int exponent = 0;
    (void)std::frexp(magnitude, &exponent);
    --exponent;
    if (exponent > 15) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    unsigned significand = integer_rne(std::ldexp(magnitude, 10 - exponent));
    if (significand == 2048U) {
        significand = 1024U;
        ++exponent;
    }
    if (exponent > 15) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(exponent + 15) << 10) |
                                      (significand - 1024U));
}

int verify_fp16_codec() {
    for (const auto& [bits, value] :
         std::vector<std::pair<std::uint16_t, double>>{{0x0000, 0.0}, {0x8000, -0.0},
             {0x0001, 0x1p-24}, {0x0400, 0x1p-14}, {0x3555, 0.333251953125},
             {0x3c00, 1.0}, {0xbc00, -1.0}, {0x7bff, 65504.0}}) {
        if (fp16_value(bits) != value || std::signbit(fp16_value(bits)) != std::signbit(value)) {
            std::cerr << "FP16 oracle decoder failed at code " << bits << '\n';
            return 1;
        }
    }
    if (fp16_rne(1.0 + 0x1p-11) != 0x3c00 || fp16_rne(1.0 + 3.0 * 0x1p-11) != 0x3c02 ||
        fp16_rne(0x1p-25) != 0 || fp16_rne(3.0 * 0x1p-25) != 2 ||
        fp16_rne(-0x1p-25) != 0x8000) {
        std::cerr << "FP16 oracle encoder failed at a ties-to-even boundary\n";
        return 1;
    }
    for (unsigned bits = 0; bits < 65536U; ++bits) {
        if ((bits & 0x7c00U) != 0x7c00U &&
            fp16_rne(fp16_value(static_cast<std::uint16_t>(bits))) != bits) {
            std::cerr << "FP16 oracle codec roundtrip failed at code " << bits << '\n';
            return 1;
        }
    }
    return 0;
}

constexpr ninfer::test::ReductionCriterion gdn_fp16_output_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/ninfer::test::kBf16GrossRelativeFloor};
}

constexpr ninfer::test::ReductionCriterion gdn_fp16_state_criterion() {
    // The existing GDN FP32 recurrence allowance plus one binary16 state-publication rounding
    // step (unit roundoff 2^-11). This bound is fixed before running the FP16 routes.
    return {/*relative_l2=*/2.7e-3 + 0x1p-11, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/ninfer::test::kBf16GrossRelativeFloor + 0x1p-11};
}

struct OracleFixture {
    std::vector<std::uint16_t> q, k, v, state;
    std::vector<float> g, beta;
};

OracleFixture oracle_fixture(int tokens, std::uint32_t seed) {
    OracleFixture fixture;
    fixture.q.resize(static_cast<std::size_t>(kDim) * kQk * tokens);
    fixture.k.resize(fixture.q.size());
    fixture.v.resize(static_cast<std::size_t>(kDim) * kHv * tokens);
    fixture.state.resize(static_cast<std::size_t>(kDim) * kDim * kHv);
    fixture.g.resize(static_cast<std::size_t>(kHv) * tokens);
    fixture.beta.resize(fixture.g.size());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    for (auto& value : fixture.q) { value = oracle::f32_to_bf16_rne(uniform(rng)); }
    for (auto& value : fixture.k) { value = oracle::f32_to_bf16_rne(uniform(rng)); }
    for (auto& value : fixture.v) { value = oracle::f32_to_bf16_rne(0.5F * uniform(rng)); }
    for (auto& value : fixture.state) { value = fp16_rne(0.02 * uniform(rng)); }
    // One near-zero Q/K row exercises the public normalization epsilon without changing the
    // fixture's representation or inserting the production normalized staging cast.
    for (int d = 0; d < kDim; ++d) {
        fixture.q[d] = oracle::f32_to_bf16_rne(1.0e-4F * uniform(rng));
        fixture.k[d] = oracle::f32_to_bf16_rne(1.0e-4F * uniform(rng));
    }
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < kHv; ++h) {
            const float alpha = 1.0F - std::pow(10.0F, -1.0F - 3.0F * h / (kHv - 1));
            const auto index = static_cast<std::size_t>(t) * kHv + h;
            fixture.g[index] = std::log(alpha) * (1.0F + 0.1F * uniform(rng));
            fixture.beta[index] = 0.5F + 0.45F * uniform(rng);
        }
    }
    fixture.beta[0] = 0.0F;
    fixture.beta[1] = 1.0F;
    fixture.g[2] = 0.0F;
    fixture.g[3] = std::log(0.2F);
    return fixture;
}

std::vector<float> represented_bf16(const std::vector<std::uint16_t>& bits, std::size_t offset,
                                   std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) { values[i] = oracle::bf16_to_f32(bits[offset + i]); }
    return values;
}

struct CapturedGdn {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool capturing = false;
    ~CapturedGdn() {
        if (capturing) { (void)cudaStreamEndCapture(stream, &graph); }
        if (executable) { (void)cudaGraphExecDestroy(executable); }
        if (graph) { (void)cudaGraphDestroy(graph); }
        if (stream) { (void)cudaStreamDestroy(stream); }
    }
};

int oracle_case(const std::vector<int>& widths, bool distinct_state, bool capture_graph,
                std::uint32_t seed) {
    namespace test = ninfer::test;
    const int max_width = *std::max_element(widths.begin(), widths.end());
    const int tokens = std::accumulate(widths.begin(), widths.end(), 0);
    const OracleFixture fixture = oracle_fixture(tokens, seed);
    const auto state_count = fixture.state.size();
    test::GuardedDeviceBuffer dq(static_cast<std::size_t>(kDim) * kQk * max_width * 2);
    test::GuardedDeviceBuffer dk(dq.bytes()), dv(static_cast<std::size_t>(kDim) * kHv * max_width * 2);
    test::GuardedDeviceBuffer dg(static_cast<std::size_t>(kHv) * max_width * 4), db(dg.bytes());
    test::GuardedDeviceBuffer state(state_count * 2), destination(state_count * 2), out(dv.bytes());
    state.copy_from_host(fixture.state.data(), state.bytes());
    destination.fill(0xff);
    const auto workspace_bytes = ninfer::ops::gated_delta_net_workspace_capacity_bytes(
        kQk, kHv, true, 1, max_width);
    ninfer::WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    Tensor state_tensor(state.data(), DType::FP16, {kDim, kDim, kHv});
    Tensor destination_tensor(destination.data(), DType::FP16, {kDim, kDim, kHv});
    std::vector<float> reference_state(state_count);
    for (std::size_t i = 0; i < state_count; ++i) {
        reference_state[i] = static_cast<float>(fp16_value(fixture.state[i]));
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(kDim));
    CapturedGdn graph;
    int failures = 0;
    std::size_t t0 = 0;
    for (std::size_t call = 0; call < widths.size(); ++call) {
        const int failures_before_call = failures;
        const int width = widths[call];
        if (capture_graph && width != widths.front()) {
            throw std::invalid_argument("graph oracle cases require one fixed width");
        }
        if (distinct_state && widths.size() != 1) {
            throw std::invalid_argument("distinct-state oracle cases require one call");
        }
        const auto nq = static_cast<std::size_t>(kDim) * kQk * width;
        const auto nv = static_cast<std::size_t>(kDim) * kHv * width;
        const auto ng = static_cast<std::size_t>(kHv) * width;
        const auto q0 = t0 * kDim * kQk, v0 = t0 * kDim * kHv, g0 = t0 * kHv;
        dq.copy_from_host(fixture.q.data() + q0, nq * 2);
        dk.copy_from_host(fixture.k.data() + q0, nq * 2);
        dv.copy_from_host(fixture.v.data() + v0, nv * 2);
        dg.copy_from_host(fixture.g.data() + g0, ng * 4);
        db.copy_from_host(fixture.beta.data() + g0, ng * 4);
        out.fill(0xff);
        test::gdn_ref::Inputs reference;
        reference.head_dim = kDim;
        reference.qk_heads = kQk;
        reference.value_heads = kHv;
        reference.tokens = width;
        reference.q = represented_bf16(fixture.q, q0, nq);
        reference.k = represented_bf16(fixture.k, q0, nq);
        reference.v = represented_bf16(fixture.v, v0, nv);
        reference.g.assign(fixture.g.begin() + g0, fixture.g.begin() + g0 + ng);
        reference.beta.assign(fixture.beta.begin() + g0, fixture.beta.begin() + g0 + ng);
        reference.state = reference_state;
        const auto expected = test::gdn_ref::evaluate(reference, static_cast<double>(scale), true);
        const Tensor q(dq.data(), DType::BF16, {kDim, kQk, width});
        const Tensor k(dk.data(), DType::BF16, {kDim, kQk, width});
        const Tensor v(dv.data(), DType::BF16, {kDim, kHv, width});
        const Tensor g(dg.data(), DType::FP32, {kHv, width});
        const Tensor beta(db.data(), DType::FP32, {kHv, width});
        Tensor output(out.data(), DType::BF16, {kDim, kHv, width});
        const auto execute = [&](cudaStream_t stream) {
            if (distinct_state) {
                ninfer::ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace,
                                             state_tensor, destination_tensor, output, stream);
            } else {
                ninfer::ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace,
                                             state_tensor, output, stream);
            }
        };
        if (capture_graph) {
            if (call == 0) {
                test::cuda_check(cudaStreamCreateWithFlags(&graph.stream, cudaStreamNonBlocking),
                                 "oracle graph stream");
                test::cuda_check(cudaStreamBeginCapture(graph.stream, cudaStreamCaptureModeGlobal),
                                 "oracle graph begin capture");
                graph.capturing = true;
                execute(graph.stream);
                test::cuda_check(cudaStreamEndCapture(graph.stream, &graph.graph),
                                 "oracle graph end capture");
                graph.capturing = false;
                test::cuda_check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr,
                                                     nullptr, 0), "oracle graph instantiate");
            }
            test::cuda_check(cudaGraphLaunch(graph.executable, graph.stream), "oracle graph replay");
        } else {
            execute(nullptr);
        }
        test::cuda_synchronize();
        const std::string label = "FP16 GDN T=" + std::to_string(width) + " call=" +
            std::to_string(call) + (capture_graph ? " graph" : distinct_state ? " distinct" : " in-place");
        const auto out_bits = test::from_device<std::uint16_t>(out.data(), nv);
        const auto state_bits = test::from_device<std::uint16_t>(
            distinct_state ? destination.data() : state.data(), state_count);
        std::vector<double> actual_out(nv), actual_state(state_count);
        for (std::size_t i = 0; i < nv; ++i) { actual_out[i] = oracle::bf16_to_f32(out_bits[i]); }
        for (std::size_t i = 0; i < state_count; ++i) { actual_state[i] = fp16_value(state_bits[i]); }
        failures += test::verify_reduction(label + " output", actual_out, expected.out,
                                           gdn_fp16_output_criterion());
        failures += test::verify_reduction(label + " state", actual_state, expected.final_state,
                                           gdn_fp16_state_criterion());
        if (distinct_state) {
            failures += test::verify_exact((label + " read state unchanged").c_str(),
                test::from_device<std::uint16_t>(state.data(), state_count), fixture.state);
        }
        failures += test::verify_exact((label + " q unchanged").c_str(),
            test::from_device<std::uint16_t>(dq.data(), nq),
            std::vector<std::uint16_t>(fixture.q.begin() + q0, fixture.q.begin() + q0 + nq));
        failures += test::verify_exact((label + " k unchanged").c_str(),
            test::from_device<std::uint16_t>(dk.data(), nq),
            std::vector<std::uint16_t>(fixture.k.begin() + q0, fixture.k.begin() + q0 + nq));
        failures += test::verify_exact((label + " v unchanged").c_str(),
            test::from_device<std::uint16_t>(dv.data(), nv),
            std::vector<std::uint16_t>(fixture.v.begin() + v0, fixture.v.begin() + v0 + nv));
        failures += test::verify_exact((label + " g unchanged").c_str(),
            test::from_device<float>(dg.data(), ng), reference.g);
        failures += test::verify_exact((label + " beta unchanged").c_str(),
            test::from_device<float>(db.data(), ng), reference.beta);
        const auto all_output = test::from_device<std::uint16_t>(out.data(), out.bytes() / 2);
        if (!std::all_of(all_output.begin() + nv, all_output.end(),
                         [](auto bits) { return bits == 0xffffU; })) {
            std::cerr << label << ": output suffix overwritten\n";
            ++failures;
        }
        failures += dq.verify_guards(label + " q");
        failures += dk.verify_guards(label + " k");
        failures += dv.verify_guards(label + " v");
        failures += dg.verify_guards(label + " g");
        failures += db.verify_guards(label + " beta");
        failures += state.verify_guards(label + " state");
        failures += destination.verify_guards(label + " destination");
        failures += out.verify_guards(label + " output");
        if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
            std::cerr << label << ": workspace exceeds its query or remains allocated\n";
            ++failures;
        }
        std::cout << (failures == failures_before_call ? "OK " : "FAIL ")
                  << label << " against FP64\n";
        // The next public call reads a new represented input. Never round at internal tokens,
        // WY chunks, two-stage packets, or BF16 output publication inside this reference.
        for (std::size_t i = 0; i < state_count; ++i) {
            reference_state[i] = static_cast<float>(fp16_value(fp16_rne(expected.final_state[i])));
        }
        t0 += static_cast<std::size_t>(width);
    }
    return failures;
}

int qualify_fp16_state(int selected_width) {
    int failures = verify_fp16_codec();
    if (failures != 0) { return failures; }
    const std::vector<int> widths = selected_width > 0 ? std::vector<int>{selected_width}
        : std::vector<int>{1, 4, 7, 15, 16, 17, 63, 64, 65};
    for (const int width : widths) {
        failures += oracle_case({width}, false, false, 0x5eedU + static_cast<unsigned>(width));
    }
    if (selected_width == 0 || selected_width == 7) {
        failures += oracle_case({7}, true, false, 0x17U);
    }
    const int carry_width = selected_width > 0 ? selected_width : 64;
    failures += oracle_case({carry_width, 1}, false, false, 0xcafeU);
    if (selected_width == 0 || selected_width == 1) {
        failures += oracle_case({1, 1}, false, true, 0xfaceU);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " direct FP64 qualification of FP16 GDN state\n";
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int width = 1;
        bool width_explicit = false, oracle_only = false, two_stage = false;
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--oracle-only") {
                oracle_only = true;
            } else if (argument == "--two-stage") {
                two_stage = true;
            } else {
                const auto parsed = std::from_chars(argument.data(), argument.data() + argument.size(), width);
                if (width_explicit || parsed.ec != std::errc{} ||
                    parsed.ptr != argument.data() + argument.size() || width <= 0 || width > kSteps) {
                    throw std::invalid_argument("expected one width in 1..4096, --oracle-only, or --two-stage");
                }
                width_explicit = true;
            }
        }
        if (two_stage) {
            // Prefill widths through the two-stage kernels, which stage FP16 state through FP32.
#ifdef _WIN32
            _putenv_s("NINFER_GDN_TWO_STAGE", "1");
#else
            setenv("NINFER_GDN_TWO_STAGE", "1", 1);
#endif
        }
        if (oracle_only) { return qualify_fp16_state(width_explicit ? width : 0); }
        if (kSteps / width < 4) {
            throw std::invalid_argument("long-chain width must leave at least four calls");
        }
        std::mt19937 rng(0x5eed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::vector<std::uint16_t> q(static_cast<std::size_t>(kDim) * kQk * kSteps);
        std::vector<std::uint16_t> k(q.size());
        std::vector<std::uint16_t> v(static_cast<std::size_t>(kDim) * kHv * kSteps);
        std::vector<float> g(static_cast<std::size_t>(kHv) * kSteps);
        std::vector<float> beta(g.size());
        for (auto& x : q) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (auto& x : k) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (auto& x : v) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (int t = 0; t < kSteps; ++t) {
            for (int h = 0; h < kHv; ++h) {
                const float alpha = 1.0F - std::pow(10.0F, -1.0F - 3.0F * h / (kHv - 1));
                g[static_cast<std::size_t>(t) * kHv + h] =
                    std::log(alpha) * (1.0F + 0.1F * normal(rng));
                beta[static_cast<std::size_t>(t) * kHv + h] = 1.0F / (1.0F + std::exp(-normal(rng)));
            }
        }
        ninfer::test::GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2);
        ninfer::test::GuardedDeviceBuffer dg(g.size() * 4), db(beta.size() * 4);
        dq.copy_from_host(q.data(), q.size() * 2);
        dk.copy_from_host(k.data(), k.size() * 2);
        dv.copy_from_host(v.data(), v.size() * 2);
        dg.copy_from_host(g.data(), g.size() * 4);
        db.copy_from_host(beta.data(), beta.size() * 4);
        const std::size_t state_elems = static_cast<std::size_t>(kDim) * kDim * kHv;
        ninfer::test::GuardedDeviceBuffer s32(state_elems * 4), s16(state_elems * 2);
        s32.fill(0);
        s16.fill(0);
        const std::size_t out_bytes = static_cast<std::size_t>(kDim) * kHv * width * 2;
        ninfer::test::GuardedDeviceBuffer o32(out_bytes), o16(out_bytes);
        ninfer::WorkspaceArena ws(std::max<std::size_t>(
            256, ninfer::ops::gated_delta_net_workspace_capacity_bytes(kQk, kHv, true, width,
                                                                        width)));
        Tensor state32(s32.data(), DType::FP32, {kDim, kDim, kHv});
        Tensor state16(s16.data(), DType::FP16, {kDim, kDim, kHv});
        const float scale = 1.0F / std::sqrt(static_cast<float>(kDim));

        std::vector<std::uint16_t> h32(static_cast<std::size_t>(kDim) * kHv * width);
        std::vector<std::uint16_t> h16(h32.size());
        std::vector<double> err(kSteps / width);
        double worst = 0.0;
        for (int call = 0; call < kSteps / width; ++call) {
            const std::size_t t0 = static_cast<std::size_t>(call) * width;
            const auto slice     = [&](void* base, std::size_t elem, std::size_t per_token,
                                   DType dtype, std::initializer_list<std::int32_t> shape) {
                return Tensor(static_cast<std::uint8_t*>(base) + t0 * per_token * elem, dtype,
                              shape);
            };
            const Tensor tq = slice(dq.data(), 2, kDim * kQk, DType::BF16, {kDim, kQk, width});
            const Tensor tk = slice(dk.data(), 2, kDim * kQk, DType::BF16, {kDim, kQk, width});
            const Tensor tv = slice(dv.data(), 2, kDim * kHv, DType::BF16, {kDim, kHv, width});
            const Tensor tg = slice(dg.data(), 4, kHv, DType::FP32, {kHv, width});
            const Tensor tb = slice(db.data(), 4, kHv, DType::FP32, {kHv, width});
            Tensor out32(o32.data(), DType::BF16, {kDim, kHv, width});
            Tensor out16(o16.data(), DType::BF16, {kDim, kHv, width});
            ninfer::ops::gated_delta_net(tq, tk, tv, tg, tb, scale, true, ws, state32, out32,
                                         nullptr);
            ninfer::ops::gated_delta_net(tq, tk, tv, tg, tb, scale, true, ws, state16, out16,
                                         nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "gated_delta_net");
            o32.copy_to_host(h32.data(), h32.size() * 2);
            o16.copy_to_host(h16.data(), h16.size() * 2);
            double num = 0.0, den = 0.0;
            for (std::size_t i = 0; i < h32.size(); ++i) {
                const double a = oracle::bf16_to_f32(h32[i]);
                const double b = oracle::bf16_to_f32(h16[i]);
                num += (a - b) * (a - b);
                den += a * a;
            }
            err[call] = std::sqrt(num / std::max(den, 1e-30));
            worst     = std::max(worst, err[call]);
        }
        const auto mean = [&](int begin, int end) {
            double sum = 0.0;
            for (int i = begin; i < end; ++i) { sum += err[i]; }
            return sum / (end - begin);
        };
        const int calls = kSteps / width;
        const double q1 = mean(0, calls / 4), q2 = mean(calls / 4, calls / 2),
                     q4 = mean(3 * calls / 4, calls);
        std::cout << "width " << width << ": relative output error, mean by quarter " << q1 << ' '
                  << q2 << ' ' << mean(calls / 2, 3 * calls / 4) << ' ' << q4 << ", worst "
                  << worst << '\n';
        const bool ok = worst <= 2e-2 && q4 <= 2.0 * q2;
        std::cout << (ok ? "OK" : "FAIL") << " FP16 GDN state tracks FP32 over " << kSteps
                  << " tokens\n";
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP16 state test failed: " << error.what() << '\n';
        return 1;
    }
}
