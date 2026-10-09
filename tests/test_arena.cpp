#include "core/arena.h"
#include "core/device.h"
#include "cuda_availability.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

using ninfer::test::cuda_unavailable;

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw expected exception\n";
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_ptr(void* actual, void* expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;
    CUDA_CHECK(cudaSetDevice(0));

    ninfer::DeviceBuffer empty;
    failures += expect_ptr(empty.p, nullptr, "empty device buffer pointer");
    failures += expect_size(empty.bytes, 0, "empty device buffer size");

    const std::array<std::uint8_t, 8> host_source{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<std::uint8_t, 8> host_destination{};
    ninfer::DeviceBuffer buffer(host_source.size());
    buffer.copy_from_host(host_source.data(), host_source.size());
    buffer.copy_to_host(host_destination.data(), host_destination.size());
    if (host_destination != host_source) {
        ++failures;
        std::cerr << "device buffer round trip changed payload\n";
    }
    failures += expect_throws<std::out_of_range>(
        [&] { buffer.copy_from_host(host_source.data(), 2, buffer.bytes - 1); },
        "device buffer upload range");
    ninfer::DeviceBuffer moved_buffer(std::move(buffer));
    failures += expect_ptr(buffer.p, nullptr, "moved-from device buffer pointer");
    failures += expect_size(buffer.bytes, 0, "moved-from device buffer size");
    failures += expect_size(moved_buffer.bytes, host_source.size(), "moved device buffer size");

    ninfer::DeviceArena arena(1024);
    failures += expect_size(arena.capacity(), 1024, "arena.capacity");
    failures += expect_size(arena.used(), 0, "arena.used initial");
    failures += expect_size(arena.peak_used(), 0, "arena.peak initial");
    if (arena.base() == nullptr) {
        ++failures;
        std::cerr << "arena base is null\n";
    }

    auto* base       = static_cast<unsigned char*>(arena.base());
    ninfer::Tensor a = arena.alloc(ninfer::DType::BF16, {3, 5});
    failures += expect_ptr(a.data, base, "first allocation pointer");
    failures += expect_size(a.bytes(), 30, "first allocation bytes");
    failures += expect_size(arena.used(), 30, "arena.used after first allocation");
    failures += expect_size(arena.peak_used(), 30, "arena.peak after first allocation");

    ninfer::Tensor b = arena.alloc(ninfer::DType::U8, {17}, 64);
    failures += expect_ptr(b.data, base + 64, "second allocation pointer");
    if (reinterpret_cast<std::uintptr_t>(b.data) % 64 != 0) {
        ++failures;
        std::cerr << "second allocation is not 64-byte aligned\n";
    }
    failures += expect_size(arena.used(), 81, "arena.used after second allocation");
    failures += expect_size(arena.peak_used(), 81, "arena.peak after second allocation");

    const std::size_t used_before_scope = arena.used();
    void* transient_ptr                 = nullptr;
    std::size_t peak_after_transient    = 0;
    {
        auto outer_scope         = arena.scope();
        ninfer::Tensor transient = arena.alloc(ninfer::DType::U8, {11}, 128);
        transient_ptr            = transient.data;
        const std::size_t used   = arena.used();
        if (used <= used_before_scope) {
            ++failures;
            std::cerr << "transient allocation did not advance arena cursor\n";
        }
        {
            auto inner_scope = arena.scope();
            (void)arena.alloc(ninfer::DType::U8, {7}, 64);
        }
        failures += expect_size(arena.used(), used, "arena.used after nested scope");
        peak_after_transient = arena.peak_used();
    }
    failures += expect_size(arena.peak_used(), peak_after_transient, "arena.peak after scope exit");
    failures += expect_size(arena.used(), used_before_scope, "arena.used after scope exit");
    ninfer::Tensor reused = arena.alloc(ninfer::DType::U8, {5}, 128);
    failures += expect_ptr(reused.data, transient_ptr, "allocation after scope pointer");

    const std::size_t used_before_exception_scope = arena.used();
    failures += expect_throws<std::runtime_error>(
        [&] {
            auto exception_scope = arena.scope();
            (void)arena.alloc(ninfer::DType::U8, {9}, 64);
            throw std::runtime_error("scope unwind");
        },
        "arena scope exception");
    failures +=
        expect_size(arena.used(), used_before_exception_scope, "arena.used after exception scope");

    const std::size_t used_before_failures = arena.used();
    const std::size_t peak_before_failures = arena.peak_used();
    failures += expect_throws<std::bad_alloc>(
        [&] { (void)arena.alloc(ninfer::DType::FP32, {300}, 256); }, "arena oom");
    failures += expect_size(arena.used(), used_before_failures, "arena.used after oom");
    failures += expect_size(arena.peak_used(), peak_before_failures, "arena.peak after oom");

    arena.reset();
    failures += expect_size(arena.used(), 0, "arena.used after reset");
    failures += expect_size(arena.peak_used(), peak_before_failures, "arena.peak after reset");
    arena.reset_peak();
    failures += expect_size(arena.peak_used(), 0, "arena.peak after reset_peak on empty arena");
    ninfer::Tensor c = arena.alloc(ninfer::DType::U8, {4});
    failures += expect_ptr(c.data, base, "allocation after reset pointer");
    failures += expect_size(arena.peak_used(), 4, "arena.peak after reset allocation");

    ninfer::DeviceArena moved(std::move(arena));
    if (arena.base() != nullptr || arena.capacity() != 0 || arena.used() != 0) {
        ++failures;
        std::cerr << "move construction did not clear source arena\n";
    }
    failures += expect_size(arena.peak_used(), 0, "moved-from arena peak");
    failures += expect_size(moved.capacity(), 1024, "moved arena capacity");
    failures += expect_size(moved.peak_used(), 4, "moved arena peak");

    void* external = nullptr;
    CUDA_CHECK(cudaMalloc(&external, 512));
    {
        ninfer::DeviceArena borrowed(ninfer::DeviceSpan{external, 512});
        failures += expect_ptr(borrowed.base(), external, "borrowed arena base");
        failures += expect_size(borrowed.capacity(), 512, "borrowed arena capacity");
        const ninfer::Tensor item = borrowed.alloc(ninfer::DType::U8, {17}, 64);
        failures += expect_ptr(item.data, external, "borrowed arena allocation");
        failures += expect_size(borrowed.used(), 17, "borrowed arena used");
    }
    CUDA_CHECK(cudaMemset(external, 0, 512));
    CUDA_CHECK(cudaFree(external));

    ninfer::PinnedHostBuffer pinned(128);
    if (pinned.data() == nullptr) {
        ++failures;
        std::cerr << "pinned data is null\n";
    }
    failures += expect_size(pinned.size(), 128, "pinned.size");
    std::memset(pinned.data(), 0x5a, pinned.size());

    // Large Host KV backing must remain a portable pinned allocation with exact transfers,
    // including a partial final page and graph replay after ownership moves.
    constexpr std::size_t large_bytes = (std::size_t{64} << 20) + 4096 + 17;
    ninfer::PinnedHostBuffer large(large_bytes, ninfer::HostPagePolicy::PreferHugePages);
    const void* original_large = large.data();
    ninfer::PinnedHostBuffer moved_large(std::move(large));
    failures += expect_ptr(large.data(), nullptr, "moved-from large pinned pointer");
    failures += expect_size(large.size(), 0, "moved-from large pinned size");
    ninfer::PinnedHostBuffer assigned_large(128);
    assigned_large = std::move(moved_large);
    failures += expect_ptr(moved_large.data(), nullptr, "assigned-from large pinned pointer");
    failures += expect_size(moved_large.size(), 0, "assigned-from large pinned size");
    if (assigned_large.data() != original_large) {
        ++failures;
        std::cerr << "large pinned move changed its address\n";
    }
    failures += expect_size(assigned_large.size(), large_bytes, "large pinned extent");
    cudaPointerAttributes pinned_attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&pinned_attributes, assigned_large.data()));
    if (pinned_attributes.type != cudaMemoryTypeHost) {
        ++failures;
        std::cerr << "large buffer is not CUDA pinned host memory\n";
    }
    ninfer::PinnedHostBuffer large_return(large_bytes);
    ninfer::DeviceBuffer large_device(large_bytes);
    cudaStream_t transfer_stream = nullptr;
    cudaGraph_t transfer_graph = nullptr;
    cudaGraphExec_t transfer_exec = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamBeginCapture(transfer_stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaMemcpyAsync(large_device.p, assigned_large.data(), large_bytes,
                              cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(large_return.data(), large_device.p, large_bytes,
                              cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaStreamEndCapture(transfer_stream, &transfer_graph));
    CUDA_CHECK(cudaGraphInstantiate(&transfer_exec, transfer_graph, 0));
    auto* large_source = static_cast<std::uint8_t*>(assigned_large.data());
    const auto* large_result = static_cast<const std::uint8_t*>(large_return.data());
    for (std::size_t replay = 0; replay < 2; ++replay) {
        for (std::size_t byte = 0; byte < large_bytes; ++byte) {
            large_source[byte] = static_cast<std::uint8_t>((byte * 11 + (byte >> 8)) ^
                                                         (0x5a + replay * 17));
        }
        CUDA_CHECK(cudaGraphLaunch(transfer_exec, transfer_stream));
        CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
        for (std::size_t byte = 0; byte < large_bytes; ++byte) {
            const auto expected = static_cast<std::uint8_t>((byte * 11 + (byte >> 8)) ^
                                                           (0x5a + replay * 17));
            if (large_result[byte] != expected) {
                ++failures;
                std::cerr << "large pinned graph round trip changed byte " << byte << '\n';
                break;
            }
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(transfer_exec));
    CUDA_CHECK(cudaGraphDestroy(transfer_graph));
    CUDA_CHECK(cudaStreamDestroy(transfer_stream));
    assigned_large = std::move(pinned);
    failures += expect_ptr(pinned.data(), nullptr, "assigned-from ordinary pinned pointer");
    failures += expect_size(assigned_large.size(), 128, "ordinary replacement of large buffer");
    if (static_cast<const std::uint8_t*>(assigned_large.data())[127] != 0x5a) {
        ++failures;
        std::cerr << "ordinary replacement of large buffer changed its payload\n";
    }
    cudaPointerAttributes released_attributes{};
    const cudaError_t released_status =
        cudaPointerGetAttributes(&released_attributes, original_large);
    if ((released_status == cudaSuccess &&
         released_attributes.type != cudaMemoryTypeUnregistered) ||
        (released_status != cudaSuccess && released_status != cudaErrorInvalidValue)) {
        ++failures;
        std::cerr << "replaced large pinned registration was not released\n";
    }
    if (released_status != cudaSuccess) { (void)cudaGetLastError(); }
    failures += expect_throws<std::invalid_argument>(
        [] { ninfer::PinnedHostBuffer zero(0, ninfer::HostPagePolicy::PreferHugePages); },
        "large-page zero extent");

    return failures == 0 ? 0 : fail("arena test failed");
}
