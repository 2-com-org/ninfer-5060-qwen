#pragma once

#include "calibration/route_catalog.h"
#include "ops/common/device_route.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/device_profile.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::test {

// Qualification must fail if the requested file cannot supply a measured route for this device.
// Engine's permissive lookup may instead retain a compiled profile, which would qualify a
// different route. Reuse the production JSON parser and profile layering after strict selection.
inline ops::DeviceRouteProfile load_device_profile_fixture(const std::filesystem::path& path,
                                                           std::string_view hardware_class,
                                                           int multiprocessors,
                                                           std::string_view route_catalog) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open qualification device profile: " + path.string()); }
    const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    for (const auto& entry : runtime::parse_device_route_profiles(json, path.string())) {
        if (entry.profile.hardware_class != hardware_class ||
            entry.profile.multiprocessors != multiprocessors) continue;
        if (!entry.calibration || entry.calibration->route_catalog != route_catalog) {
            throw std::runtime_error("qualification device profile has no matching route catalog: " +
                                     path.string());
        }
        const auto compiled = runtime::compiled_device_route_profile(hardware_class, multiprocessors);
        return compiled ? runtime::layer_device_route_profiles(entry.profile, *compiled)
                        : entry.profile;
    }
    throw std::runtime_error("qualification device profile has no matching hardware class and SM count: " +
                             path.string());
}

inline void install_device_profile_fixture_from_env() {
    const char* path = std::getenv("NINFER_TEST_DEVICE_PROFILE_PATH");
    if (path == nullptr || *path == '\0') return;
    const auto checked = [](cudaError_t status) {
        if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
    };
    int device = 0;
    checked(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    checked(cudaGetDeviceProperties(&properties, device));
    const std::string hardware = runtime::context_cost_hardware_class(
        properties.name, properties.major, properties.minor);
    auto profile = load_device_profile_fixture(path, hardware, properties.multiProcessorCount,
                                               calibration::calibration_route_catalog_digest());
    std::cout << "Qualification device profile: " << path << " (" << hardware << ", "
              << properties.multiProcessorCount << " SMs, " << profile.routes.size()
              << " route keys, " << profile.origin << ")\n";
    ops::install_device_route_profile(device,
                                      std::make_shared<const ops::DeviceRouteProfile>(std::move(profile)));
}

} // namespace ninfer::test
