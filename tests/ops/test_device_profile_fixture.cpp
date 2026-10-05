#include "ops/device_profile_fixture.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
constexpr std::string_view kCatalog = "0123456789abcdef";
constexpr std::string_view kHardware = "test-gpu-sm120";

template <class Action>
void require_failure(Action action, std::string_view reason) {
    try {
        action();
    } catch (const std::exception& error) {
        if (std::string_view(error.what()).find(reason) != std::string_view::npos) return;
        throw std::runtime_error(std::string("unexpected failure: ") + error.what());
    }
    throw std::runtime_error("invalid qualification profile was accepted");
}
} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("ninfer-test-profile-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                       ".json");
    try {
        ninfer::runtime::DeviceProfileEntry entry;
        entry.profile.hardware_class = kHardware;
        entry.profile.multiprocessors = 36;
        entry.profile.origin = "qualification fixture";
        entry.profile.routes["attn_i8_small/h24/rk4v4/w1"] = {{65536, "4x2x32"}};
        entry.calibration = ninfer::runtime::DeviceProfileCalibration{
            std::string(kCatalog), "test-build", 13010, 13010, "2026-10-04T12:00:00Z"};
        const auto write_entry = [&] {
            std::ofstream output(path);
            output << ninfer::runtime::serialize_device_route_profiles({entry});
        };
        require_failure([&] { ninfer::test::load_device_profile_fixture(path, kHardware, 36, kCatalog); },
                        "open");
        write_entry();
        const auto profile = ninfer::test::load_device_profile_fixture(path, kHardware, 36, kCatalog);
        if (profile.hardware_class != kHardware || profile.multiprocessors != 36 ||
            profile.routes.at("attn_i8_small/h24/rk4v4/w1").front().schedule != "4x2x32") {
            throw std::runtime_error("qualification profile did not preserve the measured route");
        }
        require_failure([&] { ninfer::test::load_device_profile_fixture(path, "other-gpu-sm120", 36, kCatalog); },
                        "hardware");
        require_failure([&] { ninfer::test::load_device_profile_fixture(path, kHardware, 37, kCatalog); },
                        "hardware");
        require_failure([&] { ninfer::test::load_device_profile_fixture(path, kHardware, 36, "fedcba9876543210"); },
                        "catalog");
        entry.calibration.reset();
        write_entry();
        require_failure([&] { ninfer::test::load_device_profile_fixture(path, kHardware, 36, kCatalog); },
                        "catalog");
        std::filesystem::remove(path);
        std::cout << "PASS: numerical test profile requires matching hardware, SM count and catalog\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << "device profile fixture: " << error.what() << '\n';
        return 1;
    }
}
