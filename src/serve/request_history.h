#pragma once

// The WebUI Monitor's record of recent requests: each request's state, timings, input and output,
// kept in memory until the server stops (never on disk) and served by GET /api/requests. Fed from
// the same request lifecycle as the JSONL log, whether or not that log is enabled.

#include "serve/request_events.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace ninfer::serve {

inline constexpr std::size_t kRequestHistoryCapacity = 100;
// Bytes kept of each text: the input's end (its latest messages) and the start of the output and
// reasoning.
inline constexpr std::size_t kRequestHistoryTextBytes = 64 * 1024;

// The request's messages (or raw prompt) as readable text, truncated to its last `limit` bytes on a
// UTF-8 boundary behind a marker line. Media parts appear as placeholders; tool calls with their
// arguments.
[[nodiscard]] std::string render_request_input(const GenerationRequest& request,
                                               std::size_t limit = kRequestHistoryTextBytes);

class RequestHistory {
public:
    explicit RequestHistory(std::size_t capacity = kRequestHistoryCapacity)
        : capacity_(capacity) {}

    RequestHistory(const RequestHistory&)            = delete;
    RequestHistory& operator=(const RequestHistory&) = delete;

    void start(const RequestLogContext& context, std::uint64_t unix_ms);
    // `record` is the request's JSONL request_done record (format_request_done_json).
    void done(const RequestLogContext& context, const GenerationOutcome& outcome,
              const std::string& record, std::uint64_t unix_ms);
    void failure(const RequestLogContext& context, const RequestFailure& failure,
                 std::uint64_t unix_ms);
    void rejected(const RequestRejectionLogContext& context, std::uint64_t unix_ms);

    // Every retained request, newest first, without its texts.
    [[nodiscard]] nlohmann::json list() const;
    // One request with its input, output, reasoning, tool calls and request_done record.
    [[nodiscard]] std::optional<nlohmann::json> detail(std::uint64_t id) const;

private:
    struct Entry {
        nlohmann::json summary;
        std::string input;
        std::string output;
        std::size_t output_omitted_bytes = 0;
        std::string reasoning;
        std::size_t reasoning_omitted_bytes = 0;
        nlohmann::json tool_calls = nlohmann::json::array();
        nlohmann::json record     = nullptr;
    };

    // Caller holds mutex_. The entry of `id`, created (evicting the oldest) when absent.
    Entry& entry_locked(std::uint64_t id);

    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<std::pair<std::uint64_t, Entry>> entries_; // oldest first
};

} // namespace ninfer::serve
