#include "serve/request_history.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace ninfer::serve {

namespace {

using Json = nlohmann::json;

bool utf8_continuation(char byte) noexcept {
    return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

// Keeps the last `limit` bytes, starting on a character boundary; returns the bytes dropped.
std::size_t keep_tail(std::string& text, std::size_t limit) {
    if (text.size() <= limit) { return 0; }
    std::size_t start = text.size() - limit;
    while (start < text.size() && utf8_continuation(text[start])) { ++start; }
    text.erase(0, start);
    return start;
}

// Keeps the first `limit` bytes, ending on a character boundary; returns the bytes dropped.
std::size_t keep_head(std::string& text, std::size_t limit) {
    if (text.size() <= limit) { return 0; }
    std::size_t end = limit;
    while (end > 0 && utf8_continuation(text[end])) { --end; }
    const std::size_t dropped = text.size() - end;
    text.resize(end);
    return dropped;
}

const char* role_name(ninfer::ChatRole role) noexcept {
    switch (role) {
    case ninfer::ChatRole::System:    return "system";
    case ninfer::ChatRole::Developer: return "developer";
    case ninfer::ChatRole::User:      return "user";
    case ninfer::ChatRole::Assistant: return "assistant";
    case ninfer::ChatRole::Tool:      return "tool";
    }
    return "message";
}

std::string render_turn(const ChatTurn& turn) {
    std::string out = "[";
    out += role_name(turn.role);
    if (turn.role == ninfer::ChatRole::Tool) {
        const std::string& name = turn.tool_result_name ? *turn.tool_result_name : turn.tool_call_id;
        if (!name.empty()) { out += " " + name; }
        if (turn.tool_result_is_error) { out += ", error"; }
    }
    out += "]\n";
    if (!turn.reasoning_content.empty()) {
        out += "<think>\n" + turn.reasoning_content + "\n</think>\n";
    }
    for (std::size_t i = 0; i < turn.content.size(); ++i) {
        const ContentPart& part = turn.content[i];
        if (i != 0) { out += '\n'; }
        switch (part.kind) {
        case ContentKind::Text:  out += part.text; break;
        case ContentKind::Image: out += "[image]"; break;
        case ContentKind::Video: out += "[video]"; break;
        }
    }
    for (const ToolCall& call : turn.tool_calls) {
        out += "\n-> " + call.name + "(" + call.arguments_json + ")";
    }
    return out;
}

double rate(double tokens, double seconds) { return seconds > 0.0 ? tokens / seconds : 0.0; }

} // namespace

std::string render_request_input(const GenerationRequest& request, std::size_t limit) {
    std::string out;
    bool omitted = false;
    if (request.raw_prompt) {
        out = "[prompt]\n";
        for (const RawPromptPiece& piece : *request.raw_prompt) {
            out += piece.token ? "<token " + std::to_string(*piece.token) + ">" : piece.text;
        }
    } else {
        // From the latest message back: a long conversation renders only what the limit keeps.
        std::vector<std::string> turns;
        std::size_t total = 0;
        for (auto turn = request.messages.rbegin(); turn != request.messages.rend(); ++turn) {
            turns.push_back(render_turn(*turn));
            total += turns.back().size() + 2;
            if (total > limit) { break; }
        }
        omitted = turns.size() < request.messages.size();
        for (auto turn = turns.rbegin(); turn != turns.rend(); ++turn) {
            if (!out.empty()) { out += "\n\n"; }
            out += *turn;
        }
    }
    if (keep_tail(out, limit) != 0 || omitted) { out.insert(0, "[earlier input omitted]\n"); }
    return out;
}

RequestHistory::Entry& RequestHistory::entry_locked(std::uint64_t id) {
    for (auto& [entry_id, entry] : entries_) {
        if (entry_id == id) { return entry; }
    }
    if (entries_.size() >= capacity_) { entries_.pop_front(); }
    return entries_.emplace_back(id, Entry{}).second;
}

void RequestHistory::start(const RequestLogContext& context, std::uint64_t unix_ms) {
    std::string input = context.input_preview;
    std::lock_guard lock(mutex_);
    Entry& entry  = entry_locked(context.id);
    entry.summary = Json{{"id", context.id},
                         {"protocol", context.protocol},
                         {"model", context.model},
                         {"stream", context.stream},
                         {"message_count", context.message_count},
                         {"media_item_count", context.media_item_count},
                         {"tool_count", context.tool_count},
                         {"thinking", context.enable_thinking},
                         {"requested_output_tokens", context.requested_output_tokens},
                         {"state", "running"},
                         {"started_unix_ms", unix_ms},
                         {"finished_unix_ms", nullptr}};
    entry.input = std::move(input);
}

void RequestHistory::done(const RequestLogContext& context, const GenerationOutcome& outcome,
                          const std::string& record, std::uint64_t unix_ms) {
    Json parsed = Json::parse(record, nullptr, false);
    if (parsed.is_discarded()) { parsed = nullptr; }
    std::string output              = outcome.text;
    const std::size_t output_omitted = keep_head(output, kRequestHistoryTextBytes);
    std::string reasoning              = outcome.reasoning;
    const std::size_t reasoning_omitted = keep_head(reasoning, kRequestHistoryTextBytes);
    Json tool_calls = Json::array();
    for (const ninfer::GeneratedToolCall& call : outcome.tool_calls) {
        tool_calls.push_back(Json{{"name", call.name}, {"arguments", call.arguments_json}});
    }

    const GenerationMetrics& metrics = outcome.metrics;
    const int computed               = std::max(
        0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens));
    Json result = parsed.is_object() ? parsed.value("result", Json::object()) : Json::object();
    Json speculative =
        parsed.is_object() ? parsed.value("speculative", Json::object()) : Json::object();

    std::lock_guard lock(mutex_);
    Entry& entry = entry_locked(context.id);
    if (!entry.summary.is_object()) { entry.summary = Json{{"id", context.id}}; }
    Json& summary                       = entry.summary;
    summary["state"]                    = "completed";
    summary["finished_unix_ms"]         = unix_ms;
    summary["finish_reason"]            = result.value("finish_reason", "");
    summary["prefix_reuse_path"]        = result.value("prefix_reuse_path", "");
    summary["prompt_tokens"]            = outcome.prompt_tokens;
    summary["cached_tokens"]            = metrics.prefix_cache_hit_tokens;
    summary["computed_prompt_tokens"]   = computed;
    summary["completion_tokens"]        = outcome.completion_tokens;
    summary["reasoning_tokens"]         = outcome.reasoning_tokens;
    summary["tool_call_count"]          = outcome.tool_calls.size();
    summary["queue_wait_seconds"]       = metrics.engine_timing.queue_wait_seconds;
    summary["ttft_seconds"]             = metrics.ttft_seconds;
    summary["prefill_seconds"]          = metrics.prefill_seconds;
    summary["decode_seconds"]           = metrics.decode_seconds;
    summary["total_seconds"]            = metrics.total_seconds;
    summary["prefill_tokens_per_second"] = rate(computed, metrics.prefill_seconds);
    summary["decode_tokens_per_second"] = rate(outcome.completion_tokens, metrics.decode_seconds);
    summary["speculative"] = Json{{"backend", speculative.value("backend", "")},
                                  {"drafted_tokens", metrics.speculative_draft_tokens},
                                  {"accepted_tokens", metrics.speculative_accepted_tokens},
                                  {"copy_drafted_tokens", metrics.ngram_drafted_tokens},
                                  {"copy_accepted_tokens", metrics.ngram_accepted_tokens}};
    entry.output                  = std::move(output);
    entry.output_omitted_bytes    = output_omitted;
    entry.reasoning               = std::move(reasoning);
    entry.reasoning_omitted_bytes = reasoning_omitted;
    entry.tool_calls              = std::move(tool_calls);
    entry.record                  = std::move(parsed);
}

void RequestHistory::failure(const RequestLogContext& context, const RequestFailure& failure,
                             std::uint64_t unix_ms) {
    std::lock_guard lock(mutex_);
    Entry& entry = entry_locked(context.id);
    if (!entry.summary.is_object()) { entry.summary = Json{{"id", context.id}}; }
    entry.summary["state"]            = "error";
    entry.summary["finished_unix_ms"] = unix_ms;
    entry.summary["http_status"]      = failure.http_status;
    entry.summary["error"] = Json{{"type", failure.error_type},
                                  {"code", failure.error_code},
                                  {"message", failure.machine_message}};
}

void RequestHistory::rejected(const RequestRejectionLogContext& context, std::uint64_t unix_ms) {
    std::lock_guard lock(mutex_);
    Entry& entry  = entry_locked(context.id);
    entry.summary = Json{{"id", context.id},
                         {"protocol", context.protocol},
                         {"model", context.model},
                         {"stream", context.stream},
                         {"message_count", context.message_count},
                         {"state", "rejected"},
                         {"started_unix_ms", unix_ms},
                         {"finished_unix_ms", unix_ms},
                         {"http_status", context.error.status},
                         {"error", Json{{"type", context.error.type},
                                        {"code", context.error.code},
                                        {"message", context.error.message}}}};
}

Json RequestHistory::list() const {
    Json requests = Json::array();
    std::lock_guard lock(mutex_);
    for (auto entry = entries_.rbegin(); entry != entries_.rend(); ++entry) {
        requests.push_back(entry->second.summary);
    }
    return Json{{"object", "ninfer.request_history"},
                {"capacity", capacity_},
                {"text_limit_bytes", kRequestHistoryTextBytes},
                {"requests", std::move(requests)}};
}

std::optional<Json> RequestHistory::detail(std::uint64_t id) const {
    std::lock_guard lock(mutex_);
    for (const auto& [entry_id, entry] : entries_) {
        if (entry_id != id) { continue; }
        Json out                       = entry.summary;
        out["input"]                   = entry.input;
        out["output"]                  = entry.output;
        out["output_omitted_bytes"]    = entry.output_omitted_bytes;
        out["reasoning"]               = entry.reasoning;
        out["reasoning_omitted_bytes"] = entry.reasoning_omitted_bytes;
        out["tool_calls"]              = entry.tool_calls;
        out["record"]                  = entry.record;
        return out;
    }
    return std::nullopt;
}

} // namespace ninfer::serve
