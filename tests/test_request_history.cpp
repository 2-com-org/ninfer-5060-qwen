#include "serve/request_history.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ChatTurn turn(ninfer::ChatRole role, std::string text) {
    ChatTurn out;
    out.role = role;
    ContentPart part;
    part.text = std::move(text);
    out.content.push_back(std::move(part));
    return out;
}

bool valid_utf8_start(const std::string& text) {
    return text.empty() || (static_cast<unsigned char>(text.front()) & 0xC0U) != 0x80U;
}

RequestLogContext context_for(std::uint64_t id, std::string input = "hello") {
    RequestLogContext context;
    context.id            = id;
    context.protocol      = "openai-chat";
    context.model         = "test-model";
    context.stream        = true;
    context.message_count = 1;
    context.input_preview = std::move(input);
    return context;
}

int test_render_input() {
    int failures = 0;
    GenerationRequest request;
    request.messages.push_back(turn(ninfer::ChatRole::System, "Be brief."));
    ChatTurn user = turn(ninfer::ChatRole::User, "Describe this:");
    ContentPart image;
    image.kind = ContentKind::Image;
    user.content.push_back(image);
    request.messages.push_back(std::move(user));
    ChatTurn assistant = turn(ninfer::ChatRole::Assistant, "");
    assistant.tool_calls.push_back(ToolCall{"call_1", "lookup", R"({"q":"x"})"});
    request.messages.push_back(std::move(assistant));
    ChatTurn tool         = turn(ninfer::ChatRole::Tool, "42");
    tool.tool_call_id     = "call_1";
    tool.tool_result_name = "lookup";
    request.messages.push_back(std::move(tool));

    const std::string text = render_request_input(request);
    failures += check(text == "[system]\nBe brief.\n\n[user]\nDescribe this:\n[image]\n\n"
                              "[assistant]\n\n-> lookup({\"q\":\"x\"})\n\n[tool lookup]\n42",
                      "render_request_input changed the message rendering");

    // The latest messages are kept, behind the marker, and the cut never splits a character.
    GenerationRequest long_request;
    long_request.messages.push_back(turn(ninfer::ChatRole::User, std::string(200, 'a')));
    std::string accented;
    for (int i = 0; i < 100; ++i) { accented += "\xC3\xA9"; } // e-acute, 2 bytes each
    long_request.messages.push_back(turn(ninfer::ChatRole::User, accented + "END"));
    const std::string tail = render_request_input(long_request, 101);
    const std::string marker = "[earlier input omitted]\n";
    failures += check(tail.starts_with(marker), "a truncated input carries no marker");
    const std::string kept = tail.substr(marker.size());
    failures += check(kept.size() <= 101 && kept.ends_with("END") && valid_utf8_start(kept),
                      "input truncation kept the wrong end or split a character");
    failures += check(render_request_input(long_request, 1000).find(marker) == std::string::npos,
                      "an input within the limit was marked as truncated");

    // Messages before the one that crosses the limit are not rendered at all.
    GenerationRequest many;
    for (int i = 0; i < 50; ++i) {
        many.messages.push_back(turn(ninfer::ChatRole::User, "message " + std::to_string(i)));
    }
    const std::string latest = render_request_input(many, 60);
    failures += check(latest.starts_with(marker) && latest.ends_with("message 49") &&
                          latest.find("message 0\n") == std::string::npos,
                      "a long conversation did not keep its latest messages");

    GenerationRequest raw;
    raw.raw_prompt = std::vector<RawPromptPiece>{{"Once upon", std::nullopt}, {"", 7}};
    failures += check(render_request_input(raw) == "[prompt]\nOnce upon<token 7>",
                      "raw prompt rendering changed");
    return failures;
}

int test_history() {
    int failures = 0;
    RequestHistory history(3);
    history.start(context_for(1, "first input"), 1000);
    Json list = history.list();
    failures += check(list["requests"].size() == 1 && list["requests"][0]["state"] == "running" &&
                          list["requests"][0]["started_unix_ms"] == 1000,
                      "a started request is not listed as running");
    failures += check(!list["requests"][0].contains("input"), "the list carries request texts");

    GenerationOutcome outcome;
    outcome.text                            = std::string(kRequestHistoryTextBytes - 1, 'x') + "\xC3\xA9tail";
    outcome.reasoning                       = "thought";
    outcome.prompt_tokens                   = 100;
    outcome.completion_tokens               = 50;
    outcome.finish_reason                   = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens = 60;
    outcome.metrics.prefill_seconds         = 0.5;
    outcome.metrics.decode_seconds          = 2.0;
    outcome.metrics.total_seconds           = 2.6;
    outcome.metrics.ttft_seconds            = 0.6;
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{"lookup", "{}"});
    const RequestLogContext first = context_for(1, "first input");
    history.done(first, outcome, format_request_done_json("srv", 2000, first, outcome), 2000);

    const std::optional<Json> detail = history.detail(1);
    failures += check(detail.has_value(), "a finished request has no detail");
    if (detail) {
        const Json& d = *detail;
        failures += check(d["state"] == "completed" && d["finished_unix_ms"] == 2000 &&
                              d["input"] == "first input" && d["reasoning"] == "thought",
                          "detail lost the state, input or reasoning");
        const std::string output = d["output"];
        // The 2-byte character straddles the limit, so the cut falls before it.
        failures += check(output.size() == kRequestHistoryTextBytes - 1 &&
                              d["output_omitted_bytes"] == 6,
                          "output truncation split a character or miscounted the rest");
        failures += check(d["cached_tokens"] == 60 && d["computed_prompt_tokens"] == 40 &&
                              d["decode_tokens_per_second"] == 25.0 &&
                              d["prefill_tokens_per_second"] == 80.0 &&
                              d["finish_reason"] == "stop_token",
                          "detail timings or token counts are wrong");
        failures += check(d["tool_calls"].size() == 1 && d["tool_calls"][0]["name"] == "lookup",
                          "detail lost the tool calls");
        failures += check(d["record"].is_object() && d["record"]["event"] == "request_done",
                          "detail lost the request_done record");
    }

    RequestFailure failure;
    failure.http_status     = 500;
    failure.error_type      = "internal_error";
    failure.machine_message = "boom";
    history.start(context_for(2), 3000);
    history.failure(context_for(2), failure, 3100);
    RequestRejectionLogContext rejected;
    rejected.id            = 3;
    rejected.protocol      = "anthropic-messages";
    rejected.error.status  = 400;
    rejected.error.message = "bad";
    history.rejected(rejected, 3200);
    history.start(context_for(4), 3300);

    list = history.list();
    failures += check(list["requests"].size() == 3, "capacity 3 kept another number of requests");
    failures += check(list["requests"][0]["id"] == 4 && list["requests"][1]["id"] == 3 &&
                          list["requests"][2]["id"] == 2,
                      "the list is not newest first or did not evict the oldest");
    failures += check(list["requests"][1]["state"] == "rejected" &&
                          list["requests"][1]["error"]["message"] == "bad" &&
                          list["requests"][2]["state"] == "error" &&
                          list["requests"][2]["http_status"] == 500,
                      "failure or rejection not recorded");
    failures += check(!history.detail(1).has_value() && !history.detail(99).has_value(),
                      "an evicted or unknown request still has a detail");
    return failures;
}

} // namespace

int main() {
    int failures = test_render_input() + test_history();
    if (failures == 0) { std::cout << "request history: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
