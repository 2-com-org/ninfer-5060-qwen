#pragma once

#include "ops/softmax_attention/oracle.h"

namespace ninfer::test {

inline std::vector<int> sampled_attention_heads(ops::AttentionHeadGeometry geometry, int tokens) {
    if (!ops::valid_attention_head_geometry(geometry) || tokens <= 0) {
        throw std::invalid_argument("invalid sampled attention geometry or width");
    }
    const int group = geometry.query_heads / geometry.kv_heads;
    std::vector<int> heads;
    for (int head = 0; head < geometry.kv_heads; ++head) {
        // Vary the query head with the width, while covering every distinct KV head.
        heads.push_back(head * group + (head + tokens - 1) % group);
    }
    return heads;
}

inline std::vector<int> sampled_attention_queries(int tokens) {
    if (tokens <= 0) throw std::invalid_argument("sampled attention requires a query");
    return tokens == 1 ? std::vector<int>{0} : std::vector<int>{0, tokens - 1};
}

template <class Query, class Key, class Value>
std::vector<double> sampled_dense_attention(ops::AttentionHeadGeometry geometry,
                                            const std::vector<int>& positions, double scale,
                                            Query query, Key key, Value value) {
    const auto heads = sampled_attention_heads(geometry, static_cast<int>(positions.size()));
    const auto queries = sampled_attention_queries(static_cast<int>(positions.size()));
    const ops::AttentionHeadGeometry sampled{geometry.head_dim, geometry.kv_heads, geometry.kv_heads};
    std::vector<double> result(static_cast<std::size_t>(sampled.head_dim) * sampled.query_heads *
                               queries.size());
    // Each selected query still attends every visible key, using the same independent FP64
    // mathematical oracle and every output dimension. Only query rows are sampled.
    naive_dense_softmax_attention(
        sampled, static_cast<int>(queries.size()), positions.back() + 1, scale,
        [&](int d, int head, int token) {
            return query(d, heads[static_cast<std::size_t>(head)], queries[static_cast<std::size_t>(token)]);
        },
        key, value,
        [&](int token, int position) {
            return position <= positions[static_cast<std::size_t>(queries[static_cast<std::size_t>(token)])];
        },
        [&](int d, int head, int token, double output) {
            result[static_cast<std::size_t>(d + sampled.head_dim * (head + sampled.query_heads * token))] = output;
        });
    return result;
}

inline std::vector<double> sample_attention_output(ops::AttentionHeadGeometry geometry, int tokens,
                                                  const std::vector<double>& output) {
    const auto heads = sampled_attention_heads(geometry, tokens);
    const auto queries = sampled_attention_queries(tokens);
    std::vector<double> sampled;
    sampled.reserve(static_cast<std::size_t>(geometry.head_dim) * heads.size() * queries.size());
    for (const int token : queries) {
        for (const int head : heads) {
            for (int d = 0; d < geometry.head_dim; ++d) {
                sampled.push_back(output.at(static_cast<std::size_t>(
                    d + geometry.head_dim * (head + geometry.query_heads * token))));
            }
        }
    }
    return sampled;
}

} // namespace ninfer::test
