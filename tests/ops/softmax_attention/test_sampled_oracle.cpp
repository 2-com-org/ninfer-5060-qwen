#include "ops/softmax_attention/sampled_oracle.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        constexpr ninfer::ops::AttentionHeadGeometry geometry{2, 4, 2};
        const std::vector<int> positions{0, 0, 1};
        const auto query = [](int d, int head, int token) {
            return d == 0 ? static_cast<double>(1 + head + 2 * token) : 0.0;
        };
        const auto key = [](int d, int, int position) {
            return d == 0 ? static_cast<double>(position) : 0.0;
        };
        const auto value = [](int d, int head, int position) {
            return static_cast<double>(2 + 8 * head + 4 * position + 2 * d);
        };
        const auto actual = ninfer::test::sampled_dense_attention(
            geometry, positions, 1.0, query, key, value);
        const double first_weight = std::exp(5.0);
        const double second_weight = std::exp(8.0);
        const std::vector<double> expected{
            2.0, 4.0, 10.0, 12.0,
            (2.0 + 6.0 * first_weight) / (1.0 + first_weight),
            (4.0 + 8.0 * first_weight) / (1.0 + first_weight),
            (10.0 + 14.0 * second_weight) / (1.0 + second_weight),
            (12.0 + 16.0 * second_weight) / (1.0 + second_weight)};
        if (actual.size() != expected.size()) throw std::runtime_error("sample size mismatch");
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(actual[i] - expected[i]) > 1e-12) {
                throw std::runtime_error("sampled oracle changed head, query, visibility or reduction");
            }
        }
        std::vector<double> output(geometry.head_dim * geometry.query_heads * positions.size());
        for (int token = 0; token < 3; ++token) {
            for (int head = 0; head < 4; ++head) {
                for (int d = 0; d < 2; ++d) {
                    output[d + 2 * (head + 4 * token)] = 100.0 * token + 10.0 * head + d;
                }
            }
        }
        if (ninfer::test::sample_attention_output(geometry, 3, output) !=
            std::vector<double>{0, 1, 30, 31, 200, 201, 230, 231}) {
            throw std::runtime_error("sampled output did not select the qualified rows");
        }
        if (ninfer::test::sample_attention_output(geometry, 1,
                                                  std::vector<double>{0, 1, 10, 11, 20, 21, 30, 31}) !=
            std::vector<double>{0, 1, 30, 31}) {
            throw std::runtime_error("single query was sampled twice");
        }
        std::cout << "PASS: sampled FP64 attention queries, KV heads, visibility and output layout\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sampled attention oracle: " << error.what() << '\n';
        return 1;
    }
}
