/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_BENCHMARK_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_BENCHMARK_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "tensor/examples/gemma3/tokenizer.h"

namespace litert::tensor::examples {

struct BenchmarkConfig {
  int prefill_tokens = 0;
  int decode_tokens = 0;
  int seed = 0;
};

inline std::vector<int32_t> BuildMlDriftDecodeCacheParams(int cache_len) {
  const int active_tokens = cache_len + 1;
  return {cache_len, active_tokens, active_tokens};
}

// Common Gemma chat stop tokens.
// These IDs match the Gemma tokenizer special tokens (<start_of_turn>=105,
// <end_of_turn>=106) used by Cactus and HF configs.
constexpr int32_t kStartOfTurnToken = 105;
constexpr int32_t kEndOfTurnToken = 106;

inline bool IsStopToken(int32_t token) {
  return token == GemmaTokenizerSP::kEosToken || token == kEndOfTurnToken ||
         token == kStartOfTurnToken;
}

inline std::vector<int32_t> MakeBenchmarkTokens(const BenchmarkConfig& cfg,
                                                int vocab_size) {
  const int prefill_tokens = std::max(1, cfg.prefill_tokens);
  std::vector<int32_t> tokens(static_cast<size_t>(prefill_tokens),
                              GemmaTokenizerSP::kBosToken);
  if (prefill_tokens == 1) {
    return tokens;
  }

  std::mt19937 rng(cfg.seed);
  const int32_t min_token_id = std::max<int32_t>(4, kEndOfTurnToken + 1);
  std::uniform_int_distribution<int32_t> dist(
      min_token_id, static_cast<int32_t>(vocab_size - 1));
  for (int i = 1; i < prefill_tokens; ++i) {
    int32_t t = dist(rng);
    while (IsStopToken(t) || t == GemmaTokenizerSP::kBosToken) {
      t = dist(rng);
    }
    tokens[static_cast<size_t>(i)] = t;
  }
  return tokens;
}

inline int32_t ArgMaxFinite(const float* logits, int vocab_size) {
  int32_t argmax = 0;
  float max_v = -std::numeric_limits<float>::infinity();
  bool found_finite = false;
  for (int32_t i = 0; i < vocab_size; ++i) {
    const float v = logits[i];
    if (!std::isfinite(v)) {
      continue;
    }
    if (!found_finite || v > max_v) {
      max_v = v;
      argmax = i;
      found_finite = true;
    }
  }
  return argmax;
}

inline int32_t ArgMaxSkipStopTokens(const float* logits, int vocab_size) {
  float best = -std::numeric_limits<float>::infinity();
  int32_t best_token = -1;
  for (int32_t i = 0; i < vocab_size; ++i) {
    if (IsStopToken(i)) {
      continue;
    }
    const float v = logits[i];
    if (!std::isfinite(v)) {
      continue;
    }
    if (v > best) {
      best = v;
      best_token = i;
    }
  }
  if (best_token >= 0) {
    return best_token;
  }
  return ArgMaxFinite(logits, vocab_size);
}

inline int32_t ArgMaxFinite(const std::vector<float>& logits, int offset,
                            int vocab_size) {
  return ArgMaxFinite(logits.data() + offset, vocab_size);
}

inline int32_t ArgMaxFinite(const std::vector<float>& logits, int vocab_size) {
  return ArgMaxFinite(logits, /*offset=*/0, vocab_size);
}

inline int32_t ArgMaxSkipStopTokens(const std::vector<float>& logits,
                                    int offset, int vocab_size) {
  return ArgMaxSkipStopTokens(logits.data() + offset, vocab_size);
}

inline int32_t ArgMaxSkipStopTokens(const std::vector<float>& logits,
                                    int vocab_size) {
  return ArgMaxSkipStopTokens(logits, /*offset=*/0, vocab_size);
}

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_BENCHMARK_UTILS_H_
