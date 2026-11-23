// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_REFERENCE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_REFERENCE_H_

// Host references of Qwen 4's sparse-attention composites
// (odml.qsa_indexer_update and odml.sparse_sdpa_transposed), written from the
// GPU delegate's kernels and the references of its tests
// (ml_drift_delegate/delegate/composite/qsa_composites_kernel_test.cc), in
// double precision: what the CUDA kernels of qsa_indexer.h and
// qsa_sparse_attention.h are checked against.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace litert::nvidia::qsa_reference {

inline constexpr int kBlockTokens = 4;

// A mask [rows, len]: bool bytes, or additive floats.
struct Mask {
  const uint8_t* bools = nullptr;
  const float* floats = nullptr;
  int64_t len = 0;
  // The indexer's test of a row's own key (its diagonal entry).
  bool Diagonal(int64_t row, int64_t column) const {
    return bools != nullptr ? bools[row * len + column] != 0
                            : floats[row * len + column] > -1.0f;
  }
  // Whether the row sees the key, for the scores and the attention.
  bool Visible(int64_t row, int64_t column) const {
    return bools != nullptr ? bools[row * len + column] != 0
                            : floats[row * len + column] == 0.0f;
  }
};

inline float HalfBitsToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1Fu;
  uint32_t mantissa = bits & 0x3FFu;
  uint32_t out;
  if (exponent == 0) {
    if (mantissa == 0) {
      out = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFu;
      out = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    out = sign | 0x7F800000u | (mantissa << 13);
  } else {
    out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

// Round to nearest even, saturating to infinity.
inline uint16_t FloatToHalfBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t exponent = (bits >> 23) & 0xFFu;
  uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent == 0xFF) {
    return sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u);
  }
  int32_t e = static_cast<int32_t>(exponent) - 127 + 15;
  if (e >= 31) return sign | 0x7C00u;
  if (e <= 0) {
    if (e < -10) return sign;
    mantissa |= 0x800000u;
    const int shift = 14 - e;
    uint32_t half = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1);
    const uint32_t mid = 1u << (shift - 1);
    if (rest > mid || (rest == mid && (half & 1u) != 0)) ++half;
    return sign | static_cast<uint16_t>(half);
  }
  uint32_t half = (static_cast<uint32_t>(e) << 10) | (mantissa >> 13);
  const uint32_t rest = mantissa & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u) != 0)) ++half;
  return sign | static_cast<uint16_t>(half);
}

inline float RoundToHalf(float value) {
  return HalfBitsToFloat(FloatToHalfBits(value));
}

inline uint16_t FloatToBf16Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0u;
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

inline float Bf16BitsToFloat(uint16_t bits) {
  const uint32_t wide = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

struct IndexerShape {
  int rows = 1;
  int index_heads = 1;
  int index_dim = 4;
  int rotary_dim = 2;
  int max_blocks = 1;
  int block_topk = 1;
  int64_t mask_len = 4;
  float eps = 1e-6f;
  float scale = 1.0f;
  // The pool and ring hold FP16 values (each stored value rounded).
  bool fp16_state = false;
};

// What one step of the indexer gives: the scores of every row against the
// blocks below `bound` (NaN where the row cannot see the block) and the
// selection of each row, ascending, padded with -1.
struct IndexerResult {
  int bound = 0;
  std::vector<double> scores;  // [rows][bound]
  std::vector<int32_t> topk;   // [rows][block_topk]
};

// The rows of a selection: the min(valid, k) best (ties to the lower index)
// in ascending order, then -1.
inline std::vector<int32_t> SelectTopk(const std::vector<double>& scores,
                                       int k) {
  std::vector<std::pair<double, int>> valid;
  for (int b = 0; b < static_cast<int>(scores.size()); ++b) {
    if (!std::isnan(scores[b])) valid.push_back({scores[b], b});
  }
  std::sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second < b.second;
  });
  std::vector<int32_t> picked;
  for (int i = 0; i < std::min<int>(k, valid.size()); ++i) {
    picked.push_back(valid[i].second);
  }
  std::sort(picked.begin(), picked.end());
  picked.resize(k, -1);
  return picked;
}

// The scores and selections of an indexer step against `pool` (after its
// update): q [rows][heads][dim].
inline IndexerResult IndexerScores(const IndexerShape& s,
                                   const std::vector<float>& q,
                                   const std::vector<float>& pool,
                                   int start_pos, const Mask& mask) {
  const int dim = s.index_dim;
  const int pos0 = std::max(start_pos, 0);
  IndexerResult result;
  result.bound = std::clamp((pos0 + s.rows) / kBlockTokens, 0, s.max_blocks);
  const bool dense = result.bound <= s.block_topk;
  result.scores.assign(static_cast<size_t>(s.rows) * result.bound, NAN);
  for (int t = 0; t < s.rows; ++t) {
    std::vector<double> row(result.bound, NAN);
    for (int b = 0; b < result.bound; ++b) {
      bool complete = true;
      for (int r = 0; r < kBlockTokens; ++r) {
        complete &= mask.Visible(t, int64_t{b} * kBlockTokens + r);
      }
      if (!complete) continue;
      if (dense) {
        row[b] = 0.0;
        continue;
      }
      double sum = 0.0;
      for (int h = 0; h < s.index_heads; ++h) {
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) {
          dot +=
              static_cast<double>(
                  q[(static_cast<size_t>(t) * s.index_heads + h) * dim + d]) *
              pool[static_cast<size_t>(b) * dim + d];
        }
        sum += std::max(dot, 0.0);
      }
      row[b] = sum * s.scale;
    }
    std::copy(row.begin(), row.end(),
              result.scores.begin() + static_cast<size_t>(t) * result.bound);
    const auto picked = SelectTopk(row, s.block_topk);
    result.topk.insert(result.topk.end(), picked.begin(), picked.end());
  }
  return result;
}

// odml.qsa_indexer_update: q [rows][heads][dim], k_new [rows][dim], ring
// [4][dim] and pool [max_blocks][dim] (updated), norm [dim], cos and sin
// [max_blocks][rotary_dim].
inline IndexerResult IndexerUpdate(
    const IndexerShape& s, const std::vector<float>& q,
    const std::vector<float>& k_new, std::vector<float>& ring,
    std::vector<float>& pool, const std::vector<float>& norm,
    const std::vector<float>& cos, const std::vector<float>& sin, int start_pos,
    const Mask& mask) {
  const int dim = s.index_dim;
  const auto store = [&](double value) {
    return s.fp16_state ? RoundToHalf(static_cast<float>(value))
                        : static_cast<float>(value);
  };
  const int pos0 = std::max(start_pos, 0);
  const int start_blk = pos0 / kBlockTokens;
  int valid = 0;
  for (int t = 0; t < s.rows; ++t) {
    const int64_t column = int64_t{pos0} + t;
    if (column < mask.len && mask.Diagonal(t, column)) valid = t + 1;
  }
  const int end_pos = pos0 + valid;
  const int end_blk = end_pos / kBlockTokens;
  const std::vector<float> old_ring = ring;
  const auto key = [&](int token, int d) -> double {
    return token >= pos0
               ? k_new[static_cast<size_t>(token - pos0) * dim + d]
               : old_ring[static_cast<size_t>(token % kBlockTokens) * dim + d];
  };
  for (int b = start_blk; b < end_blk && b < s.max_blocks; ++b) {
    std::vector<double> pooled(dim);
    double square = 0.0;
    for (int d = 0; d < dim; ++d) {
      double sum = 0.0;
      for (int r = 0; r < kBlockTokens; ++r)
        sum += key(b * kBlockTokens + r, d);
      pooled[d] = sum / kBlockTokens;
      square += pooled[d] * pooled[d];
    }
    const double inv_rms = 1.0 / std::sqrt(square / dim + s.eps);
    for (int d = 0; d < dim; ++d) pooled[d] *= inv_rms * norm[d];
    const int half = s.rotary_dim / 2;
    for (int d = 0; d < dim; ++d) {
      double value = pooled[d];
      if (d < s.rotary_dim) {
        const double c = cos[static_cast<size_t>(b) * s.rotary_dim + d];
        const double sn = sin[static_cast<size_t>(b) * s.rotary_dim + d];
        const double partner = d < half ? -pooled[d + half] : pooled[d - half];
        value = value * c + partner * sn;
      }
      pool[static_cast<size_t>(b) * dim + d] = store(value);
    }
  }
  if (valid > 0) {
    for (int r = 0; r < kBlockTokens; ++r) {
      const int token = end_blk * kBlockTokens + r;
      for (int d = 0; d < dim; ++d) {
        double value = 0.0;
        if (token >= pos0 && token < end_pos) {
          value = k_new[static_cast<size_t>(token - pos0) * dim + d];
        } else if (token < pos0 && end_blk == start_blk) {
          value = old_ring[static_cast<size_t>(r) * dim + d];
        }
        ring[static_cast<size_t>(r) * dim + d] = store(value);
      }
    }
  }
  return IndexerScores(s, q, pool, start_pos, mask);
}

// Whether `got` (one row: ascending block indices, then -1) is a top-k of
// `scores` (NaN: not a candidate) up to `tolerance` relative to the largest
// score: exactly min(candidates, k) candidates, and no candidate left out
// scores more than the lowest one taken by over the tolerance. Explains a
// failure in `why`.
inline bool IsTopk(const double* scores, int bound, const int32_t* got, int k,
                   double tolerance, std::string* why) {
  int candidates = 0;
  double largest = 0.0;
  for (int b = 0; b < bound; ++b) {
    if (std::isnan(scores[b])) continue;
    ++candidates;
    largest = std::max(largest, std::fabs(scores[b]));
  }
  const int want = std::min(candidates, k);
  std::vector<bool> taken(bound, false);
  double lowest_taken = INFINITY;
  for (int i = 0; i < k; ++i) {
    if (i >= want) {
      if (got[i] != -1) {
        *why = "entry " + std::to_string(i) + " is " + std::to_string(got[i]) +
               ", not -1";
        return false;
      }
      continue;
    }
    const int b = got[i];
    if (b < 0 || b >= bound || std::isnan(scores[b]) ||
        (i > 0 && got[i - 1] >= b)) {
      *why = "entry " + std::to_string(i) + " = " + std::to_string(b) +
             " is not an ascending candidate";
      return false;
    }
    taken[b] = true;
    lowest_taken = std::min(lowest_taken, scores[b]);
  }
  const double slack = tolerance * std::max(1.0, largest);
  for (int b = 0; b < bound; ++b) {
    if (!std::isnan(scores[b]) && !taken[b] &&
        scores[b] > lowest_taken + slack) {
      *why = "block " + std::to_string(b) + " (score " +
             std::to_string(scores[b]) + ") left out while one scoring " +
             std::to_string(lowest_taken) + " is taken";
      return false;
    }
  }
  return true;
}

struct AttentionShape {
  int rows = 1;
  int q_heads = 1;
  int kv_heads = 1;
  int head_dim = 64;
  int cache_len = 4;
  int block_topk = 1;
  float scale = 1.0f;
  bool value_columns = false;
};

// odml.sparse_sdpa_transposed (ReferenceSparseGqa of the GPU delegate's
// tests, with a visibility function): q [q_heads][rows][dim], k [kv][L][dim],
// v [kv][L][dim] or [kv][dim][L], topk [rows][block_topk] ->
// [rows][q_heads][dim].
inline std::vector<float> SparseAttention(
    const AttentionShape& s, const std::vector<float>& q,
    const std::vector<float>& k, const std::vector<float>& v,
    const std::vector<int32_t>& topk,
    const std::function<bool(int row, int token)>& visible) {
  const int dim = s.head_dim;
  const int len = s.cache_len;
  const int blocks = len / kBlockTokens;
  const int group = s.q_heads / s.kv_heads;
  std::vector<float> out(static_cast<size_t>(s.rows) * s.q_heads * dim, 0.0f);
  for (int r = 0; r < s.rows; ++r) {
    std::vector<bool> keep(len, false);
    for (int i = 0; i < s.block_topk; ++i) {
      const int b = topk[static_cast<size_t>(r) * s.block_topk + i];
      if (b >= 0 && b < blocks) {
        for (int t = 0; t < kBlockTokens; ++t)
          keep[b * kBlockTokens + t] = true;
      }
    }
    for (int b = 0; b < blocks; ++b) {
      int seen = 0;
      for (int t = 0; t < kBlockTokens; ++t) {
        seen += visible(r, b * kBlockTokens + t) ? 1 : 0;
      }
      if (seen > 0 && seen < kBlockTokens) {
        for (int t = 0; t < kBlockTokens; ++t)
          keep[b * kBlockTokens + t] = true;
      }
    }
    std::vector<int> tokens;
    for (int t = 0; t < len; ++t) {
      if (keep[t] && visible(r, t)) tokens.push_back(t);
    }
    if (tokens.empty()) continue;
    for (int h = 0; h < s.q_heads; ++h) {
      const int kv = h / group;
      std::vector<double> logits(tokens.size());
      double max_logit = -INFINITY;
      for (size_t i = 0; i < tokens.size(); ++i) {
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) {
          dot += static_cast<double>(
                     q[(static_cast<size_t>(h) * s.rows + r) * dim + d]) *
                 k[(static_cast<size_t>(kv) * len + tokens[i]) * dim + d];
        }
        logits[i] = dot * s.scale;
        max_logit = std::max(max_logit, logits[i]);
      }
      double sum = 0.0;
      for (double& logit : logits) {
        logit = std::exp(logit - max_logit);
        sum += logit;
      }
      for (int d = 0; d < dim; ++d) {
        double acc = 0.0;
        for (size_t i = 0; i < tokens.size(); ++i) {
          const int t = tokens[i];
          const float value =
              s.value_columns
                  ? v[(static_cast<size_t>(kv) * dim + d) * len + t]
                  : v[(static_cast<size_t>(kv) * len + t) * dim + d];
          acc += logits[i] / sum * value;
        }
        out[(static_cast<size_t>(r) * s.q_heads + h) * dim + d] =
            static_cast<float>(acc);
      }
    }
  }
  return out;
}

}  // namespace litert::nvidia::qsa_reference

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_REFERENCE_H_
