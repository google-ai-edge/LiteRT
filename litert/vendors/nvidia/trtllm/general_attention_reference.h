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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GENERAL_ATTENTION_REFERENCE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GENERAL_ATTENTION_REFERENCE_H_

// Host references for the tests of general_attention.h. They compute the
// composites' decompositions in double precision from the values the tensors
// hold: a cache value is its integer times its scale, a float a graph
// quantizes is the integer the CPU backend's QUANTIZE gives times the scale,
// and a bool mask's fill is far below every score (the ML Drift tests'
// ComputeSdpaReferenceOutput and ReferenceAttention, with -10000 as fill).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace litert::nvidia::general_reference {

// TFLite's QUANTIZE as the CPU backend computes it (general_attention.h).
inline int32_t QuantizeInt8(float value, float scale) {
  const float q = std::nearbyint(value * (1.0f / scale));
  return static_cast<int32_t>(std::min(std::max(q, -128.0f), 127.0f));
}

inline int32_t QuantizeInt4(float value, float scale) {
  const float q = std::round(value / scale);
  return static_cast<int32_t>(std::min(std::max(q, -8.0f), 7.0f));
}

// The caches as int values with their scales (or float values with scale 1).
struct Caches {
  int heads = 0;
  int len = 0;
  int depth = 0;
  int bits = 8;  // 8, 4, or 0 for float
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  std::vector<float> k;  // [heads][len][depth], unscaled values
  std::vector<float> v;  // [heads][depth][len], unscaled values

  float Key(int h, int j, int d) const {
    return k[(static_cast<size_t>(h) * len + j) * depth + d] * k_scale;
  }
  float Value(int h, int d, int j) const {
    return v[(static_cast<size_t>(h) * depth + d) * len + j] * v_scale;
  }
};

// The value a cache holds for `value`.
inline float OnGrid(float value, int bits, float scale) {
  if (bits == 8) return static_cast<float>(QuantizeInt8(value, scale)) * scale;
  if (bits == 4) return static_cast<float>(QuantizeInt4(value, scale)) * scale;
  return value;
}

struct AttentionCase {
  int kv_heads = 1;
  int rows = 1;
  int depth = 64;
  int new_len = 0;
  int mask_rows = 1;
  bool bool_mask = true;
  float q_int8_scale = 0.0f;
  float new_int8_scale = 0.0f;
  float scale = 1.0f;
  float softcap = 0.0f;
};

// q [kv_heads][rows][depth]; new_k, new_v [kv_heads][new_len][depth];
// mask [mask_rows][len + new_len] (bool as 0/1 floats, or additive).
inline std::vector<double> Attention(const AttentionCase& c,
                                     const Caches& caches,
                                     const std::vector<float>& q,
                                     const std::vector<float>& new_k,
                                     const std::vector<float>& new_v,
                                     const std::vector<float>& mask) {
  const int keys = caches.len + c.new_len;
  std::vector<double> out(static_cast<size_t>(c.kv_heads) * c.rows * c.depth);
  std::vector<double> scores(keys);
  for (int h = 0; h < c.kv_heads; ++h) {
    for (int r = 0; r < c.rows; ++r) {
      const float* qr = &q[(static_cast<size_t>(h) * c.rows + r) * c.depth];
      const float* mr = &mask[static_cast<size_t>(r % c.mask_rows) * keys];
      bool sees = false;
      for (int j = 0; j < keys; ++j) sees = sees || mr[j] != 0.0f;
      auto key = [&](int j, int d) -> double {
        if (j < caches.len) return caches.Key(h, j, d);
        const float x =
            new_k[(static_cast<size_t>(h) * c.new_len + (j - caches.len)) *
                      c.depth +
                  d];
        const float rounded =
            c.new_int8_scale > 0.0f
                ? QuantizeInt8(x, c.new_int8_scale) * c.new_int8_scale
                : x;
        return OnGrid(rounded, caches.bits, caches.k_scale);
      };
      auto value = [&](int j, int d) -> double {
        if (j < caches.len) return caches.Value(h, d, j);
        const float x =
            new_v[(static_cast<size_t>(h) * c.new_len + (j - caches.len)) *
                      c.depth +
                  d];
        const float rounded =
            c.new_int8_scale > 0.0f
                ? QuantizeInt8(x, c.new_int8_scale) * c.new_int8_scale
                : x;
        return OnGrid(rounded, caches.bits, caches.v_scale);
      };
      double max_score = -INFINITY;
      for (int j = 0; j < keys; ++j) {
        double dot = 0.0;
        for (int d = 0; d < c.depth; ++d) {
          const double qv =
              c.q_int8_scale > 0.0f
                  ? QuantizeInt8(qr[d], c.q_int8_scale) * c.q_int8_scale
                  : qr[d];
          dot += qv * key(j, d);
        }
        double s = dot * c.scale;
        if (c.softcap > 0.0f) s = c.softcap * std::tanh(s / c.softcap);
        if (c.bool_mask) {
          if (sees && mr[j] == 0.0f) s = -10000.0;
          if (!sees) s = -10000.0;
        } else {
          s += mr[j];
        }
        scores[j] = s;
        max_score = std::max(max_score, s);
      }
      double total = 0.0;
      for (int j = 0; j < keys; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        total += scores[j];
      }
      double* o = &out[(static_cast<size_t>(h) * c.rows + r) * c.depth];
      for (int d = 0; d < c.depth; ++d) {
        double sum = 0.0;
        for (int j = 0; j < keys; ++j) sum += scores[j] * value(j, d);
        o[d] = sum / total;
      }
    }
  }
  return out;
}

}  // namespace litert::nvidia::general_reference

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GENERAL_ATTENTION_REFERENCE_H_
