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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_LINEAR_REFERENCE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_LINEAR_REFERENCE_H_

// Host references of the kernels of quant_linear.h for their tests: the
// arithmetic of LiteRT's CPU path as that header states it, written element
// by element. Float sums that the reference leaves unordered (the RMS norms)
// follow the kernels' order, so every result but a sine, cosine or tanh is
// bit-identical to the kernels'.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "litert/vendors/nvidia/trtllm/quant_linear.h"

namespace litert::nvidia::quant_reference {

// FLOAT32 to FLOAT16 bits, round to nearest even.
inline uint16_t HalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112;
  uint32_t mantissa = bits & 0x7FFFFF;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa |= 0x800000;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7C00);
  uint32_t half =
      sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t rest = mantissa & 0x1FFF;
  if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) ++half;
  return static_cast<uint16_t>(half);
}

inline float HalfValue(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
  const uint32_t exponent = (half >> 10) & 0x1F;
  const uint32_t mantissa = half & 0x3FF;
  if (exponent == 0) {
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return (half & 0x8000) ? -magnitude : magnitude;
  }
  const uint32_t bits =
      exponent == 31 ? sign | 0x7F800000 | (mantissa << 13)
                     : sign | ((exponent + 112) << 23) | (mantissa << 13);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline float RoundToFp16(float value) { return HalfValue(HalfBits(value)); }

inline uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

inline float Bf16Value(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline int Saturate8(long value) {
  return static_cast<int>(std::max(-128L, std::min(127L, value)));
}

// QUANTIZE of a FLOAT16 value; `scale` is fl16(1 / tensor scale).
inline int Quantize(float value, float scale) {
  return Saturate8(std::lrintf(value * scale));
}

// Element `index` of packed signed weights of `bit_width` bits.
inline int PackedWeight(const uint8_t* weights, int bit_width, size_t index) {
  const int per_byte = 8 / bit_width;
  const int shift = static_cast<int>(index % per_byte) * bit_width;
  const int field =
      (weights[index / per_byte] >> shift) & ((1 << bit_width) - 1);
  const int sign = 1 << (bit_width - 1);
  return (field ^ sign) - sign;
}

inline int OutputIndex(const LiteRtNvidiaQuantLinearParams& p, int row) {
  if (p.swap0 == 1) return row;
  const int inner = p.rows / (p.swap0 * p.swap1);
  const int a = row / (p.swap1 * inner);
  const int b = row / inner % p.swap1;
  return (b * p.swap0 + a) * inner + row % inner;
}

// LiteRtNvidiaLaunchQuantLinear. `activation` holds INT8 values, or the
// FLOAT16 values of a float activation. Returns the INT8 outputs as ints, or
// the logits.
inline std::vector<float> Linear(const LiteRtNvidiaQuantLinearParams& p, int m,
                                 const std::vector<float>& activation,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& multipliers,
                                 const std::vector<int8_t>& lut,
                                 const std::vector<float>& row_scale) {
  std::vector<float> output(static_cast<size_t>(m) * p.rows);
  const int width = p.segments * p.columns;
  const int segment_rows = p.rows / p.segments;
  for (int row_index = 0; row_index < m; ++row_index) {
    std::vector<int> x(width);
    for (int i = 0; i < width; ++i) {
      const float value =
          activation[static_cast<size_t>(row_index) * width + i];
      x[i] = p.activation == kLiteRtNvidiaQuantInt8
                 ? static_cast<int>(value)
                 : Quantize(value, p.quantize_scale);
    }
    for (int r = 0; r < p.rows; ++r) {
      const int segment = r / segment_rows;
      long dot = 0;
      long sum = 0;
      for (int k = 0; k < p.columns; ++k) {
        const size_t weight = p.layout == kLiteRtNvidiaQuantLayoutRows
                                  ? static_cast<size_t>(r) * p.columns + k
                                  : (static_cast<size_t>(segment) * p.columns +
                                     k) * segment_rows +
                                        r % segment_rows;
        const int xk = x[segment * p.columns + k];
        dot += xk * PackedWeight(weights.data(), p.bit_width, weight);
        sum += xk;
      }
      const size_t out = static_cast<size_t>(row_index) * p.rows;
      if (p.requant == kLiteRtNvidiaQuantRequantFp32) {
        const float scaled = static_cast<float>(dot) * multipliers[r];
        output[out + r] = static_cast<float>(Saturate8(std::lrintf(scaled)));
        continue;
      }
      // TFLite: scale * (dot - zero_point * sum), zero point -0.5.
      const double acc =
          static_cast<double>(multipliers[r]) *
          (static_cast<double>(dot) + 0.5 * static_cast<double>(sum));
      if (p.requant == kLiteRtNvidiaQuantRequantLogits) {
        float logit = RoundToFp16(static_cast<float>(
            acc * static_cast<double>(row_scale[row_index])));
        if (p.cap_in != 0.0f) logit = std::tanh(logit * p.cap_in) * p.cap_out;
        output[out + r] = logit;
        continue;
      }
      int q = static_cast<int>(
          std::clamp(std::round(acc * p.ratio), -128.0, 127.0));
      if (p.lut != 0) q = lut[q + 128];
      output[out + OutputIndex(p, r)] = static_cast<float>(q);
    }
  }
  return output;
}

// The sum of `values` in the order of the kernels' BlockSum: the threads of
// a 256-thread CTA hold the elements t, t + 256, ... (already added per
// thread in `per_thread`), a warp adds its lanes as a butterfly and the warp
// totals are added in order.
inline float BlockSum(const std::vector<float>& per_thread) {
  float total = 0.0f;
  for (size_t warp = 0; warp < per_thread.size() / 32; ++warp) {
    float lanes[32];
    std::copy_n(per_thread.begin() + warp * 32, 32, lanes);
    for (int offset = 16; offset > 0; offset /= 2) {
      float next[32];
      for (int lane = 0; lane < 32; ++lane) {
        next[lane] = lanes[lane] + lanes[lane ^ offset];
      }
      std::copy_n(next, 32, lanes);
    }
    total += lanes[0];
  }
  return total;
}

// rsqrt(mean(v^2) + eps) of `count` values, with the kernels' summation.
template <typename Value>
float NormFactor(int count, float epsilon, Value value) {
  std::vector<float> per_thread(256, 0.0f);
  for (int i = 0; i < count; ++i) {
    const float v = value(i);
    const float square = v * v;
    per_thread[i % 256] += square;
  }
  return 1.0f /
         std::sqrt(BlockSum(per_thread) / static_cast<float>(count) + epsilon);
}

struct GlueResult {
  std::vector<float> x;  // x', FLOAT16 values
  std::vector<float> z;
  std::vector<int> q;
  std::vector<float> row_scale;
};

// LiteRtNvidiaLaunchQuantGlue. `x` holds the FLOAT16 values of x.
inline GlueResult Glue(const LiteRtNvidiaQuantGlueParams& p, int m,
                       const std::vector<float>& x,
                       const std::vector<int8_t>& y,
                       const std::vector<float>& branch_weights,
                       const std::vector<float>& norm_weights) {
  GlueResult result;
  const size_t total = static_cast<size_t>(m) * p.width;
  result.x.resize(total);
  result.z.resize(total);
  result.q.resize(total);
  result.row_scale.resize(m, 1.0f);
  for (int row = 0; row < m; ++row) {
    const size_t base = static_cast<size_t>(row) * p.width;
    std::vector<float> values(x.begin() + base, x.begin() + base + p.width);
    if (p.branch != 0) {
      std::vector<float> branch(p.width);
      for (int i = 0; i < p.width; ++i) {
        branch[i] =
            RoundToFp16(static_cast<float>(y[base + i]) * p.branch_scale);
      }
      const float factor =
          NormFactor(p.width, p.branch_eps, [&](int i) { return branch[i]; });
      for (int i = 0; i < p.width; ++i) {
        const float scaled = branch[i] * factor;
        const float normed = RoundToFp16(scaled * branch_weights[i]);
        values[i] = RoundToFp16(RoundToFp16(values[i] * p.c_x) +
                                RoundToFp16(normed * p.c_y));
      }
    }
    std::copy(values.begin(), values.end(), result.x.begin() + base);
    if (p.norm != 0) {
      const float factor =
          NormFactor(p.width, p.norm_eps, [&](int i) { return values[i]; });
      for (int i = 0; i < p.width; ++i) {
        const float scaled = values[i] * factor;
        values[i] = RoundToFp16(scaled * norm_weights[i]);
      }
    }
    std::copy(values.begin(), values.end(), result.z.begin() + base);
    if (p.quantize == kLiteRtNvidiaQuantGlueDynamic) {
      float largest = 0.0f;
      for (float value : values) largest = std::max(largest, std::fabs(value));
      const float inverse = largest == 0.0f ? 0.0f : 127.0f / largest;
      result.row_scale[row] = largest == 0.0f ? 1.0f : largest / 127.0f;
      for (int i = 0; i < p.width; ++i) {
        result.q[base + i] = static_cast<int>(
            std::clamp(std::round(values[i] * inverse), -127.0f, 127.0f));
      }
    } else if (p.quantize == kLiteRtNvidiaQuantGlueStatic) {
      for (int i = 0; i < p.width; ++i) {
        result.q[base + i] = Quantize(values[i], p.quantize_scale);
      }
    }
  }
  return result;
}

// The inverse timescales of odml.rope as the model's decomposition holds
// them: the timescale in doubles rounded to FLOAT32, then its reciprocal.
inline std::vector<float> InverseTimescales(int depth, float min_timescale,
                                            float max_timescale) {
  std::vector<float> inverse(depth / 2);
  for (int i = 0; i < depth / 2; ++i) {
    const float timescale = static_cast<float>(
        static_cast<double>(min_timescale) *
        std::pow(static_cast<double>(max_timescale) / min_timescale,
                 static_cast<double>(i) / (depth / 2)));
    inverse[i] = 1.0f / timescale;
  }
  return inverse;
}

struct QkvRopeResult {
  std::vector<float> q;  // [q_heads, tokens, depth]
  std::vector<float> k;  // [k_heads, tokens, depth]
  std::vector<float> v;  // [tokens, v_heads, depth]
};

// LiteRtNvidiaLaunchQuantQkvRope.
inline QkvRopeResult QkvRope(const LiteRtNvidiaQuantQkvRopeParams& p,
                             const std::vector<int8_t>& qkv,
                             const std::vector<int32_t>& positions,
                             const std::vector<float>& q_norm,
                             const std::vector<float>& k_norm,
                             const std::vector<float>& v_norm,
                             const std::vector<float>& inverse_timescales) {
  QkvRopeResult result;
  const int depth = p.depth;
  const int half = depth / 2;
  const int total_heads = p.q_heads + p.k_heads + p.v_heads;
  result.q.resize(static_cast<size_t>(p.q_heads) * p.tokens * depth);
  result.k.resize(static_cast<size_t>(p.k_heads) * p.tokens * depth);
  result.v.resize(static_cast<size_t>(p.v_heads) * p.tokens * depth);
  for (int token = 0; token < p.tokens; ++token) {
    for (int head = 0; head < total_heads; ++head) {
      const bool is_q = head < p.q_heads;
      const bool is_k = !is_q && head < p.q_heads + p.k_heads;
      const float scale = is_q ? p.q_scale : (is_k ? p.k_scale : p.v_scale);
      const std::vector<float>& weights =
          is_q ? q_norm : (is_k ? k_norm : v_norm);
      const int8_t* row =
          qkv.data() +
          (static_cast<size_t>(token) * total_heads + head) * depth;
      std::vector<float> values(depth);
      for (int i = 0; i < depth; ++i) {
        values[i] = RoundToFp16(static_cast<float>(row[i]) * scale);
      }
      const float factor =
          NormFactor(depth, p.epsilon, [&](int i) { return values[i]; });
      for (int i = 0; i < depth; ++i) {
        const float scaled = values[i] * factor;
        values[i] = RoundToFp16(scaled * weights[i]);
      }
      if (!is_q && !is_k) {
        std::copy(values.begin(), values.end(),
                  result.v.begin() + (static_cast<size_t>(token) * p.v_heads +
                                      (head - p.q_heads - p.k_heads)) *
                                         depth);
        continue;
      }
      float* out =
          is_q
              ? result.q.data() +
                    (static_cast<size_t>(head) * p.tokens + token) * depth
              : result.k.data() +
                    (static_cast<size_t>(head - p.q_heads) * p.tokens + token) *
                        depth;
      const float position = static_cast<float>(positions[token]);
      for (int i = 0; i < half; ++i) {
        const float angle = position * inverse_timescales[i];
        const float sine = RoundToFp16(std::sin(angle));
        const float cosine = RoundToFp16(std::cos(angle));
        const float x1 = values[i];
        const float x2 = values[i + half];
        float first =
            RoundToFp16(RoundToFp16(x1 * cosine) - RoundToFp16(x2 * sine));
        float second =
            RoundToFp16(RoundToFp16(x2 * cosine) + RoundToFp16(x1 * sine));
        if (is_q) {
          first = RoundToFp16(first * p.q_multiplier);
          second = RoundToFp16(second * p.q_multiplier);
        }
        out[i] = first;
        out[i + half] = second;
      }
    }
  }
  return result;
}

}  // namespace litert::nvidia::quant_reference

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_LINEAR_REFERENCE_H_
