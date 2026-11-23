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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_ATTENTION_REFERENCE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_ATTENTION_REFERENCE_H_

// Host references for the tests of the composites over quantized KV caches
// (odml.cache_update, odml.sdpa_transposed and odml.fused_sdpa_cache_update
// with INT8 queries over INT8 or INT4 caches). They follow the composites'
// decomposition subgraphs op by op, with the tensor types those have: FP16
// where a tensor is FLOAT16, fp32 products, TFLite's QUANTIZE rounding.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace litert::nvidia::quant_reference {

// Round to nearest even; the values of the tests stay inside the FP16 range.
inline uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112;
  uint32_t mantissa = bits & 0x7FFFFF;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);
  }
  uint32_t half =
      sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t rest = mantissa & 0x1FFF;
  if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) {
    ++half;
  }
  return static_cast<uint16_t>(half);
}

inline float HalfBitsToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
  const uint32_t exponent = (half >> 10) & 0x1F;
  const uint32_t mantissa = half & 0x3FF;
  if (exponent == 0) {
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return (half & 0x8000) ? -magnitude : magnitude;
  }
  uint32_t bits = exponent == 31
                      ? sign | 0x7F800000 | (mantissa << 13)
                      : sign | ((exponent + 112) << 23) | (mantissa << 13);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

inline float Bf16BitsToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// A value held in a FLOAT16 tensor.
inline float ThroughHalf(float value) {
  return HalfBitsToFloat(FloatToHalfBits(value));
}

// QUANTIZE with a zero point of 0, as the CPU backend computes it for the
// graphs' FLOAT16 tensors. To INT8 XNNPACK converts (xnn_f16_qs8_vcvt, whose
// scale is xnn_init_f16_qs8_cvt_scalar_params'): the value times the
// reciprocal of the scale rounded to FLOAT16, rounded half to even, saturated
// to [-128, 127]. To INT4, which XNNPACK does not take, TFLite's kernel
// divides (AffineQuantizeToInt4 in tflite/kernels/quantize.cc): the fp32
// quotient rounded half away from zero, saturated to [-8, 7].
inline int8_t Quantize(float value, float scale, int bits) {
  if (bits == 8) {
    const float reciprocal = ThroughHalf(1.0f / scale);
    const float product = value * reciprocal;
    const float rounded = std::nearbyint(product);
    return static_cast<int8_t>(std::min(std::max(rounded, -128.0f), 127.0f));
  }
  const int32_t rounded = static_cast<int32_t>(std::round(value / scale));
  return static_cast<int8_t>(std::min(std::max(rounded, -8), 7));
}

// exp(x) of x <= 0 as xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2
// computes it in fp32: a second-degree polynomial, and zero where a FLOAT16
// result would be denormal (below -9.703125).
inline float XnnHalfExpMinus(float x) {
  const float magic_bias = 0x1.8000FEp23f;
  float n = std::fma(x, 0x1.715476p0f, magic_bias);
  uint32_t bits = 0;
  std::memcpy(&bits, &n, sizeof(bits));
  bits <<= 23;
  float s = 0.0f;
  std::memcpy(&s, &bits, sizeof(s));
  n = n - magic_bias;
  float t = std::fma(n, -0x1.62E43p-1f, x);
  const float p = std::fma(0x1.FF3A32p-2f, t, 0x1.039E10p+0f);
  t = t * s;
  const float f = std::fma(t, p, s);
  return x < -0x1.368000p+3f ? 0.0f : f;
}

// SOFTMAX on a FLOAT16 tensor, in place (XNNPACK's softmax operator with the
// microkernels x86 gets with AVX2 and F16C): the maximum; exp(x - max) as
// above, each rounded to FLOAT16; the fp32 sum of the unrounded values as
// the microkernel adds them (eight lanes over the whole vectors, the lanes
// in pairs, the last keys of a length that is no multiple of eight into the
// pairs, then (0 + 2) + (1 + 3)); the reciprocal of the sum rounded to
// FLOAT16 (compute_reciprocal_f16); the products rounded to FLOAT16
// (xnn_f16_vmulc).
inline void HalfSoftmax(std::vector<float>* values) {
  std::vector<float>& x = *values;
  const int n = static_cast<int>(x.size());
  float max = -INFINITY;
  for (const float value : x) {
    max = std::max(max, value);
  }
  const int whole = n & ~7;
  float lanes[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < n; ++j) {
    x[j] = XnnHalfExpMinus(x[j] - max);
    if (j < whole) {
      lanes[j & 7] = lanes[j & 7] + x[j];
    }
  }
  float pair[4];
  for (int i = 0; i < 4; ++i) {
    pair[i] = lanes[i] + lanes[i + 4];
  }
  int next = whole;
  const int rest = n - whole;
  if ((rest & 4) != 0) {
    for (int i = 0; i < 4; ++i) {
      pair[i] = pair[i] + x[next + i];
    }
    next += 4;
  }
  if ((rest & 2) != 0) {
    pair[0] = pair[0] + x[next];
    pair[1] = pair[1] + x[next + 1];
    next += 2;
  }
  if ((rest & 1) != 0) {
    pair[0] = pair[0] + x[next];
  }
  const float low = pair[0] + pair[2];
  const float high = pair[1] + pair[3];
  const float sum = low + high;
  const float scale = ThroughHalf(1.0f / sum);
  for (int j = 0; j < n; ++j) {
    const float product = ThroughHalf(x[j]) * scale;
    x[j] = ThroughHalf(product);
  }
}

// INT4 values two per byte, the first in the low nibble, as LiteRT packs an
// INT4 tensor (tensor_utils::PackInt8IntoDenseInt).
inline std::vector<uint8_t> PackInt4(const std::vector<int8_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    packed[i / 2] |= i % 2 == 0 ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

inline std::vector<int8_t> UnpackInt4(const std::vector<uint8_t>& packed,
                                      size_t count) {
  std::vector<int8_t> values(count);
  for (size_t i = 0; i < count; ++i) {
    const uint8_t nibble = (packed[i / 2] >> (i % 2 == 0 ? 0 : 4)) & 0x0F;
    values[i] = static_cast<int8_t>(static_cast<int8_t>(nibble << 4) >> 4);
  }
  return values;
}

// The values of a key cache [1, heads, cache_len, depth] and of a value cache
// [1, heads, depth, cache_len], one per element whatever the tensors' type.
struct Caches {
  int heads = 0;
  int cache_len = 0;
  int depth = 0;
  std::vector<int8_t> k;
  std::vector<int8_t> v;

  Caches(int heads, int cache_len, int depth)
      : heads(heads),
        cache_len(cache_len),
        depth(depth),
        k(static_cast<size_t>(heads) * cache_len * depth, 0),
        v(static_cast<size_t>(heads) * cache_len * depth, 0) {}

  int8_t& Key(int head, int slot, int dim) {
    return k[(static_cast<size_t>(head) * cache_len + slot) * depth + dim];
  }
  int8_t& Value(int head, int slot, int dim) {
    return v[(static_cast<size_t>(head) * depth + dim) * cache_len + slot];
  }
  int8_t Key(int head, int slot, int dim) const {
    return k[(static_cast<size_t>(head) * cache_len + slot) * depth + dim];
  }
  int8_t Value(int head, int slot, int dim) const {
    return v[(static_cast<size_t>(head) * depth + dim) * cache_len + slot];
  }
};

// odml.cache_update: its inputs are re-quantized with their own scales (an
// identity) and written by two DYNAMIC_UPDATE_SLICE ops, which clamp their
// start so that the update fits. k_rows and v_rows are the quantized keys and
// values of `tokens` tokens, [heads][tokens][depth].
inline void CacheUpdate(Caches* caches, const std::vector<int8_t>& k_rows,
                        const std::vector<int8_t>& v_rows, int tokens,
                        int k_position, int v_position) {
  const int last = caches->cache_len - tokens;
  const int k_start = std::min(std::max(k_position, 0), last);
  const int v_start = std::min(std::max(v_position, 0), last);
  for (int h = 0; h < caches->heads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      for (int d = 0; d < caches->depth; ++d) {
        const size_t from =
            (static_cast<size_t>(h) * tokens + t) * caches->depth + d;
        caches->Key(h, k_start + t, d) = k_rows[from];
        caches->Value(h, v_start + t, d) = v_rows[from];
      }
    }
  }
}

// The cache update of odml.fused_sdpa_cache_update (ops 33 to 69 of its
// decomposition): params[0] and params[1] are the positions of the chunk's
// first token and after its last one.
inline void RingUpdate(Caches* caches, const std::vector<int8_t>& k_rows,
                       const std::vector<int8_t>& v_rows, int tokens, int start,
                       int end) {
  const int n = std::min(std::max(end - start, 0), tokens);  // jit(clip)
  const int last = start + n - 1;                            // sub73
  for (int slot = 0; slot < caches->cache_len; ++slot) {     // iota
    int back = (last - slot) % caches->cache_len;            // FLOOR_MOD
    if (back < 0) {
      back += caches->cache_len;
    }
    const int index = n - 1 - back;             // sub76
    if (!(n > 0 && index >= 0 && index < n)) {  // and1
      continue;
    }
    const int token = std::min(std::max(index, 0), tokens - 1);
    for (int h = 0; h < caches->heads; ++h) {
      for (int d = 0; d < caches->depth; ++d) {
        const size_t from =
            (static_cast<size_t>(h) * tokens + token) * caches->depth + d;
        caches->Key(h, slot, d) = k_rows[from];
        caches->Value(h, slot, d) = v_rows[from];
      }
    }
  }
}

struct AttentionInputs {
  // The quantized query, [heads][rows][depth], and the scales.
  const std::vector<int8_t>* q = nullptr;
  int rows = 0;
  float q_scale = 1.0f;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  const Caches* caches = nullptr;
  // The chunk's quantized keys and values after the cache's, [heads]
  // [new_len][depth] (odml.fused_sdpa_cache_update), or null.
  const std::vector<int8_t>* k_new = nullptr;
  const std::vector<int8_t>* v_new = nullptr;
  int new_len = 0;
  // [mask_rows][cache_len + new_len]; query row r reads row r % mask_rows.
  const std::vector<uint8_t>* mask = nullptr;
  int mask_rows = 1;
  float fill = 0.0f;
};

// The attention of odml.sdpa_transposed (ops 0 to 16 of its decomposition)
// and of odml.fused_sdpa_cache_update (ops 0 to 32), tensor by tensor as the
// CPU backend computes them (XNNPACK on x86 with AVX2, FMA3 and F16C): the
// query dequantized to FP16 with the scale rounded to FP16 (xnn_qs8_f16_vcvt);
// the keys dequantized to fp32; the scores one fused multiply-add per depth
// element (xnn_f32_gemm), cast to FP16 and masked with `fill`; the softmax,
// which odml.sdpa_transposed takes on the FP16 tensor (HalfSoftmax) and the
// fused composite in fp32 between two casts; fp32 products with the
// dequantized values, one fused multiply-add per key in the cache's order;
// and the result cast to FP16 (the fused composite casts the cache's and the
// chunk's parts and adds them in FP16). Returns [heads][rows][depth].
//
// The kernels of the first composite (decode_attention.cu) have these bits.
// The tensor-core kernel of the second (tiled_attention.cu) multiplies FP16
// operands and folds the softmax in tile by tile: it agrees to RowsMatch.
inline std::vector<float> Attention(const AttentionInputs& in) {
  const Caches& caches = *in.caches;
  const int depth = caches.depth;
  const int keys = caches.cache_len + in.new_len;
  std::vector<float> out(static_cast<size_t>(caches.heads) * in.rows * depth);
  std::vector<float> q(depth);
  std::vector<float> scores(keys);
  std::vector<float> cache_part(depth);
  std::vector<float> new_part(depth);
  for (int h = 0; h < caches.heads; ++h) {
    for (int r = 0; r < in.rows; ++r) {
      const size_t row = (static_cast<size_t>(h) * in.rows + r) * depth;
      const float q_scale = ThroughHalf(in.q_scale);
      for (int d = 0; d < depth; ++d) {
        const float product = static_cast<float>((*in.q)[row + d]) * q_scale;
        q[d] = ThroughHalf(product);
      }
      const uint8_t* mask_row =
          in.mask->data() + static_cast<size_t>(r % in.mask_rows) * keys;
      float max_score = -INFINITY;
      for (int j = 0; j < keys; ++j) {
        float score = 0.0f;
        for (int d = 0; d < depth; ++d) {
          const int8_t key =
              j < caches.cache_len
                  ? caches.Key(h, j, d)
                  : (*in.k_new)[(static_cast<size_t>(h) * in.new_len + j -
                                 caches.cache_len) *
                                    depth +
                                d];
          const float dequantized = static_cast<float>(key) * in.k_scale;
          score = std::fma(q[d], dequantized, score);
        }
        scores[j] = mask_row[j] != 0 ? ThroughHalf(score) : in.fill;
        max_score = std::max(max_score, scores[j]);
      }
      if (in.new_len == 0) {
        HalfSoftmax(&scores);
      } else {
        float sum = 0.0f;
        for (int j = 0; j < keys; ++j) {
          scores[j] = std::exp(scores[j] - max_score);
          sum += scores[j];
        }
        for (int j = 0; j < keys; ++j) {
          scores[j] = ThroughHalf(scores[j] / sum);
        }
      }
      std::fill(cache_part.begin(), cache_part.end(), 0.0f);
      std::fill(new_part.begin(), new_part.end(), 0.0f);
      for (int j = 0; j < keys; ++j) {
        const float p = scores[j];
        if (p == 0.0f) {
          continue;
        }
        for (int d = 0; d < depth; ++d) {
          if (j < caches.cache_len) {
            const float dequantized =
                static_cast<float>(caches.Value(h, j, d)) * in.v_scale;
            cache_part[d] = std::fma(p, dequantized, cache_part[d]);
          } else {
            const float dequantized =
                static_cast<float>(
                    (*in.v_new)[(static_cast<size_t>(h) * in.new_len + j -
                                 caches.cache_len) *
                                    depth +
                                d]) *
                in.v_scale;
            new_part[d] = std::fma(p, dequantized, new_part[d]);
          }
        }
      }
      for (int d = 0; d < depth; ++d) {
        out[row + d] = in.new_len == 0
                           ? ThroughHalf(cache_part[d])
                           : ThroughHalf(ThroughHalf(cache_part[d]) +
                                         ThroughHalf(new_part[d]));
      }
    }
  }
  return out;
}

// The rows of `depth` values that differ from the reference by more than
// `tolerance` somewhere, and the largest difference of all.
struct RowErrors {
  int rows = 0;
  int rows_off = 0;
  float max_error = 0.0f;
};

inline RowErrors CompareRows(const std::vector<float>& actual,
                             const std::vector<float>& expected, int depth,
                             float tolerance) {
  RowErrors errors;
  errors.rows = static_cast<int>(expected.size() / depth);
  for (int row = 0; row < errors.rows; ++row) {
    float row_error = 0.0f;
    for (int d = 0; d < depth; ++d) {
      const size_t i = static_cast<size_t>(row) * depth + d;
      row_error = std::max(row_error, std::abs(actual[i] - expected[i]));
    }
    errors.rows_off += row_error > tolerance ? 1 : 0;
    errors.max_error = std::max(errors.max_error, row_error);
  }
  return errors;
}

// The decompositions cast their scores to FP16, whose steps at the size of
// these scores are 0.016 to 0.06. A score that two orders of fp32 summation
// put on different sides of such a step moves the weight of one key by a few
// percent, and with it a whole row of the result by up to 0.1 when that key
// matters. So an implementation agrees with another to `tolerance` except in
// a rare row: at most one in 256 here, none by more than 0.25.
inline bool RowsMatch(const RowErrors& errors) {
  return errors.rows_off <= (errors.rows + 255) / 256 &&
         errors.max_error < 0.25f;
}

}  // namespace litert::nvidia::quant_reference

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_ATTENTION_REFERENCE_H_
