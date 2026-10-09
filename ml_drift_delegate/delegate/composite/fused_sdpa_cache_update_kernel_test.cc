// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

// Rearranges logical K data [BK, S, H] into the CPU memory order required by
// TensorDescriptor::UploadData to produce the kOSpatialIOGroupO4I4 GPU layout.
// Copied from sdpa_transposed_kernel_test.cc. The mapping is a permutation, so
// it also maps a logical cache to the data downloaded from the GPU tensor.
absl::Status RearrangeK(const std::vector<float>& data,
                        std::vector<float>& rearranged_data,
                        const ::ml_drift::OHWI weights_shape) {
  if (data.size() != weights_shape.DimensionsProduct()) {
    return absl::InvalidArgumentError(
        "Raw data size does not match weights shape.");
  }
  const int S = weights_shape.o;
  const int BK = weights_shape.h;
  const int H = weights_shape.i;
  if (H % 4 != 0 || S % 4 != 0) {
    return absl::InvalidArgumentError("Dimensions must be multiples of 4.");
  }
  rearranged_data.assign(data.size(), 0.0f);
  for (int bk = 0; bk < BK; ++bk) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        int orig_idx = (bk * S + s) * H + h;
        int linear_head_batch = bk * (H / 4) + (h / 4);
        int bk_src = linear_head_batch % BK;
        int h_src_slice = linear_head_batch / BK;
        int h_src = h_src_slice * 4 + (h % 4);
        int rearranged_idx = (bk_src * S + s) * H + h_src;
        rearranged_data[rearranged_idx] = data[orig_idx];
      }
    }
  }
  return absl::OkStatus();
}

// Rearranges logical V data [BK, H, S] into the CPU memory order required by
// TensorDescriptor::UploadData to produce the kOSpatialIOGroupI4O4 GPU layout.
// Copied from sdpa_transposed_kernel_test.cc.
absl::Status RearrangeV(const std::vector<float>& data,
                        std::vector<float>& rearranged_data,
                        const ::ml_drift::OHWI weights_shape) {
  if (data.size() != weights_shape.DimensionsProduct()) {
    return absl::InvalidArgumentError(
        "Raw data size does not match weights shape.");
  }
  const int H = weights_shape.o;
  const int BK = weights_shape.h;
  const int S = weights_shape.i;
  if (H % 4 != 0 || S % 4 != 0) {
    return absl::InvalidArgumentError("Dimensions must be multiples of 4.");
  }
  rearranged_data.assign(data.size(), 0.0f);
  for (int bk = 0; bk < BK; ++bk) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int orig_idx = (bk * H + h) * S + s;
        int rhs = ((bk * (S / 4) + (s / 4)) * (H / 4) + (h / 4)) * 4 + (s % 4);
        int h_src = rhs % H;
        int temp = rhs / H;
        int bk_src = temp % BK;
        int s_src_slice = temp / BK;
        int s_src = s_src_slice * 4 + (h % 4);
        int rearranged_idx = (bk_src * H + h_src) * S + s_src;
        rearranged_data[rearranged_idx] = data[orig_idx];
      }
    }
  }
  return absl::OkStatus();
}

enum class MaskMode { kBool, kFloatAdditive };

std::string ToString(MaskMode mask_mode) {
  switch (mask_mode) {
    case MaskMode::kBool:
      return "BoolMask";
    case MaskMode::kFloatAdditive:
      return "FloatAdditiveMask";
  }
}

struct TestCase {
  int heads = 1;       // Hkv.
  int groups = 1;      // G, query heads per KV head.
  int new_len = 1;     // T.
  int cache_size = 4;  // W.
  int head_dim = 4;    // D.
  int start = 0;       // param[0].
  int end = 1;         // param[1].
  std::optional<float> softcap;
  bool update_cache = true;
  // Masks every column of the last mask row, as LiteRT-LM does for the pending
  // token that is held back from a prefill chunk.
  bool fully_mask_last_row = false;
};

// Deterministic values in [-1, 1).
std::vector<float> MakeData(int size, int seed) {
  std::vector<float> data(size);
  uint32_t state = 0x9E3779B9u * static_cast<uint32_t>(seed + 1);
  for (float& v : data) {
    state = state * 1664525u + 1013904223u;
    v = static_cast<float>((state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  }
  return data;
}

struct Inputs {
  std::vector<float> q;         // [H, R, D]
  std::vector<float> k_cache;   // [H, W, D]
  std::vector<float> v_cache;   // [H, D, W]
  std::vector<float> k_new;     // [H, T, D]
  std::vector<float> v_new;     // [H, D, T]
  std::vector<float> mask;      // [T, W + T], 1/0 for bool, additive else.
};

// Applies the ring buffer write of odml.fused_sdpa_cache_update to logical
// caches.
void ApplyRingWrite(const TestCase& c, const Inputs& in,
                    std::vector<float>& k_cache, std::vector<float>& v_cache) {
  const int H = c.heads, T = c.new_len, W = c.cache_size, D = c.head_dim;
  if (c.start < 0) return;
  const int valid = std::clamp(c.end - c.start, 0, T);
  const int first = std::max(0, valid - W);
  for (int x = first; x < valid; ++x) {
    const int slot = (c.start + x) % W;
    for (int h = 0; h < H; ++h) {
      for (int d = 0; d < D; ++d) {
        k_cache[(h * W + slot) * D + d] = in.k_new[(h * T + x) * D + d];
        v_cache[(h * D + d) * W + slot] = in.v_new[(h * D + d) * T + x];
      }
    }
  }
}

// Until the ring wraps, slot j holds position j, so only the slots below
// param[0] hold tokens.
int FilledSlots(const TestCase& c) {
  return std::clamp(c.start, 0, c.cache_size);
}

// CPU reference attention over [k_cache | k_new] and [v_cache | v_new].
std::vector<float> ReferenceAttention(const TestCase& c, MaskMode mask_mode,
                                      const Inputs& in,
                                      const std::vector<float>& k_cache,
                                      const std::vector<float>& v_cache) {
  const int H = c.heads, T = c.new_len, W = c.cache_size, D = c.head_dim;
  const int R = c.groups * T;
  const int L = W + T;
  // Empty ring slots (see FilledSlots) take no part in the softmax.
  const int filled_slots = FilledSlots(c);
  std::vector<float> out(H * R * D, 0.0f);
  std::vector<float> logits(L);
  for (int h = 0; h < H; ++h) {
    for (int r = 0; r < R; ++r) {
      const int t = r % T;  // Query rows are packed g-major.
      for (int j = 0; j < L; ++j) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
          const float k = j < W ? k_cache[(h * W + j) * D + d]
                                : in.k_new[(h * T + (j - W)) * D + d];
          acc += in.q[(h * R + r) * D + d] * k;
        }
        if (c.softcap.has_value()) {
          acc = std::tanh(acc / *c.softcap) * *c.softcap;
        }
        const float m = in.mask[t * L + j];
        if (mask_mode == MaskMode::kBool) {
          if (m == 0.0f) acc = -10000.0f;
        } else {
          acc += m;
        }
        logits[j] = j < W && j >= filled_slots
                        ? -std::numeric_limits<float>::infinity()
                        : acc;
      }
      const float max_logit = *std::max_element(logits.begin(), logits.end());
      float sum = 0.0f;
      for (float& l : logits) {
        l = std::exp(l - max_logit);
        sum += l;
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < L; ++j) {
          const float v = j < W ? v_cache[(h * D + d) * W + j]
                                : in.v_new[(h * D + d) * T + (j - W)];
          acc += logits[j] / sum * v;
        }
        out[(h * R + r) * D + d] = acc;
      }
    }
  }
  return out;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float result = 0.0f;
  for (int i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(a[i] - b[i]));
  }
  return result;
}

// Whether the ring write overwrites a slot that holds a token, so that reading
// the cache before or after the write can make a difference.
bool OverwritesFilledSlot(const TestCase& c) {
  if (c.start < 0) return false;
  const int valid = std::clamp(c.end - c.start, 0, c.new_len);
  for (int x = std::max(0, valid - c.cache_size); x < valid; ++x) {
    if ((c.start + x) % c.cache_size < FilledSlots(c)) return true;
  }
  return false;
}

absl::Status RunFusedSdpaCacheUpdateTest(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision,
    ::ml_drift::TensorStorageType storage, MaskMode mask_mode,
    const TestCase& c) {
  const int H = c.heads, T = c.new_len, W = c.cache_size, D = c.head_dim;
  const int R = c.groups * T;
  const int L = W + T;
  const ::ml_drift::DataType data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);
  constexpr auto kBuffer = ::ml_drift::TensorStorageType::kBuffer;

  ::ml_drift::GpuModelBuilder builder(env.GetGpuInfo(), {}, precision, storage);
  auto q = builder.AddTensor(::ml_drift::BHWC(1, H, R, D), data_type);
  // The KV caches come from odml.cache_update style buffers.
  auto k_cache = builder.AddTensor(1, H, W, D, kBuffer, data_type);
  auto v_cache = builder.AddTensor(1, H, D, W, kBuffer, data_type);
  auto k_new = builder.AddTensor(::ml_drift::BHWC(1, H, T, D), data_type);
  auto v_new = builder.AddTensor(::ml_drift::BHWC(1, H, D, T), data_type);
  auto mask_feed = builder.AddTensor(::ml_drift::BHWC(1, 1, T, L), data_type);
  auto mask = mask_mode == MaskMode::kBool
                  ? builder.Cast(mask_feed, ::ml_drift::DataType::kBool)
                  : mask_feed;

  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      param_cpu;
  param_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 7);
  param_cpu.data = {c.start, c.end, 0, 0, 0, 0, 0};
  ::ml_drift::TensorDescriptor param_desc(
      ::ml_drift::DataType::kInt32, kBuffer, ::ml_drift::Layout::kBHWC);
  param_desc.UploadData(param_cpu);
  auto param = builder.AddConstantTensor(std::move(param_desc));

  auto out = builder.AddTensor(::ml_drift::BHWC(1, H, R, D), data_type);
  std::vector<uint32_t> output_ids = {out.id};
  std::vector<std::pair<uint32_t, uint32_t>> model_outputs = {{out.id, 0}};
  if (c.update_cache) {
    auto k_cache_out = builder.AddTensor(1, H, W, D, kBuffer, data_type);
    auto v_cache_out = builder.AddTensor(1, H, D, W, kBuffer, data_type);
    output_ids.push_back(k_cache_out.id);
    output_ids.push_back(v_cache_out.id);
    model_outputs.push_back({k_cache_out.id, 1});
    model_outputs.push_back({v_cache_out.id, 2});
  }

  FusedSdpaCacheUpdateAttributes attr;
  attr.softcap = c.softcap;
  attr.update_cache = c.update_cache;
  ABSL_RETURN_IF_ERROR(BuildFusedSdpaCacheUpdateGpuGraph(
      {q.id, k_cache.id, v_cache.id, k_new.id, v_new.id, mask.id, param.id},
      output_ids, attr, &builder));

  ::ml_drift::GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(
      std::vector<std::pair<uint32_t, uint32_t>>{{q.id, 0},
                                                 {k_cache.id, 1},
                                                 {v_cache.id, 2},
                                                 {k_new.id, 3},
                                                 {v_new.id, 4},
                                                 {mask_feed.id, 5}},
      model_outputs, &gpu_model));

  if (c.update_cache) {
    // In LiteRT-LM the updated caches are bound in place to the input caches.
    // Emulate that aliasing by redirecting the cache write to the input cache
    // tensors: the write then keeps the unwritten slots, and any attention op
    // scheduled after it would observe the post-write cache.
    uint32_t k_in = 0, v_in = 0, k_out = 0, v_out = 0;
    for (const auto& [id, ref] : gpu_model.input_ids_and_refs) {
      if (ref == 1) k_in = id;
      if (ref == 2) v_in = id;
    }
    for (const auto& [id, ref] : gpu_model.output_ids_and_refs) {
      if (ref == 1) k_out = id;
      if (ref == 2) v_out = id;
    }
    // The cache write must be the last node and write both caches.
    const auto& last = gpu_model.nodes.back();
    EXPECT_THAT(last.outputs, ::testing::UnorderedElementsAre(k_out, v_out));
    int cache_readers = 0;
    for (int i = 0; i + 1 < gpu_model.nodes.size(); ++i) {
      for (uint32_t id : gpu_model.nodes[i].inputs) {
        if (id == k_in || id == v_in) ++cache_readers;
      }
      for (uint32_t id : gpu_model.nodes[i].outputs) {
        EXPECT_NE(id, k_out);
        EXPECT_NE(id, v_out);
      }
    }
    EXPECT_EQ(cache_readers, 2);
    for (auto& node : gpu_model.nodes) {
      for (uint32_t& id : node.outputs) {
        if (id == k_out) id = k_in;
        if (id == v_out) id = v_in;
      }
    }
    for (auto& [id, ref] : gpu_model.output_ids_and_refs) {
      if (ref == 1) id = k_in;
      if (ref == 2) id = v_in;
    }
  }

  Inputs in;
  in.q = MakeData(H * R * D, 1);
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (float& v : in.q) v *= q_scale;
  in.k_cache = MakeData(H * W * D, 2);
  in.v_cache = MakeData(H * D * W, 3);
  in.k_new = MakeData(H * T * D, 4);
  in.v_new = MakeData(H * D * T, 5);
  in.mask.resize(T * L);
  // Empty ring slots are always masked, as LiteRT-LM does.
  const int filled_slots = FilledSlots(c);
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < L; ++j) {
      // Filled cache slots: a pseudo-random pattern. New tokens: causal. New
      // token 0 is attended so that no row is fully masked, unless the case
      // asks for a fully masked last row.
      const bool attend =
          !(c.fully_mask_last_row && t == T - 1) &&
          (j < W ? j < filled_slots && (j * 5 + t * 3) % 4 != 0 : (j - W) <= t);
      if (mask_mode == MaskMode::kBool) {
        in.mask[t * L + j] = attend ? 1.0f : 0.0f;
      } else {
        in.mask[t * L + j] =
            attend ? -0.25f * static_cast<float>((j + t) % 3) : -10000.0f;
      }
    }
  }

  std::vector<float> k_cache_post = in.k_cache;
  std::vector<float> v_cache_post = in.v_cache;
  ApplyRingWrite(c, in, k_cache_post, v_cache_post);
  const std::vector<float> expected_out =
      ReferenceAttention(c, mask_mode, in, in.k_cache, in.v_cache);

  const bool f16 = data_type == ::ml_drift::DataType::kFloat16;
  const float out_tolerance = f16 ? (D > 64 ? 2e-2f : 1e-2f) : 2e-3f;
  const float cache_tolerance = f16 ? 1e-3f : 1e-6f;

  // Make sure the data can tell a pre-write read from a post-write read. With
  // large caches a few written slots barely move the output, so this is only
  // enforced for the small cases.
  if (c.update_cache && OverwritesFilledSlot(c) && W <= 64) {
    const std::vector<float> post_write_out =
        ReferenceAttention(c, mask_mode, in, k_cache_post, v_cache_post);
    EXPECT_GT(MaxAbsDiff(expected_out, post_write_out), 5 * out_tolerance)
        << "Test data does not distinguish pre- and post-write caches.";
  }

  const ::ml_drift::OHWI k_weights_shape(W, H, 1, D);
  const ::ml_drift::OHWI v_weights_shape(D, H, 1, W);
  auto make_tensor = [](const ::ml_drift::BHWC& shape,
                        std::vector<float> data) {
    ::ml_drift::TensorFloat32 t;
    t.shape = shape;
    t.data = std::move(data);
    return t;
  };
  std::vector<float> k_cache_packed, v_cache_packed;
  ABSL_RETURN_IF_ERROR(RearrangeK(in.k_cache, k_cache_packed, k_weights_shape));
  ABSL_RETURN_IF_ERROR(RearrangeV(in.v_cache, v_cache_packed, v_weights_shape));
  std::vector<::ml_drift::TensorFloat32> src_cpu = {
      make_tensor(::ml_drift::BHWC(1, H, R, D), in.q),
      make_tensor(::ml_drift::BHWC(1, H, W, D), k_cache_packed),
      make_tensor(::ml_drift::BHWC(1, H, D, W), v_cache_packed),
      make_tensor(::ml_drift::BHWC(1, H, T, D), in.k_new),
      make_tensor(::ml_drift::BHWC(1, H, D, T), in.v_new),
      make_tensor(::ml_drift::BHWC(1, 1, T, L), in.mask)};

  ::ml_drift::TensorFloat32 out_cpu =
      make_tensor(::ml_drift::BHWC(1, H, R, D), {});
  ::ml_drift::TensorFloat32 k_cache_cpu =
      make_tensor(::ml_drift::BHWC(1, H, W, D), {});
  ::ml_drift::TensorFloat32 v_cache_cpu =
      make_tensor(::ml_drift::BHWC(1, H, D, W), {});
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_cpu};
  if (c.update_cache) {
    dst_cpu.push_back(&k_cache_cpu);
    dst_cpu.push_back(&v_cache_cpu);
  }
  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model));

  EXPECT_THAT(out_cpu.data, ::testing::Pointwise(
                                ::testing::FloatNear(out_tolerance),
                                expected_out));
  if (c.update_cache) {
    std::vector<float> expected_k_packed, expected_v_packed;
    ABSL_RETURN_IF_ERROR(
        RearrangeK(k_cache_post, expected_k_packed, k_weights_shape));
    ABSL_RETURN_IF_ERROR(
        RearrangeV(v_cache_post, expected_v_packed, v_weights_shape));
    EXPECT_THAT(k_cache_cpu.data, ::testing::Pointwise(
                                      ::testing::FloatNear(cache_tolerance),
                                      expected_k_packed));
    EXPECT_THAT(v_cache_cpu.data, ::testing::Pointwise(
                                      ::testing::FloatNear(cache_tolerance),
                                      expected_v_packed));
  }
  return absl::OkStatus();
}

class FusedSdpaCacheUpdateKernelTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<
          std::tuple<::ml_drift::CalculationsPrecision,
                     ::ml_drift::TensorStorageType, MaskMode>> {
 public:
  void SetUp() override {
    if (!exec_env) {
      GTEST_SKIP() << "TestExecutionEnvironment not initialized.";
    }
    const auto data_type = ::ml_drift::DeduceDataTypeFromPrecision(precision());
    if (!exec_env->IsStorageSupported(storage(), data_type) ||
        !exec_env->IsStorageSupported(::ml_drift::TensorStorageType::kBuffer,
                                      data_type)) {
      GTEST_SKIP() << "Unsupported data type: "
                   << ::ml_drift::ToString(data_type)
                   << " storage type: " << ::ml_drift::ToString(storage());
    }
  }

 protected:
  ::ml_drift::CalculationsPrecision precision() const {
    return std::get<0>(GetParam());
  }
  ::ml_drift::TensorStorageType storage() const {
    return std::get<1>(GetParam());
  }
  MaskMode mask_mode() const { return std::get<2>(GetParam()); }

  void Run(const TestCase& c) {
    const absl::Status status = RunFusedSdpaCacheUpdateTest(
        *exec_env, precision(), storage(), mask_mode(), c);
    EXPECT_TRUE(status.ok()) << status;
  }
};

TEST_P(FusedSdpaCacheUpdateKernelTest, DecodeGroupedQuery) {
  Run({.heads = 1, .groups = 4, .new_len = 1, .cache_size = 16,
       .head_dim = 8, .start = 5, .end = 6});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, PrefillMultiHeadGrouped) {
  Run({.heads = 2, .groups = 2, .new_len = 4, .cache_size = 16,
       .head_dim = 16, .start = 3, .end = 7});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, NewTokensExceedCacheSize) {
  // T > W: only the last W valid tokens are written, wrapping around.
  Run({.heads = 2, .groups = 2, .new_len = 8, .cache_size = 4,
       .head_dim = 8, .start = 2, .end = 10});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, PaddedChunk) {
  // Only 3 of the 8 new tokens are valid.
  Run({.heads = 1, .groups = 2, .new_len = 8, .cache_size = 16,
       .head_dim = 8, .start = 10, .end = 13});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, PaddedChunkExceedingCacheSize) {
  // 6 of 8 tokens are valid, W = 4: tokens 2..5 are written.
  Run({.heads = 1, .groups = 2, .new_len = 8, .cache_size = 4,
       .head_dim = 8, .start = 7, .end = 13});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, Wraparound) {
  Run({.heads = 2, .groups = 1, .new_len = 4, .cache_size = 8,
       .head_dim = 8, .start = 6, .end = 10});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, Softcap) {
  Run({.heads = 1, .groups = 2, .new_len = 4, .cache_size = 8,
       .head_dim = 16, .start = 1, .end = 5, .softcap = 0.5f});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, NoValidTokens) {
  Run({.heads = 1, .groups = 2, .new_len = 4, .cache_size = 8,
       .head_dim = 8, .start = 5, .end = 5});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, NegativeStartSkipsWrite) {
  Run({.heads = 1, .groups = 2, .new_len = 4, .cache_size = 8,
       .head_dim = 8, .start = -1, .end = 3});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, AttentionOnly) {
  Run({.heads = 2, .groups = 2, .new_len = 4, .cache_size = 8,
       .head_dim = 8, .start = 2, .end = 6, .update_cache = false});
}

// LiteRT-LM holds back the pending last token of a prefill chunk, leaving its
// bool mask row all false while param still marks all T tokens valid. Masked
// logits are replaced (not offset) by the fill value, so such a row must be the
// uniform average of the value columns of all filled slots and new tokens.
TEST_P(FusedSdpaCacheUpdateKernelTest, FullyMaskedRow) {
  if (mask_mode() != MaskMode::kBool) {
    GTEST_SKIP() << "An all -10000 additive row is not representable in F16 "
                    "and is not produced by LiteRT-LM.";
  }
  Run({.heads = 2, .groups = 2, .new_len = 8, .cache_size = 16,
       .head_dim = 16, .start = 12, .end = 20, .fully_mask_last_row = true});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, Gemma3LikeDecode) {
  Run({.heads = 1, .groups = 4, .new_len = 1, .cache_size = 512,
       .head_dim = 256, .start = 700, .end = 701});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, Gemma3LikePrefill) {
  Run({.heads = 1, .groups = 4, .new_len = 32, .cache_size = 512,
       .head_dim = 256, .start = 500, .end = 532});
}

// The cases below are large enough for the softmax to split every row across
// several work items of a work group.
TEST_P(FusedSdpaCacheUpdateKernelTest, RaggedChunkSplitRows) {
  // T = 6 is not a multiple of 4, and the 18 rows only partly fill a group.
  Run({.heads = 2,
       .groups = 3,
       .new_len = 6,
       .cache_size = 64,
       .head_dim = 8,
       .start = 66,
       .end = 72});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, SoftcapSplitRows) {
  Run({.heads = 1,
       .groups = 4,
       .new_len = 8,
       .cache_size = 128,
       .head_dim = 16,
       .start = 3,
       .end = 11,
       .softcap = 0.5f});
}

// The first chunk (param[0] = 0) skips the cache, and a partly filled ring only
// reads the filled slots (rounded up to 32), with garbage in the others.
TEST_P(FusedSdpaCacheUpdateKernelTest, FirstChunkSkipsCache) {
  Run({.heads = 1,
       .groups = 4,
       .new_len = 8,
       .cache_size = 128,
       .head_dim = 16,
       .start = 0,
       .end = 8});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, PartlyFilledRing) {
  Run({.heads = 2,
       .groups = 2,
       .new_len = 8,
       .cache_size = 128,
       .head_dim = 8,
       .start = 40,
       .end = 48});
}

// Enough query rows for the FullyConnected ops to take the convolution path.
TEST_P(FusedSdpaCacheUpdateKernelTest, LargeFirstChunk) {
  Run({.heads = 1,
       .groups = 4,
       .new_len = 256,
       .cache_size = 512,
       .head_dim = 64,
       .start = 0,
       .end = 256});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, Gemma3LikeFirstChunk) {
  Run({.heads = 1,
       .groups = 4,
       .new_len = 640,
       .cache_size = 512,
       .head_dim = 256,
       .start = 0,
       .end = 640});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, LargePartlyFilledRing) {
  Run({.heads = 1,
       .groups = 4,
       .new_len = 256,
       .cache_size = 512,
       .head_dim = 64,
       .start = 100,
       .end = 356});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, LargeWrappedRing) {
  Run({.heads = 1,
       .groups = 4,
       .new_len = 256,
       .cache_size = 512,
       .head_dim = 64,
       .start = 700,
       .end = 956});
}

TEST_P(FusedSdpaCacheUpdateKernelTest, FullyMaskedRowSplitRows) {
  if (mask_mode() != MaskMode::kBool) {
    GTEST_SKIP() << "An all -10000 additive row is not representable in F16 "
                    "and is not produced by LiteRT-LM.";
  }
  Run({.heads = 1,
       .groups = 2,
       .new_len = 8,
       .cache_size = 128,
       .head_dim = 16,
       .start = 120,
       .end = 128,
       .fully_mask_last_row = true});
}

INSTANTIATE_TEST_SUITE_P(
    FusedSdpaCacheUpdateKernelTestSuite, FusedSdpaCacheUpdateKernelTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::kF32,
                      ::ml_drift::CalculationsPrecision::kF16}),
            ValuesIn({::ml_drift::TensorStorageType::kTexture2D,
                      ::ml_drift::TensorStorageType::kBuffer}),
            ValuesIn({MaskMode::kBool, MaskMode::kFloatAdditive})),
    [](const TestParamInfo<FusedSdpaCacheUpdateKernelTest::ParamType>& info) {
      std::string name =
          absl::StrCat(::ml_drift::ToString(std::get<0>(info.param)), "_",
                       ::ml_drift::ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

}  // namespace
}  // namespace litert::ml_drift
