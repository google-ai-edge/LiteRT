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
#include <memory>
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
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.h"
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
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
  ABSL_RETURN_IF_ERROR(BuildSdpaCacheUpdateGpuGraph(
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

class ReadLinearBytesOp : public ::ml_drift::GPUOperation {
 public:
  explicit ReadLinearBytesOp(int total_vec4s) : total_vec4s_(total_vec4s) {}
  ReadLinearBytesOp(ReadLinearBytesOp&&) = default;
  ReadLinearBytesOp& operator=(ReadLinearBytesOp&&) = default;
  ReadLinearBytesOp(const ReadLinearBytesOp&) = delete;
  ReadLinearBytesOp& operator=(const ReadLinearBytesOp&) = delete;

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(total_vec4s_, 1, 1);
  }

 private:
  int total_vec4s_ = 1;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateReadLinearBytesToFloatOp(
    const ::ml_drift::TensorDescriptor& src,
    const ::ml_drift::TensorDescriptor& keep_alive,
    const ::ml_drift::TensorDescriptor& dst, int total_vec4s) {
  ReadLinearBytesOp op(total_vec4s);
  op.AddSrcTensor("src", src);
  op.AddSrcTensor("keep_alive", keep_alive);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("total_vec4s", total_vec4s);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  if (X >= args.total_vec4s) {
    return;
  }
  float4 v = ucl::Convert<float4>(args.src.Read(X));
  args.dst::type out_v = ucl::Convert<args.dst::type>(v);
  args.dst.WriteLinear(out_v, X);
}
)";
  return std::make_unique<ReadLinearBytesOp>(std::move(op));
}

// Runs `CreateAddValuesToCacheFromNode` (seeding a quantized INT8 or packed
// INT4 ring cache) followed by `BuildSdpaCacheUpdateGpuGraph` (with float or
// INT8 `query` and `key_new`/`value_new`), verifying both the attention output
// and in-place ring-buffer cache updates.
absl::Status RunQuantizedFusedSdpaCacheUpdateTest(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision,
    ::ml_drift::TensorStorageType storage, MaskMode mask_mode,
    const TestCase& c, int kv_quant_bits, bool int8_query, bool int8_slices,
    bool float16_output = false, bool float16_slices = false) {
  const int H = c.heads, T = c.new_len, W = c.cache_size, D = c.head_dim;
  const int R = c.groups * T;
  const int L = W + T;
  const ::ml_drift::DataType data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);
  constexpr auto kBuffer = ::ml_drift::TensorStorageType::kBuffer;

  const float kScaleK = (kv_quant_bits == 4) ? 0.0625f : 0.0078125f;
  const float kScaleV = (kv_quant_bits == 4) ? 0.125f : 0.0078125f;
  const float scale_q = 0.03125f / std::sqrt(static_cast<float>(D));
  const float kScaleKn = kScaleK;
  const float kScaleVn = kScaleV;

  using TensorHandle = ::ml_drift::GpuModelBuilder::TensorHandle;
  ::ml_drift::GpuModelBuilder builder(env.GetGpuInfo(), {}, precision, storage);
  TensorHandle query_feed =
      builder.AddTensor(::ml_drift::BHWC(1, H, R, D), data_type);
  TensorHandle query = query_feed;
  if (int8_query) {
    TensorHandle query_int8 =
        builder.AddTensor(1, H, R, D, kBuffer, ::ml_drift::DataType::kInt8);
    TensorHandle query_cast =
        builder.Cast(query_feed, ::ml_drift::DataType::kInt8);
    ABSL_RETURN_IF_ERROR(builder.UpdateOutputTensor(query_cast, query_int8.id));
    query = query_int8;
  }

  TensorHandle init_src_k =
      builder.AddTensor(::ml_drift::BHWC(1, H, W, D), data_type);
  TensorHandle init_src_v =
      builder.AddTensor(::ml_drift::BHWC(1, H, W, D), data_type);
  const int packed_d = (kv_quant_bits == 4) ? (D / 2) : D;
  TensorHandle k_cache = builder.AddTensor(1, H, W, packed_d, kBuffer,
                                           ::ml_drift::DataType::kUint8);
  TensorHandle v_cache = builder.AddTensor(1, H, packed_d, W, kBuffer,
                                           ::ml_drift::DataType::kUint8);

  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      init_param_cpu;
  init_param_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 7);
  init_param_cpu.data = {0, W, W, W, 0, 0, 0};
  ::ml_drift::TensorDescriptor init_param_desc(
      ::ml_drift::DataType::kInt32, kBuffer, ::ml_drift::Layout::kBHWC);
  init_param_desc.UploadData(init_param_cpu);
  TensorHandle init_param =
      builder.AddConstantTensor(std::move(init_param_desc));

  ::ml_drift::OperationDef init_cache_def;
  init_cache_def.src_tensors = {init_src_k.tensor_desc, init_src_v.tensor_desc,
                                init_param.tensor_desc};
  init_cache_def.dst_tensors = {k_cache.tensor_desc, v_cache.tensor_desc};
  AddValuesToCacheAttributes init_cache_attr;
  init_cache_attr.kv_cache_batch_size = H;
  init_cache_attr.cache_size = W;
  init_cache_attr.head_size = D;
  init_cache_attr.kv_quant_bits = kv_quant_bits;
  init_cache_attr.scale_k = kScaleK;
  init_cache_attr.scale_v = kScaleV;
  ::ml_drift::ir::IrOp init_cache_ir(::ml_drift::ir::IrOpId{0});
  init_cache_ir.name = kAddValuesToCacheType;
  init_cache_ir.attr = init_cache_attr;
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<::ml_drift::GPUOperation> init_cache_op,
      CreateAddValuesToCacheFromNode(init_cache_def, init_cache_ir));
  builder.AddGpuOperation({init_src_k, init_src_v, init_param},
                          {k_cache, v_cache}, std::move(init_cache_op),
                          kAddValuesToCacheType);

  TensorHandle k_new_feed =
      builder.AddTensor(::ml_drift::BHWC(1, H, T, D), data_type);
  TensorHandle v_new_feed =
      builder.AddTensor(::ml_drift::BHWC(1, H, D, T), data_type);
  TensorHandle k_new = k_new_feed;
  TensorHandle v_new = v_new_feed;
  if (int8_slices) {
    TensorHandle key_new_int8 =
        builder.AddTensor(1, H, T, D, kBuffer, ::ml_drift::DataType::kInt8);
    TensorHandle key_new_cast =
        builder.Cast(k_new_feed, ::ml_drift::DataType::kInt8);
    ABSL_RETURN_IF_ERROR(
        builder.UpdateOutputTensor(key_new_cast, key_new_int8.id));
    k_new = key_new_int8;
    TensorHandle value_new_int8 =
        builder.AddTensor(1, H, D, T, kBuffer, ::ml_drift::DataType::kInt8);
    TensorHandle value_new_cast =
        builder.Cast(v_new_feed, ::ml_drift::DataType::kInt8);
    ABSL_RETURN_IF_ERROR(
        builder.UpdateOutputTensor(value_new_cast, value_new_int8.id));
    v_new = value_new_int8;
  } else if (float16_slices) {
    k_new = builder.Cast(k_new_feed, ::ml_drift::DataType::kFloat16);
    v_new = builder.Cast(v_new_feed, ::ml_drift::DataType::kFloat16);
  }

  TensorHandle mask_feed =
      builder.AddTensor(::ml_drift::BHWC(1, 1, T, L), data_type);
  TensorHandle mask =
      mask_mode == MaskMode::kBool
          ? builder.Cast(mask_feed, ::ml_drift::DataType::kBool)
          : (float16_output
                 ? builder.Cast(mask_feed, ::ml_drift::DataType::kFloat16)
                 : mask_feed);

  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      param_cpu;
  param_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 7);
  param_cpu.data = {c.start, c.end, 0, 0, 0, 0, 0};
  ::ml_drift::TensorDescriptor param_desc(
      ::ml_drift::DataType::kInt32, kBuffer, ::ml_drift::Layout::kBHWC);
  param_desc.UploadData(param_cpu);
  TensorHandle param = builder.AddConstantTensor(std::move(param_desc));

  TensorHandle out = builder.AddTensor(
      ::ml_drift::BHWC(1, H, R, D),
      float16_output ? ::ml_drift::DataType::kFloat16 : data_type);
  std::vector<uint32_t> output_ids = {out.id};
  std::vector<std::pair<uint32_t, uint32_t>> model_outputs = {{out.id, 0}};
  uint32_t k_cache_out_id = 0;
  uint32_t v_cache_out_id = 0;
  if (c.update_cache) {
    TensorHandle k_cache_out = builder.AddTensor(
        1, H, W, packed_d, kBuffer, ::ml_drift::DataType::kUint8);
    TensorHandle v_cache_out = builder.AddTensor(
        1, H, packed_d, W, kBuffer, ::ml_drift::DataType::kUint8);
    k_cache_out_id = k_cache_out.id;
    v_cache_out_id = v_cache_out.id;
    output_ids.push_back(k_cache_out.id);
    output_ids.push_back(v_cache_out.id);
  }

  FusedSdpaCacheUpdateAttributes attr;
  attr.softcap = c.softcap;
  attr.update_cache = c.update_cache;
  attr.kv_quant_bits = kv_quant_bits;
  attr.scale_k = kScaleK;
  attr.scale_v = kScaleV;
  if (int8_query) attr.scale_q = scale_q;
  if (int8_slices) {
    attr.scale_key_new = kScaleKn;
    attr.scale_value_new = kScaleVn;
  }
  ABSL_RETURN_IF_ERROR(BuildSdpaCacheUpdateGpuGraph(
      {query.id, k_cache.id, v_cache.id, k_new.id, v_new.id, mask.id, param.id},
      output_ids, attr, &builder));

  const int total_cache_bytes = H * W * packed_d;
  if (c.update_cache) {
    // Use a standalone linear-read GPUOperation (which `MergeNodes` never links
    // into `WriteLinear`) that also takes `k_cache`/`v_cache` as `keep_alive`
    // inputs so the memory planner keeps their buffers live until the end of
    // the graph when we redirect `k_cache_out -> k_cache` in-place below.
    ABSL_ASSIGN_OR_RETURN(TensorHandle key_cache_out_handle,
                          builder.GetTensor(k_cache_out_id));
    ABSL_ASSIGN_OR_RETURN(TensorHandle value_cache_out_handle,
                          builder.GetTensor(v_cache_out_id));
    TensorHandle key_cache_float =
        builder.AddTensor(1, 1, 1, total_cache_bytes, kBuffer, data_type);
    TensorHandle value_cache_float =
        builder.AddTensor(1, 1, 1, total_cache_bytes, kBuffer, data_type);
    const int total_vec4s = total_cache_bytes / 4;
    builder.AddGpuOperation(
        {key_cache_out_handle, k_cache}, {key_cache_float},
        CreateReadLinearBytesToFloatOp(
            key_cache_out_handle.tensor_desc, k_cache.tensor_desc,
            key_cache_float.tensor_desc, total_vec4s),
        "test_read_linear_k_cache");
    builder.AddGpuOperation(
        {value_cache_out_handle, v_cache}, {value_cache_float},
        CreateReadLinearBytesToFloatOp(
            value_cache_out_handle.tensor_desc, v_cache.tensor_desc,
            value_cache_float.tensor_desc, total_vec4s),
        "test_read_linear_v_cache");
    model_outputs.push_back({key_cache_float.id, 1});
    model_outputs.push_back({value_cache_float.id, 2});
  }

  ::ml_drift::GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(
      std::vector<std::pair<uint32_t, uint32_t>>{{query_feed.id, 0},
                                                 {init_src_k.id, 1},
                                                 {init_src_v.id, 2},
                                                 {k_new_feed.id, 3},
                                                 {v_new_feed.id, 4},
                                                 {mask_feed.id, 5}},
      model_outputs, &gpu_model));

  if (c.update_cache) {
    // Redirect `fused_sdpa_ring_cache_write` outputs and downstream casts to
    // the initial cache buffers to emulate LiteRT-LM in-place aliasing.
    for (auto& node : gpu_model.nodes) {
      for (uint32_t& id : node.inputs) {
        if (id == k_cache_out_id) id = k_cache.id;
        if (id == v_cache_out_id) id = v_cache.id;
      }
      for (uint32_t& id : node.outputs) {
        if (id == k_cache_out_id) id = k_cache.id;
        if (id == v_cache_out_id) id = v_cache.id;
      }
    }
  }

  const int q_lo = (kv_quant_bits == 4) ? -8 : -128;
  const int q_hi = (kv_quant_bits == 4) ? 7 : 127;
  const int q_span = q_hi - q_lo + 1;

  Inputs in_ref;
  std::vector<float> q_feed_data(H * R * D);
  in_ref.q.resize(H * R * D);
  for (int i = 0; i < H * R * D; ++i) {
    const int qi = ((i * 7 + 3) % 41) - 20;
    if (int8_query) {
      q_feed_data[i] = static_cast<float>(qi);
      in_ref.q[i] = static_cast<float>(qi) * scale_q;
    } else {
      q_feed_data[i] = static_cast<float>(qi) * scale_q;
      in_ref.q[i] = q_feed_data[i];
    }
  }

  std::vector<float> init_k_src(H * W * D);
  std::vector<float> init_v_src(H * W * D);
  in_ref.k_cache.resize(H * W * D);
  in_ref.v_cache.resize(H * D * W);
  for (int h = 0; h < H; ++h) {
    for (int s = 0; s < W; ++s) {
      for (int d = 0; d < D; ++d) {
        const int ki = ((h * 5 + s * 3 + d) % (q_span - 1)) + q_lo;
        const int vi = ((h * 3 + s * 7 + d * 2) % (q_span - 1)) + q_lo;
        const float kv = static_cast<float>(ki) * kScaleK;
        const float vv = static_cast<float>(vi) * kScaleV;
        init_k_src[(h * W + s) * D + d] = kv;
        init_v_src[(h * W + s) * D + d] = vv;
        in_ref.k_cache[(h * W + s) * D + d] = kv;
        in_ref.v_cache[(h * D + d) * W + s] = vv;
      }
    }
  }

  std::vector<float> kn_feed_data(H * T * D);
  std::vector<float> vn_feed_data(H * D * T);
  in_ref.k_new.resize(H * T * D);
  in_ref.v_new.resize(H * D * T);
  for (int h = 0; h < H; ++h) {
    for (int x = 0; x < T; ++x) {
      for (int d = 0; d < D; ++d) {
        const int ki = ((h * 11 + x * 5 + d * 3 + 1) % (q_span - 1)) + q_lo;
        const int vi = ((h * 7 + x * 13 + d * 5 + 2) % (q_span - 1)) + q_lo;
        const float kv = static_cast<float>(ki) * kScaleK;
        const float vv = static_cast<float>(vi) * kScaleV;
        const int k_idx = (h * T + x) * D + d;
        const int v_idx = (h * D + d) * T + x;
        kn_feed_data[k_idx] = int8_slices ? std::round(kv / kScaleKn) : kv;
        vn_feed_data[v_idx] = int8_slices ? std::round(vv / kScaleVn) : vv;
        in_ref.k_new[k_idx] = kv;
        in_ref.v_new[v_idx] = vv;
      }
    }
  }

  in_ref.mask.resize(T * L);
  const int filled_slots = FilledSlots(c);
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < L; ++j) {
      const bool attend =
          !(c.fully_mask_last_row && t == T - 1) &&
          (j < W ? j < filled_slots && (j * 5 + t * 3) % 4 != 0 : (j - W) <= t);
      if (mask_mode == MaskMode::kBool) {
        in_ref.mask[t * L + j] = attend ? 1.0f : 0.0f;
      } else {
        in_ref.mask[t * L + j] =
            attend ? -0.25f * static_cast<float>((j + t) % 3) : -10000.0f;
      }
    }
  }

  std::vector<float> k_cache_post = in_ref.k_cache;
  std::vector<float> v_cache_post = in_ref.v_cache;
  ApplyRingWrite(c, in_ref, k_cache_post, v_cache_post);
  const std::vector<float> expected_out =
      ReferenceAttention(c, mask_mode, in_ref, in_ref.k_cache, in_ref.v_cache);

  auto make_tensor = [](const ::ml_drift::BHWC& shape,
                        std::vector<float> data) {
    ::ml_drift::TensorFloat32 t;
    t.shape = shape;
    t.data = std::move(data);
    return t;
  };
  std::vector<::ml_drift::TensorFloat32> src_cpu = {
      make_tensor(::ml_drift::BHWC(1, H, R, D), q_feed_data),
      make_tensor(::ml_drift::BHWC(1, H, W, D), init_k_src),
      make_tensor(::ml_drift::BHWC(1, H, W, D), init_v_src),
      make_tensor(::ml_drift::BHWC(1, H, T, D), kn_feed_data),
      make_tensor(::ml_drift::BHWC(1, H, D, T), vn_feed_data),
      make_tensor(::ml_drift::BHWC(1, 1, T, L), in_ref.mask)};

  ::ml_drift::TensorFloat32 out_cpu =
      make_tensor(::ml_drift::BHWC(1, H, R, D), {});
  ::ml_drift::TensorFloat32 k_cache_cpu =
      make_tensor(::ml_drift::BHWC(1, 1, 1, total_cache_bytes), {});
  ::ml_drift::TensorFloat32 v_cache_cpu =
      make_tensor(::ml_drift::BHWC(1, 1, 1, total_cache_bytes), {});
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_cpu};
  if (c.update_cache) {
    dst_cpu.push_back(&k_cache_cpu);
    dst_cpu.push_back(&v_cache_cpu);
  }
  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model));

  const bool f16 =
      data_type == ::ml_drift::DataType::kFloat16 || float16_output ||
      float16_slices;
  const float out_tolerance = f16 ? (D > 64 ? 2e-2f : 1e-2f) : 2e-3f;
  EXPECT_THAT(out_cpu.data,
              ::testing::Pointwise(::testing::FloatNear(out_tolerance),
                                   expected_out));

  if (c.update_cache) {
    // Compare directly against the linear `kOSpatialIOGroupO4I4` and
    // `kOSpatialIOGroupI4O4` `uchar4` words written by `AddValuesToCache` and
    // `CreateRingCacheWriteOp`.
    const int head_slices = packed_d / 4;
    const int cache_slices = W / 4;
    std::vector<float> expected_k_linear(total_cache_bytes, 0.0f);
    std::vector<float> expected_v_linear(total_cache_bytes, 0.0f);
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < W; ++s) {
        const int s_slice = s / 4;
        const int s_lane = s % 4;
        for (int hs = 0; hs < head_slices; ++hs) {
          const int k_vec4 =
              ((h * head_slices + hs) * cache_slices + s_slice) * 4 + s_lane;
          const int v_vec4 =
              ((h * cache_slices + s_slice) * head_slices + hs) * 4 + s_lane;
          for (int lane = 0; lane < 4; ++lane) {
            if (kv_quant_bits == 8) {
              const int d = hs * 4 + lane;
              expected_k_linear[k_vec4 * 4 + lane] =
                  std::round(k_cache_post[(h * W + s) * D + d] / kScaleK) +
                  128.0f;
              expected_v_linear[v_vec4 * 4 + lane] =
                  std::round(v_cache_post[(h * D + d) * W + s] / kScaleV) +
                  128.0f;
            } else {
              const int c0 = hs * 8 + lane;
              const int c1 = hs * 8 + 4 + lane;
              const float k0 = k_cache_post[(h * W + s) * D + c0] / kScaleK;
              const float k1 = k_cache_post[(h * W + s) * D + c1] / kScaleK;
              const int k_lo = static_cast<int>(std::round(k0)) + 8;
              const int k_hi = static_cast<int>(std::round(k1)) + 8;
              expected_k_linear[k_vec4 * 4 + lane] =
                  static_cast<float>(k_lo + k_hi * 16);

              const float v0 = v_cache_post[(h * D + c0) * W + s] / kScaleV;
              const float v1 = v_cache_post[(h * D + c1) * W + s] / kScaleV;
              const int v_lo = static_cast<int>(std::round(v0)) + 8;
              const int v_hi = static_cast<int>(std::round(v1)) + 8;
              expected_v_linear[v_vec4 * 4 + lane] =
                  static_cast<float>(v_lo + v_hi * 16);
            }
          }
        }
      }
    }
    EXPECT_THAT(k_cache_cpu.data,
                ::testing::Pointwise(::testing::FloatNear(1e-3f),
                                     expected_k_linear));
    EXPECT_THAT(v_cache_cpu.data,
                ::testing::Pointwise(::testing::FloatNear(1e-3f),
                                     expected_v_linear));
  }
  return absl::OkStatus();
}

TEST_P(FusedSdpaCacheUpdateKernelTest, Gemma5SlidingInt8KvFloatQueryAndSlices) {
  EXPECT_OK(RunQuantizedFusedSdpaCacheUpdateTest(
      *exec_env, precision(), storage(), mask_mode(),
      {.heads = 2,
       .groups = 4,
       .new_len = 16,
       .cache_size = 64,
       .head_dim = 256,
       .start = 56,
       .end = 72},
      /*kv_quant_bits=*/8, /*int8_query=*/false, /*int8_slices=*/false));
}

TEST_P(FusedSdpaCacheUpdateKernelTest,
       Int8QueryAndInt8SlicesWithInt8KvWraparound) {
  for (int new_len : {1, 8}) {
    SCOPED_TRACE(absl::StrCat("new_len=", new_len));
    EXPECT_OK(RunQuantizedFusedSdpaCacheUpdateTest(
        *exec_env, precision(), storage(), mask_mode(),
        {.heads = 2,
         .groups = 2,
         .new_len = new_len,
         .cache_size = 16,
         .head_dim = 64,
         .start = 14,
         .end = 14 + new_len},
        /*kv_quant_bits=*/8, /*int8_query=*/true, /*int8_slices=*/true));
  }
}

TEST_P(FusedSdpaCacheUpdateKernelTest,
       Int8QueryAndInt8SlicesWithPackedInt4KvWraparound) {
  for (const TestCase& tc : {
           TestCase{.heads = 2,
                    .groups = 4,
                    .new_len = 8,
                    .cache_size = 32,
                    .head_dim = 256,
                    .start = 28,
                    .end = 36,
                    .update_cache = true},
           TestCase{.heads = 2,
                    .groups = 2,
                    .new_len = 4,
                    .cache_size = 32,
                    .head_dim = 64,
                    .start = 0,
                    .end = 4,
                    .update_cache = true},
           TestCase{.heads = 2,
                    .groups = 2,
                    .new_len = 1,
                    .cache_size = 16,
                    .head_dim = 64,
                    .start = 10,
                    .end = 11,
                    .update_cache = false},
       }) {
    SCOPED_TRACE(absl::StrCat("start=", tc.start));
    EXPECT_OK(RunQuantizedFusedSdpaCacheUpdateTest(
        *exec_env, precision(), storage(), mask_mode(), tc,
        /*kv_quant_bits=*/4, /*int8_query=*/true, /*int8_slices=*/true));
  }
}

TEST_P(FusedSdpaCacheUpdateKernelTest,
       Float32QueryFloat16OutputQuantizedKvWraparound) {
  if (precision() != ::ml_drift::CalculationsPrecision::kF32) {
    GTEST_SKIP() << "Output already matches query precision.";
  }
  if (!exec_env->IsStorageSupported(storage(),
                                    ::ml_drift::DataType::kFloat16)) {
    GTEST_SKIP() << "FLOAT16 tensors are not supported.";
  }
  for (int kv_bits : {8, 4}) {
    for (bool float16_slices : {false, true}) {
      SCOPED_TRACE(
          absl::StrCat("kv_bits=", kv_bits, ",f16_slices=", float16_slices));
      EXPECT_OK(RunQuantizedFusedSdpaCacheUpdateTest(
          *exec_env, precision(), storage(), mask_mode(),
          {.heads = 2,
           .groups = 4,
           .new_len = 8,
           .cache_size = 32,
           .head_dim = 256,
           .start = 28,
           .end = 36,
           .update_cache = true},
          kv_bits, /*int8_query=*/false, /*int8_slices=*/false,
          /*float16_output=*/true, float16_slices));
    }
  }
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
