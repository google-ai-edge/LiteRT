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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

// Rearranges logical K data [BK, S, H] into the CPU memory order required by
// TensorDescriptor::UploadData to produce the kOSpatialIOGroupO4I4 GPU layout.
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

  if (H % 4 != 0) {
    return absl::InvalidArgumentError(
        "Head dimension H must be a multiple of 4.");
  }
  if (S % 4 != 0) {
    return absl::InvalidArgumentError(
        "Sequence length S must be a multiple of 4.");
  }

  rearranged_data.assign(data.size(), 0.0f);

  for (int bk = 0; bk < BK; ++bk) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        int orig_idx = (bk * S + s) * H + h;
        // Invert UploadData's DSHWBC4 packing for K layout:
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

  if (H % 4 != 0) {
    return absl::InvalidArgumentError(
        "Head dimension H must be a multiple of 4.");
  }
  if (S % 4 != 0) {
    return absl::InvalidArgumentError(
        "Sequence length S must be a multiple of 4.");
  }

  rearranged_data.assign(data.size(), 0.0f);

  for (int bk = 0; bk < BK; ++bk) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int orig_idx = (bk * H + h) * S + s;
        // Invert UploadData's DSHWBC4 packing for V layout.
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

enum class MaskMode { kBool, kFloatAdditive, kNone };

inline std::string ToString(MaskMode mask_mode) {
  switch (mask_mode) {
    case MaskMode::kBool:
      return "BoolMask";
    case MaskMode::kFloatAdditive:
      return "FloatAdditiveMask";
    case MaskMode::kNone:
      return "NoMask";
  }
}

// Computes reference SDPA output on CPU given raw Q, K, V, and mask data.
// `KV` is the number of key/value heads; when it is smaller than the number of
// query heads `BK`, query head `bk` reads KV head `bk / (BK / KV)`, matching
// grouped-query attention. `q_start` is the absolute position of the first
// query token inside the KV cache, which is non-zero when a prompt is
// prefilled in several chunks.
// The fused prefill kernel derives the causal bound from the token positions
// and therefore applies it even when no mask tensor is supplied, whereas the
// decomposed fallback graph attends to every key in that case.
// `implicit_causal` selects which of the two the reference should model;
// it only matters for `MaskMode::kNone`, since the other modes carry causality
// in the mask data.
// Keys at or past `active_tokens` (all `S` keys when it is 0) are not attended,
// matching the active token count the kernels read from the param tensor.
std::vector<float> ComputeSdpaReferenceOutput(
    const std::vector<float>& q_data, const std::vector<float>& k_data,
    const std::vector<float>& v_data, const std::vector<float>& mask_data,
    int BK, int T, int S, int H, MaskMode mask_mode = MaskMode::kBool,
    int KV = 0, int q_start = 0, bool implicit_causal = true,
    int active_tokens = 0) {
  if (KV <= 0) KV = BK;
  if (active_tokens <= 0) active_tokens = S;
  const int gqa_ratio = BK / KV;
  std::vector<float> out_data(BK * T * H, 0.0f);
  for (int bk = 0; bk < BK; ++bk) {
    const int kvh = bk / gqa_ratio;
    for (int t = 0; t < T; ++t) {
      // Absolute position of this query token inside the KV cache.
      const int pos = t + q_start;
      std::vector<float> scores(S, -10000.0f);
      for (int s = 0; s < active_tokens; ++s) {
        float score = 0.0f;
        for (int h = 0; h < H; ++h) {
          int q_idx = (bk * T + t) * H + h;
          int k_idx = (kvh * S + s) * H + h;
          score += q_data[q_idx] * k_data[k_idx];
        }

        if (mask_mode == MaskMode::kNone) {
          scores[s] =
              (implicit_causal && T > 1 && s > pos) ? -10000.0f : score;
        } else if (mask_mode == MaskMode::kBool) {
          if (mask_data[t * S + s] != 0.0f && (T == 1 || s <= pos)) {
            scores[s] = score;
          }
        } else if (mask_mode == MaskMode::kFloatAdditive) {
          if (T > 1 && s > pos) {
            scores[s] = -10000.0f;
          } else {
            scores[s] = score + mask_data[t * S + s];
          }
        }
      }

      float max_score = -1e9f;
      for (int s = 0; s < S; ++s) {
        if (scores[s] > max_score) {
          max_score = scores[s];
        }
      }

      float sum_exp = 0.0f;
      std::vector<float> probs(S, 0.0f);
      for (int s = 0; s < S; ++s) {
        probs[s] = std::exp(scores[s] - max_score);
        sum_exp += probs[s];
      }
      for (int s = 0; s < S; ++s) {
        probs[s] /= sum_exp;
      }

      for (int h = 0; h < H; ++h) {
        float val = 0.0f;
        for (int s = 0; s < S; ++s) {
          int v_idx = (kvh * H + h) * S + s;
          val += probs[s] * v_data[v_idx];
        }
        int out_idx = (bk * T + t) * H + h;
        out_data[out_idx] = val;
      }
    }
  }
  return out_data;
}

class SdpaTransposedKernelExecuteTest
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
    if (!exec_env->IsStorageSupported(storage(), data_type)) {
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
};

bool IsAppleMetal(const ::ml_drift::GpuInfo& gpu_info) {
  return gpu_info.IsApple() && gpu_info.IsApiMetal();
}

// Matches an (actual, expected) pair whose difference is at most
// `abs_tolerance + rel_tolerance * |expected|`.
MATCHER_P2(FloatNearWithRelativeTolerance, abs_tolerance, rel_tolerance,
           absl::StrCat("is within ", abs_tolerance, " + ", rel_tolerance,
                        " * |expected| of expected")) {
  const float actual = std::get<0>(arg);
  const float expected = std::get<1>(arg);
  const float diff = std::abs(actual - expected);
  *result_listener << "which is " << diff << " from " << expected;
  return diff <= abs_tolerance + rel_tolerance * std::abs(expected);
}

// `KV` is the number of key/value heads (defaults to `BK`, i.e. plain MHA) and
// `q_start` is the absolute position of the first query token inside the KV
// cache (non-zero when a prompt is prefilled in several chunks).
// `from_cache_update` selects the packed 4D K/V layout produced by
// `odml.cache_update`; when false the test feeds plain tensors, which routes
// the graph through the decomposed BMM fallback.
// `flatten_output` writes a single-token result as [1, 1, 1, BK * H] instead
// of [1, BK, 1, H], the layout `FuseSdpaTransposedReshape` produces for decode.
// `active_tokens` is the number of filled cache entries written to the param
// tensor (all `S` entries when it is 0).
// For a single-token decode, the mask also hides every key at or past
// `decode_mask_keys` (`active_tokens` when it is 0), as the runtime mask hides
// the unfilled cache entries.
absl::Status RunSdpaTransposedTest(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision,
    ::ml_drift::TensorStorageType storage, int BK = 2, int T = 2, int S = 4,
    int H = 8, MaskMode mask_mode = MaskMode::kBool, int KV = 0,
    int q_start = 0, bool from_cache_update = true, bool is_causal = false,
    bool flatten_output = false, int active_tokens = 0,
    int decode_mask_keys = 0) {
  if (KV <= 0) KV = BK;
  if (active_tokens <= 0) active_tokens = S;
  if (decode_mask_keys <= 0) decode_mask_keys = active_tokens;
  if (KV > BK || BK % KV != 0) {
    return absl::InvalidArgumentError(
        "The number of query heads must be a multiple of the number of KV "
        "heads.");
  }
  if (flatten_output && T != 1) {
    return absl::InvalidArgumentError(
        "Only single-token outputs can be flattened.");
  }
  ::ml_drift::GpuModelBuilder builder(env.GetGpuInfo(), {}, precision, storage);

  ::ml_drift::DataType datatype;
  switch (precision) {
    case ::ml_drift::CalculationsPrecision::kF16:
      datatype = ::ml_drift::DataType::kFloat16;
      break;
    case ::ml_drift::CalculationsPrecision::kF32:
      datatype = ::ml_drift::DataType::kFloat32;
      break;
    default:
      return absl::InvalidArgumentError("Unsupported precision.");
  }

  ::ml_drift::TensorStorageType kv_storage_type =
      from_cache_update ? ::ml_drift::TensorStorageType::kBuffer : storage;

  auto q = builder.AddTensor(::ml_drift::BHWC(1, BK, T, H), datatype);
  auto q_shape = q.tensor_desc.GetBHWCShape();
  auto k = builder.AddTensor(1, KV, S, H, kv_storage_type, datatype);
  auto k_activation_shape = k.tensor_desc.GetBHWCShape();
  auto v = builder.AddTensor(1, KV, H, S, kv_storage_type, datatype);
  auto v_activation_shape = v.tensor_desc.GetBHWCShape();

  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> mask_handle;
  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> mask_feed_handle;
  ::ml_drift::BHWC mask_shape(1, 1, T, S);

  if (mask_mode == MaskMode::kBool) {
    auto mask_float = builder.AddTensor(mask_shape, datatype);
    mask_feed_handle = mask_float;
    mask_handle = builder.Cast(mask_float, ::ml_drift::DataType::kBool);
  } else if (mask_mode == MaskMode::kFloatAdditive) {
    auto mask_float = builder.AddTensor(mask_shape, datatype);
    mask_feed_handle = mask_float;
    mask_handle = mask_float;
  }

  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      param_tensor_cpu;
  param_tensor_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 7);
  // {cache update start index, cache update end index, active tokens}.
  param_tensor_cpu.data = {q_start, S, active_tokens, 0, 0, 0, 0};

  ::ml_drift::TensorDescriptor param_desc(
      ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kBHWC);
  param_desc.UploadData(param_tensor_cpu);
  auto param_tensor = builder.AddConstantTensor(std::move(param_desc));

  const ::ml_drift::BHWC out_shape = flatten_output
                                         ? ::ml_drift::BHWC(1, 1, 1, BK * H)
                                         : ::ml_drift::BHWC(1, BK, T, H);
  auto out_tensor = builder.AddTensor(out_shape, datatype);

  SdpaTransposedAttributes attr;
  attr.runtime_check.src_end_ch_index = 2;
  attr.from_cache_update = from_cache_update;
  attr.is_prefill = (T > 1);
  attr.is_causal = is_causal;

  auto k_weights_shape = ::ml_drift::OHWI(
      k_activation_shape.w, k_activation_shape.h, 1, k_activation_shape.c);
  attr.bmm1_weights.weights_shape = k_weights_shape;
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::kFloat32, attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;

  auto v_weights_shape = ::ml_drift::OHWI(
      v_activation_shape.w, v_activation_shape.h, 1, v_activation_shape.c);
  attr.bmm2_weights.weights_shape = v_weights_shape;
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::kFloat32, attr.bmm2_weights.weights_shape);

  std::vector<uint32_t> graph_input_ids = {q.id, k.id, v.id};
  if (mask_handle.has_value()) {
    graph_input_ids.push_back(mask_handle->id);
  }
  graph_input_ids.push_back(param_tensor.id);

  ABSL_RETURN_IF_ERROR(BuildSdpaTransposedGpuGraph(
      graph_input_ids, out_tensor.id, attr, &builder));

  std::vector<std::pair<uint32_t, uint32_t>> model_inputs = {
      {q.id, 0}, {k.id, 1}, {v.id, 2}};
  if (mask_feed_handle.has_value()) {
    model_inputs.push_back({mask_feed_handle->id, 3});
  }

  ::ml_drift::GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(
      model_inputs,
      std::vector<std::pair<uint32_t, uint32_t>>{{out_tensor.id, 0}},
      &gpu_model));

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(H));

  std::vector<float> q_data(BK * T * H, 0.0f);
  for (int bk = 0; bk < BK; ++bk) {
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < H; ++h) {
        int idx = (bk * T + t) * H + h;
        q_data[idx] = q_scale * (0.05f * (bk + 1) + 0.02f * (t + 1) +
                                 0.01f * ((h % 8) + 1));
      }
    }
  }

  std::vector<float> k_data(KV * S * H, 0.0f);
  for (int kvh = 0; kvh < KV; ++kvh) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        int idx = (kvh * S + s) * H + h;
        k_data[idx] =
            0.05f * (kvh + 1) + 0.02f * (s + 1) - 0.01f * ((h % 8) + 1);
      }
    }
  }

  std::vector<float> v_data(KV * H * S, 0.0f);
  for (int kvh = 0; kvh < KV; ++kvh) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int idx = (kvh * H + h) * S + s;
        v_data[idx] = ((h % 8) + 1) * 0.01f + (s + 1) * 0.02f + kvh * 0.05f;
      }
    }
  }

  std::vector<float> mask_data(T * S, 0.0f);
  if (mask_mode == MaskMode::kBool) {
    for (int t = 0; t < T; ++t) {
      for (int s = 0; s < S; ++s) {
        const bool unmasked =
            (T == 1) ? (s < decode_mask_keys && (s == 0 || s % 4 != 3))
                     : (s <= t + q_start);
        mask_data[t * S + s] = unmasked ? 1.0f : 0.0f;
      }
    }
  } else if (mask_mode == MaskMode::kFloatAdditive) {
    for (int t = 0; t < T; ++t) {
      for (int s = 0; s < S; ++s) {
        const bool unmasked =
            (T == 1) ? (s < decode_mask_keys && (s == 0 || s % 4 != 3))
                     : (s <= t + q_start);
        mask_data[t * S + s] =
            unmasked ? -0.25f * static_cast<float>((t + s) % 4) : -10000.0f;
      }
    }
  }

  // Without a mask tensor (`MaskMode::kNone`), SDPA attends to all active keys
  // unless `attr.is_causal` is set (e.g., when the parser prunes a BOOL causal
  // mask for Flash SDPA).
  std::vector<float> expected_out_data = ComputeSdpaReferenceOutput(
      q_data, k_data, v_data, mask_data, BK, T, S, H, mask_mode, KV, q_start,
      /*implicit_causal=*/is_causal, active_tokens);

  std::vector<float> rearranged_k_data;
  std::vector<float> rearranged_v_data;
  if (from_cache_update) {
    ABSL_RETURN_IF_ERROR(
        RearrangeK(k_data, rearranged_k_data, k_weights_shape));
    ABSL_RETURN_IF_ERROR(
        RearrangeV(v_data, rearranged_v_data, v_weights_shape));
  }

  ::ml_drift::TensorFloat32 q_tensor;
  q_tensor.shape = q_shape;
  q_tensor.data = q_data;

  ::ml_drift::TensorFloat32 k_tensor;
  k_tensor.shape = k_activation_shape;
  k_tensor.data = from_cache_update ? rearranged_k_data : k_data;

  ::ml_drift::TensorFloat32 v_tensor;
  v_tensor.shape = v_activation_shape;
  v_tensor.data = from_cache_update ? rearranged_v_data : v_data;

  std::vector<::ml_drift::TensorFloat32> src_cpu = {q_tensor, k_tensor,
                                                    v_tensor};

  if (mask_mode != MaskMode::kNone) {
    ::ml_drift::TensorFloat32 mask_tensor;
    mask_tensor.shape = mask_shape;
    mask_tensor.data = mask_data;
    src_cpu.push_back(mask_tensor);
  }

  ::ml_drift::TensorFloat32 out_tensor_cpu;
  out_tensor_cpu.shape = out_shape;
  out_tensor_cpu.data.resize(BK * T * H);
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_tensor_cpu};

  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model));

  // At 512 keys the test data produces logits near 100, so half-precision
  // math errs by up to a few hundredths on results near 10. The fused Apple
  // kernels compute in half precision even when F32 is requested.
  const bool half_precision_math =
      precision == ::ml_drift::CalculationsPrecision::kF16 ||
      IsAppleMetal(env.GetGpuInfo());
  float tolerance = (half_precision_math && S >= 512)
                        ? 6e-2f
                        : ((H > 16) ? 1.5e-2f : 2e-3f);
  // The half-precision error also grows with the magnitude of the result,
  // which reaches about 32 at 1600 keys.
  constexpr float kRelativeTolerance = 4e-3f;
  EXPECT_THAT(out_tensor_cpu.data,
              testing::Pointwise(FloatNearWithRelativeTolerance(
                                     tolerance, kRelativeTolerance),
                                 expected_out_data));

  return absl::OkStatus();
}

TEST_P(SdpaTransposedKernelExecuteTest, BuildAndExecute) {
  auto status =
      RunSdpaTransposedTest(*exec_env, precision(), storage(),
                            /*BK=*/2, /*T=*/2, /*S=*/4, /*H=*/8, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, BuildAndExecuteLargerDimensions) {
  auto status =
      RunSdpaTransposedTest(*exec_env, precision(), storage(),
                            /*BK=*/4, /*T=*/2, /*S=*/16, /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecode) {
  auto status =
      RunSdpaTransposedTest(*exec_env, precision(), storage(),
                            /*BK=*/2, /*T=*/1, /*S=*/4, /*H=*/8, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecodeLargerDimensions) {
  auto status =
      RunSdpaTransposedTest(*exec_env, precision(), storage(),
                            /*BK=*/4, /*T=*/1, /*S=*/16, /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecodeHeadDim128) {
  auto status =
      RunSdpaTransposedTest(*exec_env, precision(), storage(),
                            /*BK=*/2, /*T=*/1, /*S=*/32, /*H=*/128,
                            mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest,
       SingleTokenDecodeHeadDim128StandardTensors) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/2, /*T=*/1, /*S=*/32, /*H=*/128,
      mask_mode(), /*KV=*/0, /*q_start=*/0, /*from_cache_update=*/false);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, PrefillMultiToken) {
  auto status =
      RunSdpaTransposedTest(*exec_env, precision(), storage(),
                            /*BK=*/2, /*T=*/4, /*S=*/8, /*H=*/8, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, StandardTensorsFallback) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/2, /*T=*/2, /*S=*/4, /*H=*/8,
      mask_mode(), /*KV=*/0, /*q_start=*/0, /*from_cache_update=*/false);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Grouped-query attention: 8 query heads share 2 KV heads. Every query head
// must apply the causal mask against its own token index, which regressed when
// the exporter packed query heads into the sequence dimension.
TEST_P(SdpaTransposedKernelExecuteTest, PrefillMultiTokenGroupedQuery) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/4, /*S=*/8, /*H=*/8,
                                      mask_mode(), /*KV=*/2);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest,
       PrefillMultiTokenGroupedQueryHeadDim128) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/8, /*S=*/16, /*H=*/128,
                                      mask_mode(), /*KV=*/2);
  EXPECT_TRUE(status.ok()) << status.message();
}

// The prefill kernel walks 4 query columns per SIMD group and 8 keys per loop
// iteration, with a separate tail loop once the causal bound is no longer a
// multiple of that step. The cases below exercise that tiling at the head dims
// the Qwen3 models use: several query tiles, several key tiles, and edges that
// do not divide evenly.
TEST_P(SdpaTransposedKernelExecuteTest, PrefillHeadDim128MultipleQueryTiles) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/24, /*S=*/32, /*H=*/128,
                                      mask_mode(), /*KV=*/2);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, PrefillHeadDim128MultipleKeyTiles) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/64, /*S=*/96, /*H=*/128,
                                      mask_mode(), /*KV=*/2);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Query and key counts that are not multiples of the 8x32 tiling.
TEST_P(SdpaTransposedKernelExecuteTest, PrefillHeadDim128RaggedTiles) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/13, /*S=*/44, /*H=*/128,
                                      mask_mode(), /*KV=*/2);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, PrefillHeadDim128ChunkedStartOffset) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/16, /*S=*/64, /*H=*/128,
                                      mask_mode(), /*KV=*/2, /*q_start=*/40);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Head dim 64 (two channels per lane instead of four) exercises the same
// tiling with a different per-lane channel split.
TEST_P(SdpaTransposedKernelExecuteTest, PrefillHeadDim64MultipleTiles) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/20, /*S=*/48, /*H=*/64,
                                      mask_mode(), /*KV=*/4);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Second chunk of a chunked prefill: the four query tokens sit at absolute
// positions 4..7 of the KV cache and may attend to everything before them.
TEST_P(SdpaTransposedKernelExecuteTest, PrefillMultiTokenChunkedStartOffset) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/2, /*T=*/4, /*S=*/8, /*H=*/8,
                                      mask_mode(), /*KV=*/2, /*q_start=*/4);
  EXPECT_TRUE(status.ok()) << status.message();
}

// The two tests below mirror the real Qwen3 head geometries at a key count
// large enough to exercise both the safe 8-key main loop and the masked tail,
// with a chunk offset. They differ only in the query head count (and therefore
// the GQA ratio), which is the only attention-shape difference between the
// 0.6B model, which retrieves correctly at long context, and the 4B model,
// which does not.
TEST_P(SdpaTransposedKernelExecuteTest, PrefillQwen3_0_6BHeadGeometry) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/16, /*T=*/32, /*S=*/512, /*H=*/128,
                                      mask_mode(), /*KV=*/8, /*q_start=*/256);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, PrefillQwen3_4BHeadGeometry) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/32, /*T=*/32, /*S=*/512, /*H=*/128,
                                      mask_mode(), /*KV=*/8, /*q_start=*/256);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Grouped-query decode. On the Metal backend, head dim 128 selects the fused
// flash-decode kernel and OpenCL selects the portable flash-decode kernel;
// elsewhere the decomposed graph folds each group of query heads into the
// token axis.
TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecodeGroupedQuery) {
  auto status = RunSdpaTransposedTest(*exec_env, precision(), storage(),
                                      /*BK=*/8, /*T=*/1, /*S=*/32, /*H=*/128,
                                      mask_mode(), /*KV=*/2);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Qwen3 0.6B decode as it reaches the delegate: 16 query heads share 8 KV
// heads, the output is flattened to [1, 1, 1, 16 * 128], and the token being
// decoded is the last of a full 512-entry cache, so every key is visible.
TEST_P(SdpaTransposedKernelExecuteTest,
       SingleTokenDecodeQwen3_0_6BHeadGeometryFlattenedOutput) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/16, /*T=*/1, /*S=*/512,
      /*H=*/128, mask_mode(), /*KV=*/8, /*q_start=*/511,
      /*from_cache_update=*/true, /*is_causal=*/false, /*flatten_output=*/true);
  EXPECT_TRUE(status.ok()) << status.message();
}

// The number of filled cache entries for a decode test. The decomposed graph
// ignores the active token count and relies on the mask to hide the unfilled
// entries, so without a mask the whole cache counts as filled.
int FilledCacheEntries(MaskMode mask_mode, int filled, int cache_size) {
  return mask_mode == MaskMode::kNone ? cache_size : filled;
}

// Qwen3 0.6B decode in a partly filled 1280-entry cache: 1101 entries are
// filled. The portable flash-decode kernel splits the keys of each K/V head
// across 5 work groups, so the last split is shorter than the others and ends
// one key into a block of 4 keys.
TEST_P(SdpaTransposedKernelExecuteTest,
       SingleTokenDecodeQwen3_0_6BHeadGeometryPartlyFilledCache) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/16, /*T=*/1, /*S=*/1280,
      /*H=*/128, mask_mode(), /*KV=*/8, /*q_start=*/1100,
      /*from_cache_update=*/true, /*is_causal=*/false, /*flatten_output=*/true,
      /*active_tokens=*/FilledCacheEntries(mask_mode(), 1101, 1280));
  EXPECT_TRUE(status.ok()) << status.message();
}

// The param tensor marks the whole 1280-entry cache as filled, but the mask
// hides every key after position 700. Of the 5 key splits of the portable
// flash-decode kernel, the third is partly visible and the last two have no
// visible key.
TEST_P(SdpaTransposedKernelExecuteTest,
       SingleTokenDecodeQwen3_0_6BHeadGeometryMaskHidesTrailingSplits) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/16, /*T=*/1, /*S=*/1280,
      /*H=*/128, mask_mode(), /*KV=*/8, /*q_start=*/700,
      /*from_cache_update=*/true, /*is_causal=*/false, /*flatten_output=*/true,
      /*active_tokens=*/0, /*decode_mask_keys=*/701);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Four query heads per KV head: the portable flash-decode kernel splits the
// keys of each K/V head in two, and each work item scores two keys.
TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecodeGroupSizeFour) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/8, /*T=*/1, /*S=*/600,
      /*H=*/64, mask_mode(), /*KV=*/2, /*q_start=*/560,
      /*from_cache_update=*/true, /*is_causal=*/false, /*flatten_output=*/false,
      /*active_tokens=*/FilledCacheEntries(mask_mode(), 561, 600));
  EXPECT_TRUE(status.ok()) << status.message();
}

// Eight query heads per KV head: the work-group Flash-Decode kernel keeps the
// keys of the K/V head in one work group, which walks them in two chunks and
// rescales the running softmax between them.
TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecodeGroupSizeEight) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/8, /*T=*/1, /*S=*/300,
      /*H=*/128, mask_mode(), /*KV=*/1, /*q_start=*/280,
      /*from_cache_update=*/true, /*is_causal=*/false, /*flatten_output=*/true,
      /*active_tokens=*/FilledCacheEntries(mask_mode(), 281, 300));
  EXPECT_TRUE(status.ok()) << status.message();
}

// Sixteen query heads per KV head: exceeds the per-work-group head cap (8), so
// the work-group Flash-Decode kernel partitions the 16 query heads across 2
// work groups of 8 heads each per key split.
TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecodeGroupSizeSixteen) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/16, /*T=*/1, /*S=*/300,
      /*H=*/64, mask_mode(), /*KV=*/1, /*q_start=*/280,
      /*from_cache_update=*/true, /*is_causal=*/false, /*flatten_output=*/true,
      /*active_tokens=*/FilledCacheEntries(mask_mode(), 281, 300));
  EXPECT_TRUE(status.ok()) << status.message();
}

// Eight query heads per KV head for 8 KV heads in a 1600-entry cache with 1501
// filled entries: the work-group Flash-Decode kernel splits the keys of each
// K/V head across 5 work groups, each of which walks its keys in two chunks.
TEST_P(SdpaTransposedKernelExecuteTest,
       SingleTokenDecodeSplitKeysWithSeveralChunks) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/64, /*T=*/1, /*S=*/1600,
      /*H=*/64, mask_mode(), /*KV=*/8, /*q_start=*/1500,
      /*from_cache_update=*/true, /*is_causal=*/false, /*flatten_output=*/false,
      /*active_tokens=*/FilledCacheEntries(mask_mode(), 1501, 1600));
  EXPECT_TRUE(status.ok()) << status.message();
}

// Grouped-query attention on plain K/V tensors, which always takes the
// decomposed BatchedMatMul graph.
TEST_P(SdpaTransposedKernelExecuteTest, StandardTensorsFallbackGroupedQuery) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/8, /*T=*/4, /*S=*/8, /*H=*/8,
      mask_mode(), /*KV=*/2, /*q_start=*/0, /*from_cache_update=*/false);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest,
       SingleTokenDecodeGroupedQueryStandardTensors) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/8, /*T=*/1, /*S=*/32, /*H=*/128,
      mask_mode(), /*KV=*/2, /*q_start=*/0, /*from_cache_update=*/false);
  EXPECT_TRUE(status.ok()) << status.message();
}

// Verifies that when the parser prunes a BOOL causal mask (`MaskMode::kNone`
// with `attr.is_causal = true`), the fused FlashAttention prefill kernel still
// enforces causal masking (`key <= q_start + X`) in registers.
TEST_P(SdpaTransposedKernelExecuteTest,
       PrefillImplicitCausalWhenBoolMaskPruned) {
  if (!IsAppleMetal(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "BOOL causal mask pruning only applies to the fused Apple "
                    "Metal FlashAttention prefill kernel.";
  }
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), /*BK=*/16, /*T=*/32, /*S=*/512,
      /*H=*/128, MaskMode::kNone, /*KV=*/8, /*q_start=*/256,
      /*from_cache_update=*/true, /*is_causal=*/true);
  EXPECT_TRUE(status.ok()) << status.message();
}

INSTANTIATE_TEST_SUITE_P(
    SdpaTransposedKernelExecuteTestSuite, SdpaTransposedKernelExecuteTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::kF32,
                      ::ml_drift::CalculationsPrecision::kF16}),
            ValuesIn({::ml_drift::TensorStorageType::kTexture2D,
                      ::ml_drift::TensorStorageType::kBuffer}),
            ValuesIn({MaskMode::kBool, MaskMode::kFloatAdditive,
                      MaskMode::kNone})),
    [](const TestParamInfo<SdpaTransposedKernelExecuteTest::ParamType>& info) {
      std::string name =
          absl::StrCat(::ml_drift::ToString(std::get<0>(info.param)), "_",
                       ::ml_drift::ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

::ml_drift::GpuInfo MakeGpuInfo(::ml_drift::GpuVendor vendor,
                                ::ml_drift::GpuApi api) {
  ::ml_drift::GpuInfo gpu_info;
  gpu_info.vendor = vendor;
  gpu_info.gpu_api = api;
  return gpu_info;
}

TEST(SupportsFusedSdpaKernelsTest, AppleGpuOnMetal) {
  EXPECT_TRUE(SupportsFusedSdpaKernels(
      MakeGpuInfo(::ml_drift::GpuVendor::kApple, ::ml_drift::GpuApi::kMetal)));
}

TEST(SupportsFusedSdpaKernelsTest, OpenClGpus) {
  EXPECT_TRUE(SupportsFusedSdpaKernels(MakeGpuInfo(
      ::ml_drift::GpuVendor::kQualcomm, ::ml_drift::GpuApi::kOpenCl)));
  EXPECT_TRUE(SupportsFusedSdpaKernels(MakeGpuInfo(
      ::ml_drift::GpuVendor::kNvidia, ::ml_drift::GpuApi::kOpenCl)));
  EXPECT_TRUE(SupportsFusedSdpaKernels(
      MakeGpuInfo(::ml_drift::GpuVendor::kApple, ::ml_drift::GpuApi::kOpenCl)));
}

TEST(SupportsFusedSdpaKernelsTest, UnsupportedApisUseFallback) {
  EXPECT_FALSE(SupportsFusedSdpaKernels(
      MakeGpuInfo(::ml_drift::GpuVendor::kApple, ::ml_drift::GpuApi::kWebGpu)));
  EXPECT_FALSE(SupportsFusedSdpaKernels(
      MakeGpuInfo(::ml_drift::GpuVendor::kAMD, ::ml_drift::GpuApi::kMetal)));
  EXPECT_FALSE(SupportsFusedSdpaKernels(
      MakeGpuInfo(::ml_drift::GpuVendor::kNvidia, ::ml_drift::GpuApi::kWebGpu)));
  EXPECT_FALSE(SupportsFusedSdpaKernels(MakeGpuInfo(
      ::ml_drift::GpuVendor::kQualcomm, ::ml_drift::GpuApi::kVulkan)));
}

}  // namespace
}  // namespace litert::ml_drift
