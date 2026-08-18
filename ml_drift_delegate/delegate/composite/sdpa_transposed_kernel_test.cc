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
std::vector<float> ComputeSdpaReferenceOutput(
    const std::vector<float>& q_data, const std::vector<float>& k_data,
    const std::vector<float>& v_data, const std::vector<float>& mask_data,
    int BK, int T, int S, int H, MaskMode mask_mode = MaskMode::kBool) {
  std::vector<float> out_data(BK * T * H, 0.0f);
  for (int bk = 0; bk < BK; ++bk) {
    for (int t = 0; t < T; ++t) {
      std::vector<float> scores(S, -10000.0f);
      for (int s = 0; s < S; ++s) {
        float score = 0.0f;
        for (int h = 0; h < H; ++h) {
          int q_idx = (bk * T + t) * H + h;
          int k_idx = (bk * S + s) * H + h;
          score += q_data[q_idx] * k_data[k_idx];
        }

        if (mask_mode == MaskMode::kNone) {
          scores[s] = score;
        } else if (mask_mode == MaskMode::kBool) {
          if (mask_data[t * S + s] != 0.0f) {
            scores[s] = score;
          }
        } else if (mask_mode == MaskMode::kFloatAdditive) {
          scores[s] = score + mask_data[t * S + s];
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
          int v_idx = (bk * H + h) * S + s;
          val += probs[s] * v_data[v_idx];
        }
        int out_idx = (bk * T + t) * H + h;
        out_data[out_idx] = val;
      }
    }
  }
  return out_data;
}

TEST(SdpaTransposedKernelTest, SelectSdpaStrategy) {
  ::ml_drift::GpuInfo non_apple_gpu;
  non_apple_gpu.vendor = ::ml_drift::GpuVendor::kQualcomm;

  // Prefill (non-decode): always composite fallback.
  EXPECT_EQ(SelectSdpaStrategy(non_apple_gpu, /*is_decode=*/false,
                               /*allow_single_kernel=*/true,
                               /*request_flash_decoding=*/true),
            SdpaImplementationStrategy::kCompositeMultiKernelFallback);
  EXPECT_EQ(SelectSdpaStrategy(non_apple_gpu, /*is_decode=*/false,
                               /*allow_single_kernel=*/false,
                               /*request_flash_decoding=*/false),
            SdpaImplementationStrategy::kCompositeMultiKernelFallback);

  // Single kernel decode.
  EXPECT_EQ(SelectSdpaStrategy(non_apple_gpu, /*is_decode=*/true,
                               /*allow_single_kernel=*/true,
                               /*request_flash_decoding=*/false),
            SdpaImplementationStrategy::kCompositeMultiKernelFallback);
  EXPECT_EQ(SelectSdpaStrategy(non_apple_gpu, /*is_decode=*/true,
                               /*allow_single_kernel=*/true,
                               /*request_flash_decoding=*/true),
            SdpaImplementationStrategy::kSingleKernelFlashDecode);

  // Two-kernel flash decode.
  EXPECT_EQ(SelectSdpaStrategy(non_apple_gpu, /*is_decode=*/true,
                               /*allow_single_kernel=*/false,
                               /*request_flash_decoding=*/true),
            SdpaImplementationStrategy::kTwoKernelFlashDecode);
}

class SdpaTransposedKernelExecuteTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<
          std::tuple<::ml_drift::CalculationsPrecision,
                     ::ml_drift::TensorStorageType, bool, bool, MaskMode>> {
 public:
  void SetUp() override {
    if (!exec_env) {
      GTEST_SKIP() << "TestExecutionEnvironment not initialized.";
    }
  }

 protected:
  ::ml_drift::CalculationsPrecision precision() const {
    return std::get<0>(GetParam());
  }
  ::ml_drift::TensorStorageType storage() const {
    return std::get<1>(GetParam());
  }
  bool allow_single_kernel_implementation() const {
    return std::get<2>(GetParam());
  }
  bool request_flash_decoding() const { return std::get<3>(GetParam()); }
  MaskMode mask_mode() const { return std::get<4>(GetParam()); }
};

absl::Status RunSdpaTransposedTest(::ml_drift::TestExecutionEnvironment& env,
                                   ::ml_drift::CalculationsPrecision precision,
                                   ::ml_drift::TensorStorageType storage,
                                   bool allow_single_kernel_implementation,
                                   bool request_flash_decoding = false,
                                   int BK = 2, int T = 2, int S = 4, int H = 8,
                                   MaskMode mask_mode = MaskMode::kBool,
                                   std::optional<SdpaImplementationStrategy>
                                       strategy_override = std::nullopt) {
  ::ml_drift::GpuModelBuilder builder(env.GetGpuInfo(), {}, precision, storage);

  ::ml_drift::DataType datatype;
  switch (precision) {
    case ::ml_drift::CalculationsPrecision::F16:
      datatype = ::ml_drift::DataType::FLOAT16;
      break;
    case ::ml_drift::CalculationsPrecision::F32:
      datatype = ::ml_drift::DataType::FLOAT32;
      break;
    default:
      return absl::InvalidArgumentError("Unsupported precision.");
  }

  ::ml_drift::TensorStorageType kv_storage_type =
      ::ml_drift::TensorStorageType::BUFFER;

  auto q = builder.AddTensor(::ml_drift::BHWC(1, BK, T, H), datatype);
  auto q_shape = q.tensor_desc.GetBHWCShape();
  auto k = builder.AddTensor(1, BK, S, H, kv_storage_type, datatype);
  auto k_activation_shape = k.tensor_desc.GetBHWCShape();
  auto v = builder.AddTensor(1, BK, H, S, kv_storage_type, datatype);
  auto v_activation_shape = v.tensor_desc.GetBHWCShape();

  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> mask_handle;
  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> mask_feed_handle;
  ::ml_drift::BHWC mask_shape(1, 1, T, S);

  if (mask_mode == MaskMode::kBool) {
    auto mask_float = builder.AddTensor(mask_shape, datatype);
    mask_feed_handle = mask_float;
    mask_handle = builder.Cast(mask_float, ::ml_drift::DataType::BOOL);
  } else if (mask_mode == MaskMode::kFloatAdditive) {
    auto mask_float = builder.AddTensor(mask_shape, datatype);
    mask_feed_handle = mask_float;
    mask_handle = mask_float;
  }

  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::BHWC>,
                     ::ml_drift::DataType::INT32>
      param_tensor_cpu;
  param_tensor_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 7);
  param_tensor_cpu.data = {0, 0, S, 0, 0, 0, 0};

  ::ml_drift::TensorDescriptor param_desc(::ml_drift::DataType::INT32,
                                          ::ml_drift::TensorStorageType::BUFFER,
                                          ::ml_drift::Layout::BHWC);
  param_desc.UploadData(param_tensor_cpu);
  auto param_tensor = builder.AddConstantTensor(std::move(param_desc));

  auto out_tensor = builder.AddTensor(::ml_drift::BHWC(1, BK, T, H), datatype);

  SdpaTransposedAttributes attr;
  attr.runtime_check.src_end_ch_index = 2;

  auto k_weights_shape = ::ml_drift::OHWI(
      k_activation_shape.w, k_activation_shape.h, 1, k_activation_shape.c);
  attr.bmm1_weights.weights_shape = k_weights_shape;
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::FLOAT32, attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;

  auto v_weights_shape = ::ml_drift::OHWI(
      v_activation_shape.w, v_activation_shape.h, 1, v_activation_shape.c);
  attr.bmm2_weights.weights_shape = v_weights_shape;
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::FLOAT32, attr.bmm2_weights.weights_shape);
  attr.bmm2_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint32_t> graph_input_ids = {q.id, k.id, v.id};
  if (mask_handle.has_value()) {
    graph_input_ids.push_back(mask_handle->id);
  }
  graph_input_ids.push_back(param_tensor.id);

  ABSL_RETURN_IF_ERROR(
      BuildSdpaTransposedGpuGraph(graph_input_ids, out_tensor.id, attr,
                                  &builder, allow_single_kernel_implementation,
                                  request_flash_decoding, strategy_override));

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

  std::vector<float> k_data(BK * S * H, 0.0f);
  for (int bk = 0; bk < BK; ++bk) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        int idx = (bk * S + s) * H + h;
        k_data[idx] =
            0.05f * (bk + 1) + 0.02f * (s + 1) - 0.01f * ((h % 8) + 1);
      }
    }
  }

  std::vector<float> v_data(BK * H * S, 0.0f);
  for (int bk = 0; bk < BK; ++bk) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int idx = (bk * H + h) * S + s;
        v_data[idx] = ((h % 8) + 1) * 0.01f + (s + 1) * 0.02f + bk * 0.05f;
      }
    }
  }

  std::vector<float> mask_data(T * S, 0.0f);
  if (mask_mode == MaskMode::kBool) {
    for (int t = 0; t < T; ++t) {
      for (int s = 0; s < S; ++s) {
        mask_data[t * S + s] = (s <= (S / 2 + t)) ? 1.0f : 0.0f;
      }
    }
  } else if (mask_mode == MaskMode::kFloatAdditive) {
    for (int t = 0; t < T; ++t) {
      for (int s = 0; s < S; ++s) {
        mask_data[t * S + s] = (s <= (S / 2 + t)) ? 0.0f : -10000.0f;
      }
    }
  }

  std::vector<float> expected_out_data = ComputeSdpaReferenceOutput(
      q_data, k_data, v_data, mask_data, BK, T, S, H, mask_mode);

  std::vector<float> rearranged_k_data;
  ABSL_RETURN_IF_ERROR(RearrangeK(k_data, rearranged_k_data, k_weights_shape));

  std::vector<float> rearranged_v_data;
  ABSL_RETURN_IF_ERROR(RearrangeV(v_data, rearranged_v_data, v_weights_shape));

  ::ml_drift::TensorFloat32 q_tensor;
  q_tensor.shape = q_shape;
  q_tensor.data = q_data;

  ::ml_drift::TensorFloat32 k_tensor;
  k_tensor.shape = k_activation_shape;
  k_tensor.data = rearranged_k_data;

  ::ml_drift::TensorFloat32 v_tensor;
  v_tensor.shape = v_activation_shape;
  v_tensor.data = rearranged_v_data;

  std::vector<::ml_drift::TensorFloat32> src_cpu = {q_tensor, k_tensor,
                                                    v_tensor};

  if (mask_mode != MaskMode::kNone) {
    ::ml_drift::TensorFloat32 mask_tensor;
    mask_tensor.shape = mask_shape;
    mask_tensor.data = mask_data;
    src_cpu.push_back(mask_tensor);
  }

  ::ml_drift::TensorFloat32 out_tensor_cpu;
  out_tensor_cpu.shape = ::ml_drift::BHWC(1, BK, T, H);
  out_tensor_cpu.data.resize(BK * T * H);
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_tensor_cpu};

  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model));

  float tolerance =
      (precision == ::ml_drift::CalculationsPrecision::F16) ? 1e-3f : 1e-5f;
  EXPECT_THAT(
      out_tensor_cpu.data,
      testing::Pointwise(testing::FloatNear(tolerance), expected_out_data));

  return absl::OkStatus();
}

TEST_P(SdpaTransposedKernelExecuteTest, BuildAndExecute) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), allow_single_kernel_implementation(),
      request_flash_decoding(),
      /*BK=*/2, /*T=*/2, /*S=*/4, /*H=*/8, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, BuildAndExecuteLargerDimensions) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), allow_single_kernel_implementation(),
      request_flash_decoding(),
      /*BK=*/4, /*T=*/2, /*S=*/16, /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecode) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), allow_single_kernel_implementation(),
      request_flash_decoding(),
      /*BK=*/2, /*T=*/1, /*S=*/4, /*H=*/8, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, SingleTokenDecodeLargerDimensions) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), allow_single_kernel_implementation(),
      request_flash_decoding(),
      /*BK=*/4, /*T=*/1, /*S=*/16, /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaTransposedKernelExecuteTest, PrefillMultiToken) {
  auto status = RunSdpaTransposedTest(
      *exec_env, precision(), storage(), allow_single_kernel_implementation(),
      request_flash_decoding(),
      /*BK=*/2, /*T=*/4, /*S=*/8, /*H=*/8, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

INSTANTIATE_TEST_SUITE_P(
    SdpaTransposedKernelExecuteTestSuite, SdpaTransposedKernelExecuteTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::F32,
                      ::ml_drift::CalculationsPrecision::F16}),
            ValuesIn({::ml_drift::TensorStorageType::TEXTURE_2D,
                      ::ml_drift::TensorStorageType::BUFFER}),
            ValuesIn({/*allow_single_kernel_implementation=*/true,
                      /*allow_single_kernel_implementation=*/false}),
            ValuesIn({/*request_flash_decoding=*/true,
                      /*request_flash_decoding=*/false}),
            ValuesIn({MaskMode::kBool, MaskMode::kFloatAdditive,
                      MaskMode::kNone})),
    [](const TestParamInfo<SdpaTransposedKernelExecuteTest::ParamType>& info) {
      std::string name = absl::StrCat(
          ::ml_drift::ToString(std::get<0>(info.param)), "_",
          ::ml_drift::ToString(std::get<1>(info.param)), "_",
          std::get<2>(info.param) ? "AllowSingleKernel" : "ForcedMultiKernel",
          "_",
          std::get<3>(info.param) ? "RequestFlashDecoding" : "NoFlashDecoding",
          "_", ToString(std::get<4>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

}  // namespace
}  // namespace litert::ml_drift
