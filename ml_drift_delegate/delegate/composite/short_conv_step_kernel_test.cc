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

#include "ml_drift_delegate/delegate/composite/short_conv_step_kernel.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class ShortConvStepFloatTest : public ::ml_drift::FloatTest {
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
};

absl::Status RunShortConvStepTest(::ml_drift::TestExecutionEnvironment& env,
                                  ::ml_drift::CalculationsPrecision precision,
                                  ::ml_drift::TensorStorageType storage,
                                  int hidden_size, bool with_bias,
                                  int conv_L_cache = 3, bool is_gated = true,
                                  bool use_silu = false) {
  const int state_len = conv_L_cache - 1;
  const int in_channels = is_gated ? (3 * hidden_size) : hidden_size;
  const int num_slices = (hidden_size + 3) / 4;
  ::ml_drift::DataType data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);

  // 1. in_proj [1, 1, 1, in_channels]
  ::ml_drift::TensorDescriptor in_proj_desc(data_type, storage,
                                            ::ml_drift::Layout::kBHWC);
  in_proj_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, in_channels));

  // 2. conv_state [1, 1, hidden_size, state_len]
  ::ml_drift::TensorDescriptor conv_state_desc(data_type, storage,
                                               ::ml_drift::Layout::kBHWC);
  conv_state_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, hidden_size, state_len));

  // 3. conv_weight [hidden_size, 1, 1, conv_L_cache]
  ::ml_drift::TensorDescriptor conv_weight_desc(data_type, storage,
                                                ::ml_drift::Layout::kBHWC);
  conv_weight_desc.SetBHWCShape(
      ::ml_drift::BHWC(hidden_size, 1, 1, conv_L_cache));

  // 4. conv_bias (optional) [1, 1, 1, hidden_size]
  std::unique_ptr<::ml_drift::TensorDescriptor> conv_bias_desc = nullptr;
  if (with_bias) {
    conv_bias_desc = std::make_unique<::ml_drift::TensorDescriptor>(
        data_type, storage, ::ml_drift::Layout::kBHWC);
    conv_bias_desc->SetBHWCShape(::ml_drift::BHWC(1, 1, 1, hidden_size));
  }

  // 5. dst [1, 1, 1, hidden_size]
  ::ml_drift::TensorDescriptor dst_desc(data_type, storage,
                                        ::ml_drift::Layout::kBHWC);
  dst_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, hidden_size));

  // 6. next_state [1, 1, hidden_size, state_len]
  ::ml_drift::TensorDescriptor next_state_desc(data_type, storage,
                                               ::ml_drift::Layout::kBHWC);
  next_state_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, hidden_size, state_len));

  // Create Op
  auto op = CreateFusedShortConvStep(
      in_proj_desc, conv_state_desc, conv_weight_desc, conv_bias_desc.get(),
      dst_desc, next_state_desc, num_slices, hidden_size, conv_L_cache,
      is_gated, use_silu);

  // Synthesize input data
  std::vector<float> in_proj_data(in_channels);
  if (is_gated) {
    for (int i = 0; i < hidden_size; ++i) {
      in_proj_data[i] = 0.5f + 0.1f * (i % 7);                     // b
      in_proj_data[hidden_size + i] = 1.2f - 0.05f * (i % 5);      // c
      in_proj_data[2 * hidden_size + i] = 0.8f + 0.15f * (i % 9);  // x
    }
  } else {
    for (int i = 0; i < hidden_size; ++i) {
      in_proj_data[i] = -0.4f + 0.15f * (i % 9);
    }
  }

  std::vector<float> conv_state_data(hidden_size * state_len);
  for (int i = 0; i < hidden_size; ++i) {
    for (int t = 0; t < state_len; ++t) {
      conv_state_data[i * state_len + t] = 0.1f * (t + 2) * ((i % 11) + 1);
    }
  }

  std::vector<float> conv_weight_data(hidden_size * conv_L_cache);
  for (int i = 0; i < hidden_size; ++i) {
    for (int k = 0; k < conv_L_cache; ++k) {
      conv_weight_data[i * conv_L_cache + k] =
          0.1f * (k + 1) + 0.01f * (k + 1) * (i % 13);
    }
  }

  std::vector<float> conv_bias_data;
  if (with_bias) {
    conv_bias_data.resize(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
      conv_bias_data[i] = 0.05f * (i % 3);
    }
  }

  // Upload data
  in_proj_desc.UploadData(in_proj_data.data());
  conv_state_desc.UploadData(conv_state_data.data());
  conv_weight_desc.UploadData(conv_weight_data.data());
  if (with_bias) {
    conv_bias_desc->UploadData(conv_bias_data.data());
  }

  // Initialize output buffers
  std::vector<float> zero_out(hidden_size, 0.0f);
  dst_desc.UploadData(zero_out.data());
  std::vector<float> zero_state(hidden_size * state_len, 0.0f);
  next_state_desc.UploadData(zero_state.data());

  std::vector<::ml_drift::TensorDescriptor*> src_cpu = {
      &in_proj_desc, &conv_state_desc, &conv_weight_desc};
  if (with_bias) {
    src_cpu.push_back(conv_bias_desc.get());
  }
  std::vector<::ml_drift::TensorDescriptor*> dst_cpu = {
      &dst_desc, &next_state_desc};

  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation(src_cpu, dst_cpu, std::move(op)));

  std::vector<float> dst_result(hidden_size);
  dst_desc.DownloadData(dst_result.data());

  std::vector<float> next_state_result(hidden_size * state_len);
  next_state_desc.DownloadData(next_state_result.data());

  // Compute reference results
  const float tol =
      (precision == ::ml_drift::CalculationsPrecision::kF16) ? 1e-2f : 1e-4f;
  for (int i = 0; i < hidden_size; ++i) {
    float p = 0.0f;
    float c = 1.0f;
    if (is_gated) {
      float b = in_proj_data[i];
      c = in_proj_data[hidden_size + i];
      float x = in_proj_data[2 * hidden_size + i];
      p = b * x;
    } else {
      p = in_proj_data[i];
    }

    float conv_val = 0.0f;
    for (int t = 0; t < state_len; ++t) {
      conv_val += conv_state_data[i * state_len + t] *
                  conv_weight_data[i * conv_L_cache + t];
    }
    conv_val += p * conv_weight_data[i * conv_L_cache + state_len];
    if (with_bias) {
      conv_val += conv_bias_data[i];
    }
    if (use_silu) {
      conv_val = conv_val / (1.0f + std::exp(-conv_val));
    }
    float expected_y = is_gated ? (c * conv_val) : conv_val;

    float actual_y = dst_result[i];
    float max_err_y = std::max(tol, tol * std::abs(expected_y));
    if (std::abs(actual_y - expected_y) > max_err_y) {
      return absl::InternalError(absl::StrCat(
          "dst_result mismatch at channel ", i, ": expected ", expected_y,
          ", got ", actual_y,
          " (diff = ", std::abs(actual_y - expected_y), ")"));
    }

    for (int t = 0; t < state_len; ++t) {
      float expected_next_s =
          (t + 1 < state_len) ? conv_state_data[i * state_len + t + 1] : p;
      float actual_next_s = next_state_result[i * state_len + t];
      float max_err_s = std::max(tol, tol * std::abs(expected_next_s));
      if (std::abs(actual_next_s - expected_next_s) > max_err_s) {
        return absl::InternalError(absl::StrCat(
            "next_state[", t, "] mismatch at channel ", i, ": expected ",
            expected_next_s, ", got ", actual_next_s));
      }
    }
  }

  return absl::OkStatus();
}

TEST_P(ShortConvStepFloatTest, SmallSize) {
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                /*hidden_size=*/4, /*with_bias=*/false));
}

TEST_P(ShortConvStepFloatTest, FullSizeHidden2048) {
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                /*hidden_size=*/2048, /*with_bias=*/false));
}

TEST_P(ShortConvStepFloatTest, WithBias) {
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                /*hidden_size=*/2048, /*with_bias=*/true));
}

TEST_P(ShortConvStepFloatTest, UngatedSiluConv4Small) {
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                 /*hidden_size=*/16, /*with_bias=*/false,
                                 /*conv_L_cache=*/4, /*is_gated=*/false,
                                 /*use_silu=*/true));
}

TEST_P(ShortConvStepFloatTest, UngatedSiluConv4FullSize) {
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                 /*hidden_size=*/6144, /*with_bias=*/false,
                                 /*conv_L_cache=*/4, /*is_gated=*/false,
                                 /*use_silu=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    ShortConvStepFloatTestSuite, ShortConvStepFloatTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::kF32,
                      ::ml_drift::CalculationsPrecision::kF16}),
            ValuesIn({::ml_drift::TensorStorageType::kBuffer})),
    [](const TestParamInfo<ShortConvStepFloatTest::ParamType>& info) {
      return ::ml_drift::ToString(info.param);
    });

}  // namespace
}  // namespace litert::ml_drift
