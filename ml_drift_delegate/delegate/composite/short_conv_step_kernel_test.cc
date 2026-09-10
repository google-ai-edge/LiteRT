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
  }
};

absl::Status RunShortConvStepTest(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision,
    ::ml_drift::TensorStorageType storage,
    int hidden_size,
    bool with_bias) {
  const int conv_L_cache = 3;
  const int num_slices = (hidden_size + 3) / 4;
  ::ml_drift::DataType data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);

  // 1. in_proj [1, 1, 1, 3 * hidden_size]
  ::ml_drift::TensorDescriptor in_proj_desc(
      data_type, storage, ::ml_drift::Layout::BHWC);
  in_proj_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 3 * hidden_size));

  // 2. conv_state [1, 1, hidden_size, conv_L_cache - 1]
  ::ml_drift::TensorDescriptor conv_state_desc(
      data_type, storage, ::ml_drift::Layout::BHWC);
  conv_state_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, 1, hidden_size, conv_L_cache - 1));

  // 3. conv_weight [hidden_size, 1, 1, conv_L_cache]
  ::ml_drift::TensorDescriptor conv_weight_desc(
      data_type, storage, ::ml_drift::Layout::BHWC);
  conv_weight_desc.SetBHWCShape(
      ::ml_drift::BHWC(hidden_size, 1, 1, conv_L_cache));

  // 4. conv_bias (optional) [1, 1, 1, hidden_size]
  std::unique_ptr<::ml_drift::TensorDescriptor> conv_bias_desc = nullptr;
  if (with_bias) {
    conv_bias_desc = std::make_unique<::ml_drift::TensorDescriptor>(
        data_type, storage, ::ml_drift::Layout::BHWC);
    conv_bias_desc->SetBHWCShape(::ml_drift::BHWC(1, 1, 1, hidden_size));
  }

  // 5. dst [1, 1, 1, hidden_size]
  ::ml_drift::TensorDescriptor dst_desc(
      data_type, storage, ::ml_drift::Layout::BHWC);
  dst_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, hidden_size));

  // 6. next_state [1, 1, hidden_size, conv_L_cache - 1]
  ::ml_drift::TensorDescriptor next_state_desc(
      data_type, storage, ::ml_drift::Layout::BHWC);
  next_state_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, 1, hidden_size, conv_L_cache - 1));

  // Create Op
  auto op = CreateFusedShortConvStep(
      in_proj_desc, conv_state_desc, conv_weight_desc,
      conv_bias_desc.get(), dst_desc, next_state_desc,
      num_slices, hidden_size, conv_L_cache);

  // Synthesize input data
  std::vector<float> in_proj_data(3 * hidden_size);
  // b in [0, hidden_size), c in [hidden_size, 2*hidden_size),
  // x in [2*hidden_size, 3*hidden_size)
  for (int i = 0; i < hidden_size; ++i) {
    in_proj_data[i] = 0.5f + 0.1f * (i % 7);                        // b
    in_proj_data[hidden_size + i] = 1.2f - 0.05f * (i % 5);         // c
    in_proj_data[2 * hidden_size + i] = 0.8f + 0.15f * (i % 9);     // x
  }

  std::vector<float> conv_state_data(hidden_size * 2);
  for (int i = 0; i < hidden_size; ++i) {
    conv_state_data[i * 2 + 0] = 0.2f * (i + 1);  // oldest
    conv_state_data[i * 2 + 1] = 0.3f * (i + 1);  // newest
  }

  std::vector<float> conv_weight_data(hidden_size * 3);
  for (int i = 0; i < hidden_size; ++i) {
    conv_weight_data[i * 3 + 0] = 0.1f + 0.01f * i;
    conv_weight_data[i * 3 + 1] = 0.2f + 0.02f * i;
    conv_weight_data[i * 3 + 2] = 0.3f + 0.03f * i;
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
  std::vector<float> zero_state(hidden_size * 2, 0.0f);
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

  std::vector<float> next_state_result(hidden_size * 2);
  next_state_desc.DownloadData(next_state_result.data());

  // Compute reference results
  const float tol =
      (precision == ::ml_drift::CalculationsPrecision::F16) ? 1e-2f : 1e-4f;
  for (int i = 0; i < hidden_size; ++i) {
    float b = in_proj_data[i];
    float c = in_proj_data[hidden_size + i];
    float x = in_proj_data[2 * hidden_size + i];
    float p = b * x;

    float s0 = conv_state_data[i * 2 + 0];
    float s1 = conv_state_data[i * 2 + 1];

    float w0 = conv_weight_data[i * 3 + 0];
    float w1 = conv_weight_data[i * 3 + 1];
    float w2 = conv_weight_data[i * 3 + 2];

    float conv_val = s0 * w0 + s1 * w1 + p * w2;
    if (with_bias) {
      conv_val += conv_bias_data[i];
    }
    float expected_y = c * conv_val;
    float expected_next_s0 = s1;
    float expected_next_s1 = p;

    float actual_y = dst_result[i];
    float actual_next_s0 = next_state_result[i * 2 + 0];
    float actual_next_s1 = next_state_result[i * 2 + 1];

    float max_err_y = std::max(tol, tol * std::abs(expected_y));
    if (std::abs(actual_y - expected_y) > max_err_y) {
      return absl::InternalError(absl::StrCat(
          "dst_result mismatch at channel ", i, ": expected ", expected_y,
          ", got ", actual_y,
          " (diff = ", std::abs(actual_y - expected_y), ")"));
    }
    float max_err_s0 = std::max(tol, tol * std::abs(expected_next_s0));
    if (std::abs(actual_next_s0 - expected_next_s0) > max_err_s0) {
      return absl::InternalError(absl::StrCat(
          "next_state[0] mismatch at channel ", i, ": expected ",
          expected_next_s0, ", got ", actual_next_s0));
    }
    float max_err_s1 = std::max(tol, tol * std::abs(expected_next_s1));
    if (std::abs(actual_next_s1 - expected_next_s1) > max_err_s1) {
      return absl::InternalError(absl::StrCat(
          "next_state[1] mismatch at channel ", i, ": expected ",
          expected_next_s1, ", got ", actual_next_s1));
    }
  }

  return absl::OkStatus();
}

TEST_P(ShortConvStepFloatTest, SmallSize) {
  if (!exec_env->IsStorageSupported(storage(), ::ml_drift::DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                /*hidden_size=*/4, /*with_bias=*/false));
}

TEST_P(ShortConvStepFloatTest, FullSizeHidden2048) {
  if (!exec_env->IsStorageSupported(storage(), ::ml_drift::DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                /*hidden_size=*/2048, /*with_bias=*/false));
}

TEST_P(ShortConvStepFloatTest, WithBias) {
  if (!exec_env->IsStorageSupported(storage(), ::ml_drift::DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunShortConvStepTest(*exec_env, precision(), storage(),
                                /*hidden_size=*/2048, /*with_bias=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    ShortConvStepFloatTestSuite, ShortConvStepFloatTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::F32,
                      ::ml_drift::CalculationsPrecision::F16}),
            ValuesIn({::ml_drift::TensorStorageType::BUFFER})),
    [](const TestParamInfo<ShortConvStepFloatTest::ParamType>& info) {
      return ::ml_drift::ToString(info.param);
    });

}  // namespace
}  // namespace litert::ml_drift
