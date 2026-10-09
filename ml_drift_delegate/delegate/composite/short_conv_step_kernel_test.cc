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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
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
  ::ml_drift::TensorDescriptor in_proj_desc(data_type, storage,
                                            ::ml_drift::Layout::kBHWC);
  in_proj_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 3 * hidden_size));

  // 2. conv_state [1, 1, hidden_size, conv_L_cache - 1]
  ::ml_drift::TensorDescriptor conv_state_desc(data_type, storage,
                                               ::ml_drift::Layout::kBHWC);
  conv_state_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, 1, hidden_size, conv_L_cache - 1));

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

  // 6. next_state [1, 1, hidden_size, conv_L_cache - 1]
  ::ml_drift::TensorDescriptor next_state_desc(data_type, storage,
                                               ::ml_drift::Layout::kBHWC);
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
      (precision == ::ml_drift::CalculationsPrecision::kF16) ? 1e-2f : 1e-4f;
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

struct ShortConvPrefillTestCase {
  int hidden_size;
  int seq_len;
  std::optional<int> num_valid_tokens;
  int conv_l_cache;
  bool with_bias;
};

::ml_drift::TensorDescriptor MakeTensorDesc(
    ::ml_drift::DataType data_type, ::ml_drift::TensorStorageType storage,
    const ::ml_drift::BHWC& shape) {
  ::ml_drift::TensorDescriptor desc(data_type, storage,
                                    ::ml_drift::Layout::kBHWC);
  desc.SetBHWCShape(shape);
  return desc;
}

absl::Status CheckNear(absl::string_view name, int i, int j, float expected,
                       float actual, float tol) {
  const float max_err = std::max(tol, tol * std::abs(expected));
  if (std::abs(actual - expected) > max_err) {
    return absl::InternalError(absl::StrCat(name, " mismatch at [", i, ", ", j,
                                            "]: expected ", expected, ", got ",
                                            actual));
  }
  return absl::OkStatus();
}

absl::Status RunShortConvPrefillTest(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision,
    ::ml_drift::TensorStorageType storage,
    const ShortConvPrefillTestCase& test_case) {
  const int hidden_size = test_case.hidden_size;
  const int seq_len = test_case.seq_len;
  const int conv_L_cache = test_case.conv_l_cache;
  const int num_state_taps = conv_L_cache - 1;
  const int num_slices = (hidden_size + 3) / 4;
  const ::ml_drift::DataType data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);

  // The shapes follow how the delegate maps the composite operands to BHWC:
  // in_proj [1, S, 3 * H] -> (1, 1, S, 3 * H), conv_state [1, H, L - 1] ->
  // (1, 1, H, L - 1), conv_weight [H, 1, L] -> (H, 1, 1, L), and 1D
  // num_valid_tokens [1] and conv_bias [H] -> (N, 1, 1, 1).
  ::ml_drift::TensorDescriptor in_proj_desc = MakeTensorDesc(
      data_type, storage, ::ml_drift::BHWC(1, 1, seq_len, 3 * hidden_size));
  ::ml_drift::TensorDescriptor conv_state_desc = MakeTensorDesc(
      data_type, storage, ::ml_drift::BHWC(1, 1, hidden_size, num_state_taps));
  ::ml_drift::TensorDescriptor conv_weight_desc = MakeTensorDesc(
      data_type, storage, ::ml_drift::BHWC(hidden_size, 1, 1, conv_L_cache));
  std::unique_ptr<::ml_drift::TensorDescriptor> conv_bias_desc;
  if (test_case.with_bias) {
    conv_bias_desc =
        std::make_unique<::ml_drift::TensorDescriptor>(MakeTensorDesc(
            data_type, storage, ::ml_drift::BHWC(hidden_size, 1, 1, 1)));
  }
  std::unique_ptr<::ml_drift::TensorDescriptor> num_valid_tokens_desc;
  if (test_case.num_valid_tokens.has_value()) {
    num_valid_tokens_desc = std::make_unique<::ml_drift::TensorDescriptor>(
        MakeTensorDesc(::ml_drift::DataType::kInt32, storage,
                       ::ml_drift::BHWC(1, 1, 1, 1)));
  }
  ::ml_drift::TensorDescriptor dst_desc = MakeTensorDesc(
      data_type, storage, ::ml_drift::BHWC(1, 1, seq_len, hidden_size));
  ::ml_drift::TensorDescriptor next_state_desc = MakeTensorDesc(
      data_type, storage, ::ml_drift::BHWC(1, 1, hidden_size, num_state_taps));

  auto op = CreateFusedShortConvStep(
      in_proj_desc, conv_state_desc, conv_weight_desc, conv_bias_desc.get(),
      dst_desc, next_state_desc, num_slices, hidden_size, conv_L_cache,
      num_valid_tokens_desc.get());

  // The inputs vary with both the token and the channel, so that reading the
  // wrong token, channel or tap changes the result.
  std::vector<float> in_proj_data(seq_len * 3 * hidden_size);
  for (int t = 0; t < seq_len; ++t) {
    float* row = &in_proj_data[t * 3 * hidden_size];
    for (int i = 0; i < hidden_size; ++i) {
      row[i] = 0.2f + 0.1f * ((i + 3 * t) % 7);                      // B
      row[hidden_size + i] = 1.2f - 0.15f * ((2 * i + t) % 5);       // C
      row[2 * hidden_size + i] = -0.6f + 0.15f * ((i + 5 * t) % 9);  // x
    }
  }
  std::vector<float> conv_state_data(hidden_size * num_state_taps);
  for (int i = 0; i < hidden_size; ++i) {
    for (int j = 0; j < num_state_taps; ++j) {
      conv_state_data[i * num_state_taps + j] =
          -0.5f + 0.1f * ((i + 4 * j) % 11);
    }
  }
  std::vector<float> conv_weight_data(hidden_size * conv_L_cache);
  for (int i = 0; i < hidden_size; ++i) {
    for (int k = 0; k < conv_L_cache; ++k) {
      conv_weight_data[i * conv_L_cache + k] =
          (k % 2 == 0 ? 1.0f : -1.0f) * (0.1f + 0.03f * ((i + k) % 13));
    }
  }
  std::vector<float> conv_bias_data;
  if (test_case.with_bias) {
    conv_bias_data.resize(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
      conv_bias_data[i] = -0.05f + 0.05f * (i % 3);
    }
  }
  std::vector<int32_t> num_valid_tokens_data;
  if (test_case.num_valid_tokens.has_value()) {
    num_valid_tokens_data = {*test_case.num_valid_tokens};
  }

  in_proj_desc.UploadData(in_proj_data.data());
  conv_state_desc.UploadData(conv_state_data.data());
  conv_weight_desc.UploadData(conv_weight_data.data());
  if (test_case.with_bias) {
    conv_bias_desc->UploadData(conv_bias_data.data());
  }
  if (test_case.num_valid_tokens.has_value()) {
    num_valid_tokens_desc->UploadData(num_valid_tokens_data.data());
  }
  // Fill the outputs with a sentinel so that elements the kernel does not
  // write fail the comparison.
  const std::vector<float> dst_init(seq_len * hidden_size, -100.0f);
  dst_desc.UploadData(dst_init.data());
  const std::vector<float> next_state_init(hidden_size * num_state_taps,
                                           -100.0f);
  next_state_desc.UploadData(next_state_init.data());

  std::vector<::ml_drift::TensorDescriptor*> src_cpu = {
      &in_proj_desc, &conv_state_desc, &conv_weight_desc};
  if (test_case.with_bias) {
    src_cpu.push_back(conv_bias_desc.get());
  }
  if (test_case.num_valid_tokens.has_value()) {
    src_cpu.push_back(num_valid_tokens_desc.get());
  }
  std::vector<::ml_drift::TensorDescriptor*> dst_cpu = {&dst_desc,
                                                        &next_state_desc};
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation(src_cpu, dst_cpu, std::move(op)));

  std::vector<float> dst_result(seq_len * hidden_size);
  dst_desc.DownloadData(dst_result.data());
  std::vector<float> next_state_result(hidden_size * num_state_taps);
  next_state_desc.DownloadData(next_state_result.data());

  // padded = concat(conv_state, B * x) along the sequence: padded[p] is
  // conv_state tap p for p < L - 1 and B * x of token p - (L - 1) otherwise.
  auto padded = [&](int i, int p) {
    if (p < num_state_taps) {
      return conv_state_data[i * num_state_taps + p];
    }
    const float* row = &in_proj_data[(p - num_state_taps) * 3 * hidden_size];
    return row[i] * row[2 * hidden_size + i];
  };
  const float tol =
      (precision == ::ml_drift::CalculationsPrecision::kF16) ? 1e-2f : 1e-4f;
  for (int t = 0; t < seq_len; ++t) {
    for (int i = 0; i < hidden_size; ++i) {
      float conv = test_case.with_bias ? conv_bias_data[i] : 0.0f;
      for (int k = 0; k < conv_L_cache; ++k) {
        conv += conv_weight_data[i * conv_L_cache + k] * padded(i, t + k);
      }
      const float expected =
          in_proj_data[t * 3 * hidden_size + hidden_size + i] * conv;
      ABSL_RETURN_IF_ERROR(
          CheckNear("y", t, i, expected, dst_result[t * hidden_size + i], tol));
    }
  }
  const int n =
      std::clamp(test_case.num_valid_tokens.value_or(seq_len), 0, seq_len);
  for (int i = 0; i < hidden_size; ++i) {
    for (int j = 0; j < num_state_taps; ++j) {
      ABSL_RETURN_IF_ERROR(CheckNear("next_state", i, j, padded(i, n + j),
                                     next_state_result[i * num_state_taps + j],
                                     tol));
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

TEST_P(ShortConvStepFloatTest, PrefillAllTokensValid) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 4,
                                     .seq_len = 5,
                                     .num_valid_tokens = 5,
                                     .conv_l_cache = 3,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillWithoutNumValidTokens) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 6,
                                     .num_valid_tokens = std::nullopt,
                                     .conv_l_cache = 3,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillPaddedSequenceHidden2048) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 2048,
                                     .seq_len = 16,
                                     .num_valid_tokens = 11,
                                     .conv_l_cache = 3,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillFewerValidTokensThanStateTaps) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 6,
                                     .num_valid_tokens = 1,
                                     .conv_l_cache = 3,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillNoValidTokens) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 6,
                                     .num_valid_tokens = 0,
                                     .conv_l_cache = 3,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillSingleTokenWithNumValidTokens) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 1,
                                     .num_valid_tokens = 1,
                                     .conv_l_cache = 3,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillWithBias) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 2048,
                                     .seq_len = 16,
                                     .num_valid_tokens = 16,
                                     .conv_l_cache = 3,
                                     .with_bias = true}));
}

TEST_P(ShortConvStepFloatTest, PrefillConvLCache2) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 7,
                                     .num_valid_tokens = 4,
                                     .conv_l_cache = 2,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillConvLCache4WithBias) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 9,
                                     .num_valid_tokens = 2,
                                     .conv_l_cache = 4,
                                     .with_bias = true}));
}

TEST_P(ShortConvStepFloatTest, SingleTokenConvLCache4WithBias) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 1,
                                     .num_valid_tokens = std::nullopt,
                                     .conv_l_cache = 4,
                                     .with_bias = true}));
}

TEST_P(ShortConvStepFloatTest, PrefillShorterThanStateTaps) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 2,
                                     .num_valid_tokens = std::nullopt,
                                     .conv_l_cache = 4,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillShorterThanStateTapsWithNumValidTokens) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 2,
                                     .num_valid_tokens = 1,
                                     .conv_l_cache = 4,
                                     .with_bias = false}));
}

TEST_P(ShortConvStepFloatTest, PrefillWithoutNumValidTokensConvLCache2) {
  ASSERT_OK(RunShortConvPrefillTest(*exec_env, precision(), storage(),
                                    {.hidden_size = 8,
                                     .seq_len = 5,
                                     .num_valid_tokens = std::nullopt,
                                     .conv_l_cache = 2,
                                     .with_bias = false}));
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
