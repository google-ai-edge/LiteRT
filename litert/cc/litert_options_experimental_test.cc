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

#include "litert/cc/litert_options_experimental.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/matchers.h"
#include "tflite/builtin_ops.h"
#include "tflite/core/c/c_api_opaque.h"
#include "tflite/core/c/common.h"

// Forward declare the custom op registration function.
namespace tflite {
namespace ops {
namespace custom {
TfLiteRegistration* Register_MAX_UNPOOLING2D();
}  // namespace custom
}  // namespace ops
}  // namespace tflite

namespace litert {
namespace {

TEST(OptionsExperimentalTest, MaxUnpooling2D) {
  auto env_expected = litert::Environment::Create({});
  ASSERT_TRUE(env_expected.HasValue());
  auto env = std::move(env_expected.Value());

  auto options_expected = Options::Create();
  ASSERT_TRUE(options_expected.HasValue());
  auto options = std::move(options_expected.Value());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  auto cpu_options_expected = options.GetCpuOptions();
  ASSERT_TRUE(cpu_options_expected.HasValue());
  auto& cpu_options = cpu_options_expected.Value();
  cpu_options.SetKernelMode(kLiteRtCpuKernelModeBuiltin);

  TfLiteRegistration* reg = ::tflite::ops::custom::Register_MAX_UNPOOLING2D();
  reg->custom_name = "LiteRtMaxUnpooling2D";
  reg->version = 1;

  // Use the new experimental API
  LITERT_ASSERT_OK(internal::AddCustomOp(options, reg));

  std::string model_path =
      "third_party/odml/litert/litert/cc/max_unpooling_2d.tflite";

  auto compiled_model_expected =
      CompiledModel::Create(env, model_path, options);
  ASSERT_TRUE(compiled_model_expected.HasValue());
  auto compiled_model = std::move(compiled_model_expected.Value());

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  std::vector<float> input0_data(8192, 1.0f);
  std::vector<int32_t> input1_data(8192, 2);

  ASSERT_TRUE(input_buffers[0].Write<float>(absl::MakeConstSpan(input0_data)));
  ASSERT_TRUE(
      input_buffers[1].Write<int32_t>(absl::MakeConstSpan(input1_data)));

  compiled_model.Run(input_buffers, output_buffers);

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 8192);
    ABSL_LOG(INFO) << "Output[0]: " << output[0];
    // You might want to add more specific assertions here based on expected
    // output.
  }
}

TEST(OptionsExperimentalTest, MaxUnpooling2D_Valid) {
  auto env_expected = litert::Environment::Create({});
  ASSERT_TRUE(env_expected.HasValue());
  auto env = std::move(env_expected.Value());

  auto options_expected = Options::Create();
  ASSERT_TRUE(options_expected.HasValue());
  auto options = std::move(options_expected.Value());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  auto cpu_options_expected = options.GetCpuOptions();
  ASSERT_TRUE(cpu_options_expected.HasValue());
  auto& cpu_options = cpu_options_expected.Value();
  cpu_options.SetKernelMode(kLiteRtCpuKernelModeBuiltin);

  TfLiteRegistration* reg = ::tflite::ops::custom::Register_MAX_UNPOOLING2D();
  reg->custom_name = "LiteRtMaxUnpooling2D";
  reg->version = 1;

  // Use the new experimental API
  LITERT_ASSERT_OK(internal::AddCustomOp(options, reg));

  auto model_path =
      "third_party/odml/litert/litert/cc/max_unpooling_2d_valid.tflite";

  auto compiled_model_expected =
      CompiledModel::Create(env, model_path, options);
  ASSERT_TRUE(compiled_model_expected.HasValue());
  auto compiled_model = std::move(compiled_model_expected.Value());

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  std::vector<float> input_data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<int32_t> indices_data{2,  23, 8,  9,  12, 15, 40, 43, 44, 47, 72,
                                    75, 80, 79, 62, 65, 0,  1,  30, 7,  14, 35,
                                    42, 21, 68, 69, 50, 51, 56, 5,  86, 63};

  ASSERT_TRUE(input_buffers[0].Write<float>(absl::MakeConstSpan(input_data)));
  ASSERT_TRUE(
      input_buffers[1].Write<int32_t>(absl::MakeConstSpan(indices_data)));

  compiled_model.Run(input_buffers, output_buffers);

  std::vector<float> expected_output{
      0,  0,  1, 0,  0,  0,  0, 0,  3,  4, 0,  0,  5,  0,  0, 6,  0,  0,
      0,  0,  0, 0,  0,  2,  0, 0,  0,  0, 0,  0,  0,  0,  0, 0,  0,  0,
      0,  0,  0, 0,  7,  0,  0, 8,  9,  0, 0,  10, 0,  0,  0, 0,  0,  0,
      0,  0,  0, 0,  0,  0,  0, 0,  15, 0, 0,  16, 0,  0,  0, 0,  0,  0,
      11, 0,  0, 12, 0,  0,  0, 14, 13, 0, 0,  0,  0,  0,  0, 0,  17, 18,
      0,  0,  0, 30, 0,  20, 0, 0,  0,  0, 0,  0,  21, 0,  0, 0,  0,  0,
      0,  24, 0, 0,  0,  0,  0, 0,  0,  0, 19, 0,  0,  0,  0, 22, 0,  0,
      0,  0,  0, 0,  23, 0,  0, 0,  0,  0, 0,  0,  27, 28, 0, 0,  0,  0,
      29, 0,  0, 0,  0,  0,  0, 32, 0,  0, 0,  0,  25, 26, 0, 0,  0,  0,
      0,  0,  0, 0,  0,  0,  0, 0,  0,  0, 0,  0,  31, 0};

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 176);
    EXPECT_THAT(output, ::testing::Pointwise(::testing::FloatNear(1e-5),
                                             expected_output));
  }
}

namespace {
TfLiteStatus SimplePrepare(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node) {
  return kTfLiteOk;
}

TfLiteStatus SimpleInvoke(TfLiteOpaqueContext* context,
                          TfLiteOpaqueNode* node) {
  const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
  TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);

  void* input_data = TfLiteOpaqueTensorData(input);
  void* output_data = TfLiteOpaqueTensorData(output);
  size_t bytes = TfLiteOpaqueTensorByteSize(input);

  std::memcpy(output_data, input_data, bytes);
  return kTfLiteOk;
}
}  // namespace

TEST(OptionsExperimentalTest, SinhCustomOp) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  LITERT_ASSERT_OK_AND_ASSIGN(auto& cpu_options, options.GetCpuOptions());
  cpu_options.SetKernelMode(kLiteRtCpuKernelModeBuiltin);

  TfLiteOperator* reg =
      TfLiteOperatorCreate(kTfLiteBuiltinCustom, "Sinh", /*version=*/1,
                           /*user_data=*/nullptr);
  absl::Cleanup cleanup_reg = [reg] { TfLiteOperatorDelete(reg); };
  TfLiteOperatorSetPrepare(reg, SimplePrepare);
  TfLiteOperatorSetInvoke(reg, SimpleInvoke);

  LITERT_ASSERT_OK(internal::AddCustomOp(options, reg));

  std::string model_path =
      "third_party/tensorflow/lite/testdata/custom_sinh.bin";

  auto compiled_model_expected =
      CompiledModel::Create(env, model_path, options);
  ASSERT_TRUE(compiled_model_expected.HasValue());
  auto compiled_model = std::move(compiled_model_expected.Value());

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  float input_value = 42.0f;
  ASSERT_TRUE(
      input_buffers[0].Write<float>(absl::MakeConstSpan(&input_value, 1)));

  compiled_model.Run(input_buffers, output_buffers);

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 1);
    EXPECT_FLOAT_EQ(output[0], input_value);
  }
}

}  // namespace
}  // namespace litert
