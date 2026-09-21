// Copyright 2025 Google LLC.
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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/vendors/qualcomm/core/utils/test_utils.h"

namespace litert {
namespace {

constexpr absl::string_view kDispatchLibraryDir = "vendors/qualcomm/dispatch";

TEST(CompiledModelTest, RunMultipleIterationsWithSameTensorBuffers) {
  if (!::qnn::IsTestHtpBackend()) {
    GTEST_SKIP() << "Skipping test because targeted backend is not supported";
  }

  const std::string dispatch_library_dir =
      testing::GetLiteRtPath(kDispatchLibraryDir);
  absl::string_view dispatch_library_dir_view(dispatch_library_dir);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          dispatch_library_dir_view,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, litert::Environment::Create(litert::EnvironmentOptions(
                           absl::MakeConstSpan(environment_options))));

  std::string model_file_path = testing::GetTestFilePath(
      "simple_model_qualcomm_sm8650_precompiled.tflite");

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));
  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());
  LITERT_LOG(LITERT_DEBUG, "Input/output buffers created");

  int num_iterations = 10;
  for (int i = 0; i < num_iterations; ++i) {
    LITERT_LOG(LITERT_DEBUG, "Iteration %d", i);
    LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));
  }
}

TEST(CompiledModelTest, RunMultipleIterationsWithNewTensorBuffers) {
  if (!::qnn::IsTestHtpBackend()) {
    GTEST_SKIP() << "Skipping test because targeted backend is not supported";
  }

  const std::string dispatch_library_dir =
      testing::GetLiteRtPath(kDispatchLibraryDir);
  absl::string_view dispatch_library_dir_view(dispatch_library_dir);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          dispatch_library_dir_view,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, litert::Environment::Create(litert::EnvironmentOptions(
                           absl::MakeConstSpan(environment_options))));

  std::string model_file_path = testing::GetTestFilePath(
      "simple_model_qualcomm_sm8650_precompiled.tflite");

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));
  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  // Creates and destroys tensor buffers each iteration to test proper memory
  // registration/deregistration in Qualcomm Dispatch.
  // Note: This number was chosen to be high enough to trigger memory
  // registration for a Tensor Buffer with the same pointer address.
  int num_iterations = 10;
  for (int i = 0; i < num_iterations; ++i) {
    LITERT_LOG(LITERT_DEBUG, "Iteration %d", i);
    LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                                compiled_model.CreateInputBuffers());
    LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                                compiled_model.CreateOutputBuffers());
    LITERT_LOG(LITERT_DEBUG, "Input/output buffers created");

    LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));
  }
}

struct BlockwiseE2eParam {
  const char* test_name;
  const char* model_name;
};

class BlockwiseE2eTest : public ::testing::TestWithParam<BlockwiseE2eParam> {};

TEST_P(BlockwiseE2eTest, CompileAndRunFromTflite) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "This test requires an Android device with a Qualcomm HTP.";
#else
  if (!::qnn::IsTestHtpBackend()) {
    GTEST_SKIP() << "Blockwise E2E is only supported by the HTP backend.";
  }

  const std::string dispatch_library_dir =
      testing::GetLiteRtPath(kDispatchLibraryDir);
  const std::string compiler_plugin_library_dir =
      ::qnn::GetTestDispatchLibraryDir();
  const std::vector<EnvironmentOptions::Option> environment_options = {
      EnvironmentOptions::Option{
          EnvironmentOptions::Tag::kDispatchLibraryDir,
          absl::string_view(dispatch_library_dir),
      },
      EnvironmentOptions::Option{
          EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          absl::string_view(compiler_plugin_library_dir),
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              Environment::Create(EnvironmentOptions(
                                  absl::MakeConstSpan(environment_options))));

  LITERT_ASSERT_OK_AND_ASSIGN(Options options, Options::Create());
  LITERT_ASSERT_OK(options.SetHardwareAccelerators(HwAccelerators::kNpu));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto& qnn_options,
      options.GetOptions<litert::qualcomm::QualcommOptions>());
  qnn_options.SetBackend(litert::qualcomm::QualcommOptions::Backend::kHtp);
  qnn_options.SetEnableJustInTime(true);

  const std::string model_path =
      testing::GetTestFilePath(GetParam().model_name);
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model_path, options));
  ASSERT_EQ(compiled_model.GetNumSignatures(), 1u);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());
  ASSERT_EQ(input_buffers.size(), 1u);
  ASSERT_EQ(output_buffers.size(), 1u);

  const std::vector<float> input_data(32, 1.0f);
  LITERT_ASSERT_OK(
      input_buffers[0].Write<float>(absl::MakeConstSpan(input_data)));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  std::vector<float> output_data(2);
  LITERT_ASSERT_OK(output_buffers[0].Read<float>(absl::MakeSpan(output_data)));
  ASSERT_THAT(output_data,
              ::testing::Pointwise(::testing::FloatNear(1e-2f),
                                   std::vector<float>{16.0f, 16.0f}));
#endif
}

INSTANTIATE_TEST_SUITE_P(
    QualcommBlockwise, BlockwiseE2eTest,
    ::testing::Values(
        BlockwiseE2eParam{"FullyConnectedW2Fp16",
                          "qualcomm_bq_fully_connected_w2_fp16.tflite"},
        BlockwiseE2eParam{"FullyConnectedW4Fp16",
                          "qualcomm_bq_fully_connected_w4_fp16.tflite"},
        BlockwiseE2eParam{"FullyConnectedW8Fp16",
                          "qualcomm_bq_fully_connected_w8_fp16.tflite"},
        BlockwiseE2eParam{"MatmulW2Fp16", "qualcomm_bq_matmul_w2_fp16.tflite"},
        BlockwiseE2eParam{"MatmulW4Fp16", "qualcomm_bq_matmul_w4_fp16.tflite"},
        BlockwiseE2eParam{"MatmulW8Fp16", "qualcomm_bq_matmul_w8_fp16.tflite"}),
    [](const ::testing::TestParamInfo<BlockwiseE2eParam>& info) {
      return info.param.test_name;
    });

}  // namespace

}  // namespace litert
