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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

constexpr absl::string_view kDispatchLibraryDir = "vendors/qualcomm/dispatch";

TEST(CompiledModelTest, RunMultipleIterationsWithSameTensorBuffers) {
  const std::string dispatch_library_dir =
      testing::GetLiteRtPath(kDispatchLibraryDir);
  absl::string_view dispatch_library_dir_view(dispatch_library_dir);
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          dispatch_library_dir_view,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

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
  const std::string dispatch_library_dir =
      testing::GetLiteRtPath(kDispatchLibraryDir);
  absl::string_view dispatch_library_dir_view(dispatch_library_dir);
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          dispatch_library_dir_view,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

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

}  // namespace

}  // namespace litert
