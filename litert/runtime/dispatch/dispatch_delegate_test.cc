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
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/internal/litert_compiled_model_next.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(DispatchDelegateTest, SwapInputTensorBufferBetweenInvocations) {
  // 1. Setup Environment
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  // 2. Load Model
  std::string model_path = litert::testing::GetTestFilePath("one_mul.tflite");

  // 3. Create CompiledModel
  // This handles plugin loading, application, compilation, and delegate
  // creation.
  LITERT_ASSERT_OK_AND_ASSIGN(auto compilation_options, Options::Create());
  LITERT_ASSERT_OK(compilation_options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModelNext::Create(env, model_path, compilation_options));

  // 4. Prepare for Execution
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  ASSERT_THAT(input_names, ::testing::Not(::testing::IsEmpty()));

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  for (auto input_name : input_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffer,
                                compiled_model.CreateInputBuffer(input_name));
    input_map[input_name] = std::move(input_buffer);
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  for (auto output_name : output_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(output_name));
    output_map[output_name] = std::move(output_buffer);
  }

  // 5. Run (First Invocation)
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));

  // 6. Swap Input Buffer
  // Create a NEW buffer for the same input tensor.
  auto input_name = input_names[0];
  LITERT_ASSERT_OK_AND_ASSIGN(auto new_input_buffer,
                              compiled_model.CreateInputBuffer(input_name));

  // Replace the buffer in the map.
  input_map[input_name] = std::move(new_input_buffer);

  // 7. Run (Second Invocation)
  // The delegate should detect the buffer change, safely release the old
  // buffer, and adopt the new one for execution.
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));
}

TEST(DispatchDelegateTest, PerRunOptionsPlumbedToDispatch) {
  // 1. Setup Environment
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  // 2. Load Model
  std::string model_path = litert::testing::GetTestFilePath("one_mul.tflite");

  // 3. Create CompiledModel (NPU via example dispatch)
  LITERT_ASSERT_OK_AND_ASSIGN(auto compilation_options, Options::Create());
  LITERT_ASSERT_OK(compilation_options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModelNext::Create(env, model_path, compilation_options));

  // 4. Prepare buffers
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  ASSERT_THAT(input_names, ::testing::Not(::testing::IsEmpty()));
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  for (auto input_name : input_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffer,
                                compiled_model.CreateInputBuffer(input_name));
    input_map[input_name] = std::move(input_buffer);

    LITERT_ASSERT_OK_AND_ASSIGN(auto input_type,
                                compiled_model.GetInputTensorType(input_name));
    LITERT_ASSERT_OK_AND_ASSIGN(const size_t input_bytes, input_type.Bytes());
    const size_t num_floats = input_bytes / sizeof(float);
    std::vector<float> input_data(num_floats, 3.0f);
    LITERT_ASSERT_OK(
        input_map[input_name].Write(absl::MakeConstSpan(input_data)));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  ASSERT_THAT(output_names, ::testing::Not(::testing::IsEmpty()));
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  for (auto output_name : output_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(output_name));
    output_map[output_name] = std::move(output_buffer);
  }

  // 5. Run without per-run options.
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));

  auto output_name = output_names[0];
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_type,
                              compiled_model.GetOutputTensorType(output_name));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t output_bytes, output_type.Bytes());
  const size_t num_out_floats = output_bytes / sizeof(float);
  std::vector<float> baseline(num_out_floats);
  LITERT_ASSERT_OK(output_map[output_name].Read(absl::MakeSpan(baseline)));

  // 6. Run with per-run options. The example dispatch scales outputs by 2 when
  // the per-run accelerator set contains CPU.
  LITERT_ASSERT_OK_AND_ASSIGN(auto run_options, Options::Create());
  LITERT_ASSERT_OK(
      run_options.SetHardwareAccelerators(litert::HwAccelerators::kCpu));

  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map, &run_options));
  std::vector<float> scaled(num_out_floats);
  LITERT_ASSERT_OK(output_map[output_name].Read(absl::MakeSpan(scaled)));

  ASSERT_EQ(scaled.size(), baseline.size());
  for (size_t i = 0; i < scaled.size(); ++i) {
    EXPECT_FLOAT_EQ(scaled[i], baseline[i] * 2.0f);
  }
}

}  // namespace
}  // namespace litert
