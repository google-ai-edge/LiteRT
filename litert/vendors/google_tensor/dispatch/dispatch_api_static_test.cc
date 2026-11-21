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

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/core/model/model_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

// Simulates a precompiled device model so that we can run tests on the x86
// NPU Simulator.
Expected<litert::OwningBufferRef<uint8_t>> CreateModelFromReferenceData(
    LiteRtEnvironment env, LiteRtModel* model) {
  auto tfl_path = testing::GetTestFilePath("simple_model_npu.tflite");
  auto bin_path = testing::GetLiteRtPath(
      "vendors/google_tensor/dispatch/"
      "simple_model_reference_google_tensor.bin");

  LITERT_ASSIGN_OR_RETURN(
      auto model_buf_with_bytecode,
      litert::internal::GetModelBufWithByteCode(tfl_path, bin_path));

  LITERT_RETURN_IF_ERROR(
      LiteRtCreateModelFromBuffer(env, model_buf_with_bytecode.Data(),
                                  model_buf_with_bytecode.Size(), model));

  return model_buf_with_bytecode;
}

TEST(DispatchApiStaticLinkTest, RegistersStaticallyLinkedAccelerator) {
  // Create environment without setting kLiteRtEnvOptionTagDispatchLibraryDir.
  // The dynamically linked path would normally error.
  LiteRtEnvironment env;
  ASSERT_EQ(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env),
      kLiteRtStatusOk);
  LiteRtDestroyEnvironment(env);
}

TEST(DispatchApiStaticLinkTest, CanCreateCompiledModel) {
  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtModel model;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buffer_holder, CreateModelFromReferenceData(environment, &model));

  LiteRtOptions compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(compilation_options,
                                                 kLiteRtHwAcceleratorNpu),
            kLiteRtStatusOk);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model, compilation_options,
                                      &compiled_model),
            kLiteRtStatusOk);

  LiteRtDestroyOptions(compilation_options);
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(DispatchApiStaticLinkTest, CanRunCompiledModel) {
  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtModel model;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buffer_holder, CreateModelFromReferenceData(environment, &model));

  LiteRtOptions compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(compilation_options,
                                                 kLiteRtHwAcceleratorNpu),
            kLiteRtStatusOk);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model, compilation_options,
                                      &compiled_model),
            kLiteRtStatusOk);

  LiteRtSubgraph subgraph;
  ASSERT_EQ(LiteRtGetModelSubgraph(model, 0, &subgraph), kLiteRtStatusOk);

  LiteRtParamIndex num_inputs;
  ASSERT_EQ(LiteRtGetNumSubgraphInputs(subgraph, &num_inputs), kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> input_buffers;
  for (int i = 0; i < num_inputs; ++i) {
    LiteRtTensor input_tensor;
    ASSERT_EQ(LiteRtGetSubgraphInput(subgraph, i, &input_tensor),
              kLiteRtStatusOk);
    LiteRtRankedTensorType tensor_type;
    ASSERT_EQ(LiteRtGetRankedTensorType(input_tensor, &tensor_type),
              kLiteRtStatusOk);

    LiteRtTensorBufferRequirements requirements;
    ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                  compiled_model, /*signature_index=*/0, i, &requirements),
              kLiteRtStatusOk);

    LiteRtTensorBufferType buffer_type;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                  requirements, 0, &buffer_type),
              kLiteRtStatusOk);

    size_t buffer_size;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements,
                                                          &buffer_size),
              kLiteRtStatusOk);

    LiteRtTensorBuffer buffer;
    ASSERT_EQ(LiteRtCreateManagedTensorBuffer(environment, buffer_type,
                                              &tensor_type, buffer_size,
                                              &buffer),
              kLiteRtStatusOk);
    input_buffers.push_back(buffer);

    void* host_mem;
    ASSERT_EQ(LiteRtLockTensorBuffer(buffer, &host_mem,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memset(host_mem, 0, buffer_size);  // Fill with zeros
    ASSERT_EQ(LiteRtUnlockTensorBuffer(buffer), kLiteRtStatusOk);
  }

  LiteRtParamIndex num_outputs;
  ASSERT_EQ(LiteRtGetNumSubgraphOutputs(subgraph, &num_outputs),
            kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> output_buffers;
  for (int i = 0; i < num_outputs; ++i) {
    LiteRtTensor output_tensor;
    ASSERT_EQ(LiteRtGetSubgraphOutput(subgraph, i, &output_tensor),
              kLiteRtStatusOk);
    LiteRtRankedTensorType tensor_type;
    ASSERT_EQ(LiteRtGetRankedTensorType(output_tensor, &tensor_type),
              kLiteRtStatusOk);

    LiteRtTensorBufferRequirements requirements;
    ASSERT_EQ(LiteRtGetCompiledModelOutputBufferRequirements(
                  compiled_model, /*signature_index=*/0, i, &requirements),
              kLiteRtStatusOk);

    LiteRtTensorBufferType buffer_type;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                  requirements, 0, &buffer_type),
              kLiteRtStatusOk);

    size_t buffer_size;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements,
                                                          &buffer_size),
              kLiteRtStatusOk);

    LiteRtTensorBuffer buffer;
    ASSERT_EQ(LiteRtCreateManagedTensorBuffer(environment, buffer_type,
                                              &tensor_type, buffer_size,
                                              &buffer),
              kLiteRtStatusOk);
    output_buffers.push_back(buffer);
  }

  ASSERT_EQ(LiteRtRunCompiledModel(compiled_model, 0, input_buffers.size(),
                                   input_buffers.data(), output_buffers.size(),
                                   output_buffers.data()),
            kLiteRtStatusOk);

  for (auto b : input_buffers) LiteRtDestroyTensorBuffer(b);
  for (auto b : output_buffers) LiteRtDestroyTensorBuffer(b);

  LiteRtDestroyOptions(compilation_options);
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(DispatchApiStaticLinkTest, CanRunCompiledModelWithPerInvokeOptions) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  auto tfl_path = testing::GetTestFilePath("simple_model_npu.tflite");
  auto bin_path = testing::GetLiteRtPath(
      "vendors/google_tensor/dispatch/"
      "simple_model_reference_google_tensor.bin");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model_buf_with_bytecode,
      litert::internal::GetModelBufWithByteCode(tfl_path, bin_path));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env,
          BufferRef<uint8_t>(model_buf_with_bytecode.Data(),
                             model_buf_with_bytecode.Size()),
          HwAccelerators::kNpu));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());
  const size_t signature_index = 0;
  const float kIn0[] = {1.0f, 2.0f};
  const float kIn1[] = {10.0f, 20.0f};
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(kIn0));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(kIn1));

  LITERT_ASSERT_OK_AND_ASSIGN(auto run_options, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto& gt_opts,
      run_options.GetOptions<google_tensor::GoogleTensorOptions>());
  gt_opts.SetPerformanceMode(
      google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
  LITERT_ASSERT_OK(compiled_model.Run(signature_index, input_buffers,
                                      output_buffers, &run_options));
  float out[2] = {0.0f, 0.0f};
  LITERT_ASSERT_OK(output_buffers[0].Read<float>(out));
  EXPECT_FLOAT_EQ(out[0], 11.0f);
  EXPECT_FLOAT_EQ(out[1], 22.0f);

  // Out-of-range PerformanceMode fails the invocation.
  gt_opts.SetPerformanceMode(
      static_cast<google_tensor::GoogleTensorOptions::PerformanceMode>(6));
  LITERT_EXPECT_ERROR(compiled_model.Run(signature_index, input_buffers,
                                         output_buffers, &run_options));

  // Running without options resets PerformanceMode.
  const float kNewIn0[] = {3.0f, 4.0f};
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(kNewIn0));
  LITERT_ASSERT_OK(
      compiled_model.Run(signature_index, input_buffers, output_buffers));
  LITERT_ASSERT_OK(output_buffers[0].Read<float>(out));
  EXPECT_FLOAT_EQ(out[0], 13.0f);
  EXPECT_FLOAT_EQ(out[1], 24.0f);
}

}  // namespace
}  // namespace litert
