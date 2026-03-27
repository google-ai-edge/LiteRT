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
#include "litert/test/common.h"


namespace litert {
namespace {

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
  auto path = testing::GetTestFilePath(
      "simple_model_npu_google_tensor_precompiled.tflite");

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  LiteRtOptions compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(compilation_options,
                                                 kLiteRtHwAcceleratorNpu),
            kLiteRtStatusOk);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
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
  auto path = testing::GetTestFilePath(
      "simple_model_npu_google_tensor_precompiled.tflite");

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  LiteRtOptions compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(compilation_options,
                                                 kLiteRtHwAcceleratorNpu),
            kLiteRtStatusOk);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
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

}  // namespace
}  // namespace litert
