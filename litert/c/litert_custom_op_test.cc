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

#include <cstddef>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_custom_op_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

namespace mycustomop {

constexpr const char* kCustomOpName = "MyCustomOp";
constexpr int kCustomOpVersion = 1;

LiteRtStatus Init(void* user_data, const void* init_data,
                  size_t init_data_size) {
  return kLiteRtStatusOk;
}

LiteRtStatus GetOutputLayouts(void* user_data, size_t num_inputs,
                              const LiteRtLayout* input_layouts,
                              size_t num_outputs,
                              LiteRtLayout* output_layouts) {
  if (!(num_inputs == 2 && num_outputs == 1)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  output_layouts[0] = input_layouts[0];

  return kLiteRtStatusOk;
}

LiteRtStatus Run(void* user_data, size_t num_inputs,
                 const LiteRtTensorBuffer* inputs, size_t num_outputs,
                 LiteRtTensorBuffer* outputs) {
  if (!(num_inputs == 2 && num_outputs == 1)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferTensorType(outputs[0], &tensor_type));

  size_t num_elements;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetNumLayoutElements(&tensor_type.layout, &num_elements));

  void* input0_addr;
  LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
      inputs[0], &input0_addr, kLiteRtTensorBufferLockModeRead));

  void* input1_addr;
  LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
      inputs[1], &input1_addr, kLiteRtTensorBufferLockModeRead));

  void* output_addr;
  LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
      outputs[0], &output_addr, kLiteRtTensorBufferLockModeWrite));

  auto* input0 = static_cast<const float*>(input0_addr);
  auto* input1 = static_cast<const float*>(input1_addr);
  auto* output = static_cast<float*>(output_addr);

  for (size_t i = 0; i < num_elements; ++i) {
    output[i] = input0[i] + input1[i];
  }

  LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(inputs[0]));
  LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(inputs[1]));
  LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(outputs[0]));

  return kLiteRtStatusOk;
}

LiteRtStatus Destroy(void* user_data) {
  // Nothing to do.
  return kLiteRtStatusOk;
}

}  // namespace mycustomop

TEST(CompiledModelTest, CustomOp) {
  auto path = testing::GetTestFilePath(kModelFileName);

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);

  LiteRtCustomOpKernel custom_op_kernel = {
      /*.init=*/mycustomop::Init,
      /*.get_output_layouts=*/mycustomop::GetOutputLayouts,
      /*.run=*/mycustomop::Run,
      /*.destroy=*/mycustomop::Destroy,
  };
  ASSERT_EQ(LiteRtAddCustomOpKernelOption(
                jit_compilation_options, mycustomop::kCustomOpName,
                mycustomop::kCustomOpVersion, &custom_op_kernel,
                /*custom_op_kernel_user_data=*/nullptr),
            kLiteRtStatusOk);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model,
                                      jit_compilation_options, &compiled_model),
            kLiteRtStatusOk);

  LiteRtDestroyOptions(jit_compilation_options);

  LiteRtSubgraph subgraph;
  ASSERT_EQ(LiteRtGetModelSubgraph(model, 0, &subgraph), kLiteRtStatusOk);

  LiteRtParamIndex num_inputs;
  ASSERT_EQ(LiteRtGetNumSubgraphInputs(subgraph, &num_inputs), kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> input_tensor_buffers;
  input_tensor_buffers.reserve(num_inputs);
  for (auto i = 0; i < num_inputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                  compiled_model, /*signature_index=*/0, i,
                  &tensor_buffer_requirements),
              kLiteRtStatusOk);
    LiteRtTensorBufferType tensor_buffer_type;
    EXPECT_EQ(
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
            tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type),
        kLiteRtStatusOk);
    size_t tensor_buffer_size;
    EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                  tensor_buffer_requirements, &tensor_buffer_size),
              kLiteRtStatusOk);
    LiteRtTensorBuffer tensor_buffer;
    EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                  environment, tensor_buffer_type, &kInput0TensorType,
                  tensor_buffer_size, &tensor_buffer),
              kLiteRtStatusOk);
    input_tensor_buffers.push_back(tensor_buffer);
  }

  LiteRtParamIndex num_outputs;
  ASSERT_EQ(LiteRtGetNumSubgraphOutputs(subgraph, &num_outputs),
            kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> output_tensor_buffers;
  output_tensor_buffers.reserve(num_outputs);
  for (auto i = 0; i < num_outputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    ASSERT_EQ(LiteRtGetCompiledModelOutputBufferRequirements(
                  compiled_model, /*signature_index=*/0, i,
                  &tensor_buffer_requirements),
              kLiteRtStatusOk);
    LiteRtTensorBufferType tensor_buffer_type;
    EXPECT_EQ(
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
            tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type),
        kLiteRtStatusOk);
    size_t tensor_buffer_size;
    EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                  tensor_buffer_requirements, &tensor_buffer_size),
              kLiteRtStatusOk);
    LiteRtTensorBuffer tensor_buffer;
    EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                  environment, tensor_buffer_type, &kInput0TensorType,
                  tensor_buffer_size, &tensor_buffer),
              kLiteRtStatusOk);
    output_tensor_buffers.push_back(tensor_buffer);
  }

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
    void* host_mem_addr;

    ASSERT_EQ(LiteRtLockTensorBuffer(input_tensor_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_tensor_buffers[0]),
              kLiteRtStatusOk);

    ASSERT_EQ(LiteRtLockTensorBuffer(input_tensor_buffers[1], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_tensor_buffers[1]),
              kLiteRtStatusOk);
  }

  ASSERT_EQ(LiteRtRunCompiledModel(
                compiled_model, /*signature_index=*/0,
                input_tensor_buffers.size(), input_tensor_buffers.data(),
                output_tensor_buffers.size(), output_tensor_buffers.data()),
            kLiteRtStatusOk);

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_tensor_buffers[0]),
              kLiteRtStatusOk);
  }

  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);

  for (auto tensor_buffer : input_tensor_buffers) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
  }
  for (auto tensor_buffer : output_tensor_buffers) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
  }
}

}  // namespace
}  // namespace litert
