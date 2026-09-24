// Copyright 2024 Google LLC.
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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Verifies that the default graph I/O tensor mem type is raw. This lives in its
// own test binary because LiteRtDispatchInitialize initializes the QNN backend
// once per process (IsTheApiInitialized), so a binary can only exercise one mem
// type. The memhandle (FastRPC/DMA-BUF) path is covered in
// dispatch_api_qualcomm_test.cc.

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/core/filesystem.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/qualcomm/core/utils/test_utils.h"

namespace {

using ::litert::Environment;
using ::litert::Options;
using ::testing::Pointwise;
static constexpr const float kTol = 5e-2;

litert::Expected<Environment> CreateDefaultEnvironment() {
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          qnn::GetTestDispatchLibraryDir(),
      },
  };
  return litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
}

litert::Expected<Options> CreateOptionsWithRawMemType() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  LITERT_ASSIGN_OR_RETURN(auto qualcomm_options,
                          litert::qualcomm::QualcommOptions::Create());
  qualcomm_options.SetGraphIOTensorMemType(
      litert::qualcomm::QualcommOptions::GraphIOTensorMemType::kRaw);

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_RETURN_IF_ERROR(qualcomm_options.GetOpaqueOptionsData(
      &identifier, &payload, &payload_deleter));
  LITERT_ASSIGN_OR_RETURN(
      auto opaque_options,
      litert::OpaqueOptions::Create(identifier, payload, payload_deleter));
  LITERT_RETURN_IF_ERROR(options.AddOpaqueOptions(std::move(opaque_options)));
  return options;
}

TEST(Qualcomm, DispatchApiWithDefaultRawMemType) {
#if !defined(__ANDROID__)
  GTEST_SKIP()
      << "This test is specific to Android devices with a Qualcomm NPU";
#else
  if (!::qnn::IsTestHtpBackend()) {
    GTEST_SKIP() << "Skipping test because targeted backend is not supported";
  }
#endif

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateOptionsWithRawMemType());
  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_opts,
                              litert::internal::LiteRtOptionsPtrBuilder::Build(
                                  options, env.GetHolder()));

  ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(),
                                     env.GetHolder().handle, litert_opts.get()),
            kLiteRtStatusOk);

  LiteRtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LiteRtDispatchDeviceContextCreate(
                LrtGetRuntimeContext(), litert_opts.get(), &device_context),
            kLiteRtStatusOk);

  auto model_file_name =
      litert::testing::GetTestFilePath(kQualcommModelFileName);
  auto model = litert::internal::LoadBinaryFile(model_file_name);
  EXPECT_TRUE(model) << model.Error();

  LiteRtMemBuffer exec_bytecode_buffer = {/*.fd=*/-1,
                                          /*.base_addr=*/model->Data(),
                                          /*.offset=*/0,
                                          /*.size=*/model->Size()};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_EQ(LiteRtDispatchInvocationContextCreate(
                LrtGetRuntimeContext(), device_context,
                kLiteRtDispatchExecutableTypeMlModel, &exec_bytecode_buffer,
                /*function_name=*/"simple",
                /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Determine tensor buffer requirements. The default (raw) mem type must
  // advertise host memory.
  // ///////////////////////////////////////////////////////////////////////////

  int num_tensor_buffer_types;
  LiteRtTensorBufferRequirements input_0_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/0, &kInput0TensorType,
                &input_0_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_0_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType input_0_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_0_tensor_buffer_requirements, /*type_index=*/0,
                &input_0_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_0_tensor_buffer_type, kLiteRtTensorBufferTypeHostMemory);
  size_t input_0_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_0_tensor_buffer_requirements, &input_0_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_0_tensor_buffer_size, sizeof(kTestInput0Tensor));

  LiteRtTensorBufferRequirements input_1_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/1, &kInput1TensorType,
                &input_1_tensor_buffer_requirements),
            kLiteRtStatusOk);
  LiteRtTensorBufferType input_1_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_1_tensor_buffer_requirements, /*type_index=*/0,
                &input_1_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_1_tensor_buffer_type, kLiteRtTensorBufferTypeHostMemory);
  size_t input_1_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_1_tensor_buffer_requirements, &input_1_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_1_tensor_buffer_size, sizeof(kTestInput1Tensor));

  LiteRtTensorBufferRequirements output_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetOutputRequirements(
                invocation_context, /*output_index=*/0, &kOutputTensorType,
                &output_tensor_buffer_requirements),
            kLiteRtStatusOk);
  LiteRtTensorBufferType output_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                output_tensor_buffer_requirements, /*type_index=*/0,
                &output_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(output_tensor_buffer_type, kLiteRtTensorBufferTypeHostMemory);
  size_t output_tensor_buffer_size;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                output_tensor_buffer_requirements, &output_tensor_buffer_size),
            kLiteRtStatusOk);
  EXPECT_GE(output_tensor_buffer_size, sizeof(kTestOutputTensor));

  // ///////////////////////////////////////////////////////////////////////////
  // Allocate, register, and attach host memory tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtTensorBuffer input_0_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, input_0_tensor_buffer_type,
                &kInput0TensorType, input_0_tensor_buffer_size,
                &input_0_tensor_buffer),
            kLiteRtStatusOk);
  LiteRtTensorBuffer input_1_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, input_1_tensor_buffer_type,
                &kInput1TensorType, input_1_tensor_buffer_size,
                &input_1_tensor_buffer),
            kLiteRtStatusOk);
  LiteRtTensorBuffer output_tensor_buffer;
  EXPECT_EQ(
      LiteRtCreateManagedTensorBuffer(
          env.GetHolder().handle, output_tensor_buffer_type, &kOutputTensorType,
          output_tensor_buffer_size, &output_tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferHandle input_1_handle;
  EXPECT_EQ(LiteRtDispatchRegisterTensorBuffer(
                device_context, input_1_tensor_buffer, &input_1_handle),
            kLiteRtStatusOk);
  LiteRtTensorBufferHandle input_0_handle;
  EXPECT_EQ(LiteRtDispatchRegisterTensorBuffer(
                device_context, input_0_tensor_buffer, &input_0_handle),
            kLiteRtStatusOk);
  LiteRtTensorBufferHandle output_handle;
  EXPECT_EQ(LiteRtDispatchRegisterTensorBuffer(
                device_context, output_tensor_buffer, &output_handle),
            kLiteRtStatusOk);

  EXPECT_EQ(LiteRtDispatchAttachInput(invocation_context,
                                      /*graph_input_index=*/0, input_0_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchAttachInput(invocation_context,
                                      /*graph_input_index=*/1, input_1_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchAttachOutput(invocation_context,
                                       /*graph_output_index=*/0, output_handle),
            kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Fill inputs, execute, and verify the output.
  // ///////////////////////////////////////////////////////////////////////////

  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(input_0_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_0_tensor_buffer), kLiteRtStatusOk);

    ASSERT_EQ(LiteRtLockTensorBuffer(input_1_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_1_tensor_buffer), kLiteRtStatusOk);
  }

  EXPECT_EQ(LiteRtDispatchInvoke(invocation_context), kLiteRtStatusOk);

  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    EXPECT_THAT(output, Pointwise(testing::FloatNear(kTol), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_tensor_buffer), kLiteRtStatusOk);
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Clean up resources.
  // ///////////////////////////////////////////////////////////////////////////

  EXPECT_EQ(LiteRtDispatchDetachInput(invocation_context,
                                      /*graph_input_index=*/0, input_0_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchDetachInput(invocation_context,
                                      /*graph_input_index=*/1, input_1_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchDetachOutput(invocation_context,
                                       /*graph_output_index=*/0, output_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchUnregisterTensorBuffer(device_context, output_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtDispatchUnregisterTensorBuffer(device_context, input_1_handle),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtDispatchUnregisterTensorBuffer(device_context, input_0_handle),
      kLiteRtStatusOk);
  LiteRtDestroyTensorBuffer(output_tensor_buffer);
  LiteRtDestroyTensorBuffer(input_1_tensor_buffer);
  LiteRtDestroyTensorBuffer(input_0_tensor_buffer);
  EXPECT_EQ(LiteRtDispatchInvocationContextDestroy(invocation_context),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchDeviceContextDestroy(device_context),
            kLiteRtStatusOk);
}

}  // namespace
