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

#include <any>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/core/filesystem.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace {

using ::litert::Environment;
using ::litert::Options;
using ::testing::Pointwise;
static constexpr const float kTol = 5e-2;

constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

litert::Expected<Environment> CreateDefaultEnvironment() {
  const std::vector<litert::Environment::Option> environment_options = {
      Environment::Option{
          Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  return Environment::Create(absl::MakeConstSpan(environment_options));
}

TEST(Qualcomm, DispatchApiWithFastRpc) {
#if !defined(__ANDROID__)
  GTEST_SKIP()
      << "This test is specific to Android devices with a Qualcomm NPU";
#endif

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(env.GetHolder().handle, options.Get()),
            kLiteRtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LiteRtDispatchGetVendorId(&vendor_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "vendor_id: " << vendor_id;

  const char* build_id;
  EXPECT_EQ(LiteRtDispatchGetBuildId(&build_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "build_id: " << build_id;

  LiteRtApiVersion api_version;
  EXPECT_EQ(LiteRtDispatchGetApiVersion(&api_version), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "api_version: " << api_version.major << "."
                 << api_version.minor << "." << api_version.patch;

  int capabilities;
  EXPECT_EQ(LiteRtDispatchGetCapabilities(&capabilities), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "capabilities: " << capabilities;

  LiteRtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LiteRtDispatchDeviceContextCreate(&device_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "device_context: " << device_context;

  auto model_file_name =
      litert::testing::GetTestFilePath(kQualcommModelFileName);
  auto model = litert::internal::LoadBinaryFile(model_file_name);
  EXPECT_TRUE(model) << model.Error();
  ABSL_LOG(INFO) << "Loaded model " << model_file_name << ", " << model->Size()
                 << " bytes";

  // ///////////////////////////////////////////////////////////////////////////
  // Set up an invocation context for a given model.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtMemBuffer exec_bytecode_buffer = {/*.fd=*/-1,
                                          /*.base_addr=*/model->Data(),
                                          /*.offset=*/0,
                                          /*.size=*/model->Size()};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_EQ(LiteRtDispatchInvocationContextCreate(
                device_context, kLiteRtDispatchExecutableTypeMlModel,
                &exec_bytecode_buffer, /*function_name=*/"simple",
                /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "Invocation context: " << invocation_context;

  // ///////////////////////////////////////////////////////////////////////////
  // Determine tensor buffer requirements.
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
  EXPECT_EQ(input_0_tensor_buffer_type, kLiteRtTensorBufferTypeFastRpc);
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
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_1_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType input_1_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_1_tensor_buffer_requirements, /*type_index=*/0,
                &input_1_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_1_tensor_buffer_type, kLiteRtTensorBufferTypeFastRpc);
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
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                output_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType output_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                output_tensor_buffer_requirements, /*type_index=*/0,
                &output_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(output_tensor_buffer_type, kLiteRtTensorBufferTypeFastRpc);
  size_t output_tensor_buffer_size;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                output_tensor_buffer_requirements, &output_tensor_buffer_size),
            kLiteRtStatusOk);
  EXPECT_GE(output_tensor_buffer_size, sizeof(kTestOutputTensor));

  // ///////////////////////////////////////////////////////////////////////////
  // Allocate tensor buffers.
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

  // ///////////////////////////////////////////////////////////////////////////
  // Register tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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

  // ///////////////////////////////////////////////////////////////////////////
  // Attach tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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
  // Fill the input buffers with data.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
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

  // ///////////////////////////////////////////////////////////////////////////
  // Execute model.
  // ///////////////////////////////////////////////////////////////////////////

  ABSL_LOG(INFO) << "Invoking execution...";
  EXPECT_EQ(LiteRtDispatchInvoke(invocation_context), kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Check output for correctness.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
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

TEST(Qualcomm, DispatchApiWithDmaBuf) {
#if !defined(__ANDROID__)
  GTEST_SKIP()
      << "This test is specific to Android devices with a Qualcomm NPU";
#endif

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(env.GetHolder().handle, options.Get()),
            kLiteRtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LiteRtDispatchGetVendorId(&vendor_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "vendor_id: " << vendor_id;

  const char* build_id;
  EXPECT_EQ(LiteRtDispatchGetBuildId(&build_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "build_id: " << build_id;

  LiteRtApiVersion api_version;
  EXPECT_EQ(LiteRtDispatchGetApiVersion(&api_version), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "api_version: " << api_version.major << "."
                 << api_version.minor << "." << api_version.patch;

  int capabilities;
  EXPECT_EQ(LiteRtDispatchGetCapabilities(&capabilities), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "capabilities: " << capabilities;

  LiteRtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LiteRtDispatchDeviceContextCreate(&device_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "device_context: " << device_context;

  auto model_file_name =
      litert::testing::GetTestFilePath(kQualcommModelFileName);
  auto model = litert::internal::LoadBinaryFile(model_file_name);
  EXPECT_TRUE(model) << model.Error();
  ABSL_LOG(INFO) << "Loaded model " << model_file_name << ", " << model->Size()
                 << " bytes";

  // ///////////////////////////////////////////////////////////////////////////
  // Set up an invocation context for a given model.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtMemBuffer exec_bytecode_buffer = {/*.fd=*/-1,
                                          /*.base_addr=*/model->Data(),
                                          /*.offset=*/0,
                                          /*.size=*/model->Size()};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_EQ(LiteRtDispatchInvocationContextCreate(
                device_context, kLiteRtDispatchExecutableTypeMlModel,
                &exec_bytecode_buffer, /*function_name=*/"simple",
                /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "Invocation context: " << invocation_context;

  // ///////////////////////////////////////////////////////////////////////////
  // Determine tensor buffer requirements.
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
                input_0_tensor_buffer_requirements, /*type_index=*/1,
                &input_0_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_0_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
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
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_1_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType input_1_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_1_tensor_buffer_requirements, /*type_index=*/1,
                &input_1_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_1_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
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
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                output_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType output_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                output_tensor_buffer_requirements, /*type_index=*/1,
                &output_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(output_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
  size_t output_tensor_buffer_size;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                output_tensor_buffer_requirements, &output_tensor_buffer_size),
            kLiteRtStatusOk);
  EXPECT_GE(output_tensor_buffer_size, sizeof(kTestOutputTensor));

  // ///////////////////////////////////////////////////////////////////////////
  // Allocate tensor buffers.
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

  // ///////////////////////////////////////////////////////////////////////////
  // Register tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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

  // ///////////////////////////////////////////////////////////////////////////
  // Attach tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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
  // Fill the input buffers with data.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
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

  // ///////////////////////////////////////////////////////////////////////////
  // Execute model.
  // ///////////////////////////////////////////////////////////////////////////

  ABSL_LOG(INFO) << "Invoking execution...";
  EXPECT_EQ(LiteRtDispatchInvoke(invocation_context), kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Check output for correctness.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
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

TEST(Qualcomm, DispatchApiWithFastRpcInt16Model) {
#if !defined(__ANDROID__)
  GTEST_SKIP()
      << "This test is specific to Android devices with a Qualcomm NPU";
#endif
  // ///////////////////////////////////////////////////////////////////////////
  // Set up data for input and output.
  // ///////////////////////////////////////////////////////////////////////////

  const std::vector<std::int16_t> input_tensor_0 = {
      qnn::Quantize<int16_t>(kTestInput0Tensor_3[0], kInputScale,
                             kInputZeroPoint),
      qnn::Quantize<int16_t>(kTestInput0Tensor_3[1], kInputScale,
                             kInputZeroPoint)};
  const std::vector<std::int16_t> input_tensor_1 = {
      qnn::Quantize<int16_t>(kTestInput1Tensor_3[0], kInputScale,
                             kInputZeroPoint),
      qnn::Quantize<int16_t>(kTestInput1Tensor_3[1], kInputScale,
                             kInputZeroPoint)};
  const std::vector<std::int16_t> output_tensor_0 = {
      qnn::Quantize<int16_t>(kTestOutputTensor_3[0], kOutputScale,
                             kOutputZeroPoint),
      qnn::Quantize<int16_t>(kTestOutputTensor_3[1], kOutputScale,
                             kOutputZeroPoint)};

  const size_t input_tensor_0_bytes =
      input_tensor_0.size() * sizeof(decltype(input_tensor_0)::value_type);
  const size_t input_tensor_1_bytes =
      input_tensor_1.size() * sizeof(decltype(input_tensor_1)::value_type);
  const size_t output_tensor_0_bytes =
      output_tensor_0.size() * sizeof(decltype(output_tensor_0)::value_type);

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(env.GetHolder().handle, options.Get()),
            kLiteRtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LiteRtDispatchGetVendorId(&vendor_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "vendor_id: " << vendor_id;

  const char* build_id;
  EXPECT_EQ(LiteRtDispatchGetBuildId(&build_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "build_id: " << build_id;

  LiteRtApiVersion api_version;
  EXPECT_EQ(LiteRtDispatchGetApiVersion(&api_version), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "api_version: " << api_version.major << "."
                 << api_version.minor << "." << api_version.patch;

  int capabilities;
  EXPECT_EQ(LiteRtDispatchGetCapabilities(&capabilities), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "capabilities: " << capabilities;

  LiteRtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LiteRtDispatchDeviceContextCreate(&device_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "device_context: " << device_context;

  auto model_file_name =
      litert::testing::GetTestFilePath(kQualcommUInt16ModelFileName);
  auto model = litert::internal::LoadBinaryFile(model_file_name);
  EXPECT_TRUE(model) << model.Error();
  ABSL_LOG(INFO) << "Loaded model " << model_file_name << ", " << model->Size()
                 << " bytes";

  // ///////////////////////////////////////////////////////////////////////////
  // Set up an invocation context for a given model.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtMemBuffer exec_bytecode_buffer = {/*.fd=*/-1,
                                          /*.base_addr=*/model->Data(),
                                          /*.offset=*/0,
                                          /*.size=*/model->Size()};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_EQ(LiteRtDispatchInvocationContextCreate(
                device_context, kLiteRtDispatchExecutableTypeMlModel,
                &exec_bytecode_buffer, /*function_name=*/"qnn_partition_0",
                /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "Invocation context: " << invocation_context;

  // ///////////////////////////////////////////////////////////////////////////
  // Determine tensor buffer requirements.
  // ///////////////////////////////////////////////////////////////////////////

  int num_tensor_buffer_types;
  LiteRtTensorBufferRequirements input_0_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/0, &kInput0TensorType_3,
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
  EXPECT_EQ(input_0_tensor_buffer_type, kLiteRtTensorBufferTypeFastRpc);
  size_t input_0_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_0_tensor_buffer_requirements, &input_0_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_0_tensor_buffer_size, input_tensor_0_bytes);

  LiteRtTensorBufferRequirements input_1_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/1, &kInput1TensorType_3,
                &input_1_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_1_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType input_1_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_1_tensor_buffer_requirements, /*type_index=*/0,
                &input_1_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_1_tensor_buffer_type, kLiteRtTensorBufferTypeFastRpc);
  size_t input_1_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_1_tensor_buffer_requirements, &input_1_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_1_tensor_buffer_size, input_tensor_1_bytes);

  LiteRtTensorBufferRequirements output_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetOutputRequirements(
                invocation_context, /*output_index=*/0, &kOutputTensorType_3,
                &output_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                output_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType output_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                output_tensor_buffer_requirements, /*type_index=*/0,
                &output_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(output_tensor_buffer_type, kLiteRtTensorBufferTypeFastRpc);
  size_t output_tensor_buffer_size;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                output_tensor_buffer_requirements, &output_tensor_buffer_size),
            kLiteRtStatusOk);
  EXPECT_GE(output_tensor_buffer_size, output_tensor_0_bytes);

  // ///////////////////////////////////////////////////////////////////////////
  // Allocate tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtTensorBuffer input_0_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, input_0_tensor_buffer_type,
                &kInput0TensorType_3, input_0_tensor_buffer_size,
                &input_0_tensor_buffer),
            kLiteRtStatusOk);

  LiteRtTensorBuffer input_1_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, input_1_tensor_buffer_type,
                &kInput1TensorType_3, input_1_tensor_buffer_size,
                &input_1_tensor_buffer),
            kLiteRtStatusOk);

  LiteRtTensorBuffer output_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, output_tensor_buffer_type,
                &kOutputTensorType_3, output_tensor_buffer_size,
                &output_tensor_buffer),
            kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Register tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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

  // ///////////////////////////////////////////////////////////////////////////
  // Attach tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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
  // Fill the input buffers with data.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
    void* host_mem_addr;

    ASSERT_EQ(LiteRtLockTensorBuffer(input_0_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, input_tensor_0.data(), input_tensor_0_bytes);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_0_tensor_buffer), kLiteRtStatusOk);

    ASSERT_EQ(LiteRtLockTensorBuffer(input_1_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, input_tensor_1.data(), input_tensor_1_bytes);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_1_tensor_buffer), kLiteRtStatusOk);
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Execute model.
  // ///////////////////////////////////////////////////////////////////////////

  ABSL_LOG(INFO) << "Invoking execution...";
  EXPECT_EQ(LiteRtDispatchInvoke(invocation_context), kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Check output for correctness.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto int16_output =
        absl::MakeSpan(static_cast<const std::int16_t*>(host_mem_addr),
                       output_tensor_0.size());

    std::vector<float> dequant_output;
    for (const auto& data : int16_output) {
      dequant_output.emplace_back(
          qnn::Dequantize<std::int16_t>(data, kOutputScale, kOutputZeroPoint));
    }

    for (auto i = 0; i < dequant_output.size(); ++i) {
      ABSL_LOG(INFO) << dequant_output[i] << "\t" << kTestOutputTensor_3[i];
    }
    EXPECT_THAT(dequant_output,
                Pointwise(testing::FloatNear(kTol), kTestOutputTensor_3));
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

TEST(Qualcomm, DispatchApiWithDmaBufInt16Model) {
#if !defined(__ANDROID__)
  GTEST_SKIP()
      << "This test is specific to Android devices with a Qualcomm NPU";
#endif
  // ///////////////////////////////////////////////////////////////////////////
  // Set up data for input and output.
  // ///////////////////////////////////////////////////////////////////////////

  const std::vector<std::int16_t> input_tensor_0 = {
      qnn::Quantize<int16_t>(kTestInput0Tensor_3[0], kInputScale,
                             kInputZeroPoint),
      qnn::Quantize<int16_t>(kTestInput0Tensor_3[1], kInputScale,
                             kInputZeroPoint)};
  const std::vector<std::int16_t> input_tensor_1 = {
      qnn::Quantize<int16_t>(kTestInput1Tensor_3[0], kInputScale,
                             kInputZeroPoint),
      qnn::Quantize<int16_t>(kTestInput1Tensor_3[1], kInputScale,
                             kInputZeroPoint)};
  const std::vector<std::int16_t> output_tensor_0 = {
      qnn::Quantize<int16_t>(kTestOutputTensor_3[0], kOutputScale,
                             kOutputZeroPoint),
      qnn::Quantize<int16_t>(kTestOutputTensor_3[1], kOutputScale,
                             kOutputZeroPoint)};

  const size_t input_tensor_0_bytes =
      input_tensor_0.size() * sizeof(decltype(input_tensor_0)::value_type);
  const size_t input_tensor_1_bytes =
      input_tensor_1.size() * sizeof(decltype(input_tensor_1)::value_type);
  const size_t output_tensor_0_bytes =
      output_tensor_0.size() * sizeof(decltype(output_tensor_0)::value_type);

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(env.GetHolder().handle, options.Get()),
            kLiteRtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LiteRtDispatchGetVendorId(&vendor_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "vendor_id: " << vendor_id;

  const char* build_id;
  EXPECT_EQ(LiteRtDispatchGetBuildId(&build_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "build_id: " << build_id;

  LiteRtApiVersion api_version;
  EXPECT_EQ(LiteRtDispatchGetApiVersion(&api_version), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "api_version: " << api_version.major << "."
                 << api_version.minor << "." << api_version.patch;

  int capabilities;
  EXPECT_EQ(LiteRtDispatchGetCapabilities(&capabilities), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "capabilities: " << capabilities;

  LiteRtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LiteRtDispatchDeviceContextCreate(&device_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "device_context: " << device_context;

  auto model_file_name =
      litert::testing::GetTestFilePath(kQualcommUInt16ModelFileName);
  auto model = litert::internal::LoadBinaryFile(model_file_name);
  EXPECT_TRUE(model) << model.Error();
  ABSL_LOG(INFO) << "Loaded model " << model_file_name << ", " << model->Size()
                 << " bytes";

  // ///////////////////////////////////////////////////////////////////////////
  // Set up an invocation context for a given model.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtMemBuffer exec_bytecode_buffer = {/*.fd=*/-1,
                                          /*.base_addr=*/model->Data(),
                                          /*.offset=*/0,
                                          /*.size=*/model->Size()};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_EQ(LiteRtDispatchInvocationContextCreate(
                device_context, kLiteRtDispatchExecutableTypeMlModel,
                &exec_bytecode_buffer, /*function_name=*/"qnn_partition_0",
                /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);
  ABSL_LOG(INFO) << "Invocation context: " << invocation_context;

  // ///////////////////////////////////////////////////////////////////////////
  // Determine tensor buffer requirements.
  // ///////////////////////////////////////////////////////////////////////////

  int num_tensor_buffer_types;
  LiteRtTensorBufferRequirements input_0_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/0, &kInput0TensorType_3,
                &input_0_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_0_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType input_0_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_0_tensor_buffer_requirements, /*type_index=*/1,
                &input_0_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_0_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
  size_t input_0_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_0_tensor_buffer_requirements, &input_0_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_0_tensor_buffer_size, input_tensor_0_bytes);

  LiteRtTensorBufferRequirements input_1_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/1, &kInput1TensorType_3,
                &input_1_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_1_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType input_1_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_1_tensor_buffer_requirements, /*type_index=*/1,
                &input_1_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_1_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
  size_t input_1_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_1_tensor_buffer_requirements, &input_1_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_1_tensor_buffer_size, input_tensor_1_bytes);

  LiteRtTensorBufferRequirements output_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetOutputRequirements(
                invocation_context, /*output_index=*/0, &kOutputTensorType_3,
                &output_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                output_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LiteRtTensorBufferType output_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                output_tensor_buffer_requirements, /*type_index=*/1,
                &output_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(output_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
  size_t output_tensor_buffer_size;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                output_tensor_buffer_requirements, &output_tensor_buffer_size),
            kLiteRtStatusOk);
  EXPECT_GE(output_tensor_buffer_size, output_tensor_0_bytes);

  // ///////////////////////////////////////////////////////////////////////////
  // Allocate tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtTensorBuffer input_0_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, input_0_tensor_buffer_type,
                &kInput0TensorType_3, input_0_tensor_buffer_size,
                &input_0_tensor_buffer),
            kLiteRtStatusOk);

  LiteRtTensorBuffer input_1_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, input_1_tensor_buffer_type,
                &kInput1TensorType_3, input_1_tensor_buffer_size,
                &input_1_tensor_buffer),
            kLiteRtStatusOk);

  LiteRtTensorBuffer output_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, output_tensor_buffer_type,
                &kOutputTensorType_3, output_tensor_buffer_size,
                &output_tensor_buffer),
            kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Register tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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

  // ///////////////////////////////////////////////////////////////////////////
  // Attach tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

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
  // Fill the input buffers with data.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
    void* host_mem_addr;

    ASSERT_EQ(LiteRtLockTensorBuffer(input_0_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, input_tensor_0.data(), input_tensor_0_bytes);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_0_tensor_buffer), kLiteRtStatusOk);

    ASSERT_EQ(LiteRtLockTensorBuffer(input_1_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, input_tensor_1.data(), input_tensor_1_bytes);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_1_tensor_buffer), kLiteRtStatusOk);
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Execute model.
  // ///////////////////////////////////////////////////////////////////////////

  ABSL_LOG(INFO) << "Invoking execution...";
  EXPECT_EQ(LiteRtDispatchInvoke(invocation_context), kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Check output for correctness.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto int16_output =
        absl::MakeSpan(static_cast<const std::int16_t*>(host_mem_addr),
                       output_tensor_0.size());

    std::vector<float> dequant_output;
    for (const auto& data : int16_output) {
      dequant_output.emplace_back(
          qnn::Dequantize<std::int16_t>(data, kOutputScale, kOutputZeroPoint));
    }

    for (auto i = 0; i < dequant_output.size(); ++i) {
      ABSL_LOG(INFO) << dequant_output[i] << "\t" << kTestOutputTensor_3[i];
    }
    EXPECT_THAT(dequant_output,
                Pointwise(testing::FloatNear(kTol), kTestOutputTensor_3));
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
