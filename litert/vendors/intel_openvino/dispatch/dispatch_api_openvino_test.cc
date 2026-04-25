// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_dispatch.h"

constexpr absl::string_view kDispatchLibraryDir =
    "litert/vendors/intel_openvino/dispatch";

litert::Expected<litert::Environment> CreateDefaultEnvironment() {
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  return litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
}

TEST(OpenVino, DispatchApi) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env.Get(),
                                     options.Get()),
            kLiteRtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LiteRtDispatchGetVendorId(&vendor_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "vendor_id: " << vendor_id;

  const char* build_id;
  EXPECT_EQ(LiteRtDispatchGetBuildId(&build_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "build_id " << build_id;

  LiteRtApiVersion api_version;
  EXPECT_EQ(LiteRtDispatchGetApiVersion(&api_version), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "api_version: " << api_version.major << "."
                 << api_version.minor << "." << api_version.patch;

  int capabilities;
  EXPECT_EQ(LiteRtDispatchGetCapabilities(&capabilities), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "capabilities: " << capabilities;

  LiteRtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                              options.Get(), &device_context),
            kLiteRtStatusOk);
  EXPECT_NE(device_context, nullptr);

  // Clean up.
  EXPECT_EQ(LiteRtDispatchDeviceContextDestroy(device_context),
            kLiteRtStatusOk);
}

// ===== Negative tests for dispatch invocation context =====

// Null bytecode buffer should fail invocation context creation.
TEST(OpenVino, InvocationContextNullBytecodeBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env.Get(),
                                     options.Get()),
            kLiteRtStatusOk);

  LiteRtDispatchDeviceContext device_context = nullptr;
  ASSERT_EQ(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                              options.Get(), &device_context),
            kLiteRtStatusOk);

  LiteRtDispatchInvocationContext invocation_context = nullptr;
  // Pass nullptr for exec_bytecode_buffer.
  EXPECT_NE(LiteRtDispatchInvocationContextCreate(
                LrtGetRuntimeContext(), device_context,
                kLiteRtDispatchExecutableTypeMlModel,
                /*exec_bytecode_buffer=*/nullptr,
                /*function_name=*/nullptr,
                /*num_inputs=*/1, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);

  EXPECT_EQ(LiteRtDispatchDeviceContextDestroy(device_context),
            kLiteRtStatusOk);
}

// Zero-size bytecode buffer should fail invocation context creation.
TEST(OpenVino, InvocationContextEmptyBytecodeBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env.Get(),
                                     options.Get()),
            kLiteRtStatusOk);

  LiteRtDispatchDeviceContext device_context = nullptr;
  ASSERT_EQ(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                              options.Get(), &device_context),
            kLiteRtStatusOk);

  char dummy = 0;
  LiteRtMemBuffer empty_buffer = {/*.fd=*/-1,
                                  /*.base_addr=*/&dummy,
                                  /*.offset=*/0,
                                  /*.size=*/0};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_NE(LiteRtDispatchInvocationContextCreate(
                LrtGetRuntimeContext(), device_context,
                kLiteRtDispatchExecutableTypeMlModel, &empty_buffer,
                /*function_name=*/nullptr,
                /*num_inputs=*/1, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);

  EXPECT_EQ(LiteRtDispatchDeviceContextDestroy(device_context),
            kLiteRtStatusOk);
}

// Offset exceeding size should fail invocation context creation.
TEST(OpenVino, InvocationContextOffsetExceedsSize) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env.Get(),
                                     options.Get()),
            kLiteRtStatusOk);

  LiteRtDispatchDeviceContext device_context = nullptr;
  ASSERT_EQ(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                              options.Get(), &device_context),
            kLiteRtStatusOk);

  char dummy[16] = {};
  LiteRtMemBuffer bad_offset_buffer = {/*.fd=*/-1,
                                       /*.base_addr=*/dummy,
                                       /*.offset=*/100,
                                       /*.size=*/16};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_NE(LiteRtDispatchInvocationContextCreate(
                LrtGetRuntimeContext(), device_context,
                kLiteRtDispatchExecutableTypeMlModel, &bad_offset_buffer,
                /*function_name=*/nullptr,
                /*num_inputs=*/1, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);

  EXPECT_EQ(LiteRtDispatchDeviceContextDestroy(device_context),
            kLiteRtStatusOk);
}

// NOTE: AttachInput/AttachOutput index validation tests are excluded here
// because they require a valid compiled model blob (kOpenvinoModelBlobFileName)
// to create an invocation context. They should be run when the blob is
// available.
