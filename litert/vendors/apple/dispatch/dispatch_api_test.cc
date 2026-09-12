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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

using ::litert::testing::SimpleBuffer;

class AppleMlxDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Compile a simple model to get bytecode for testing dispatch.
    plugin_ = CreatePlugin(LrtGetCompilerContext());
    ExtendedModel model =
        testing::LoadTestFileModel("fully_connected_cst.tflite");

    LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(
        plugin_.get(), "apple_mlx", model.Get(), &compiled_result_));

    const void* byte_code_ptr;
    LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
        compiled_result_, 0, &byte_code_ptr, &bytecode_size_));

    // Copy bytecode to local buffer
    bytecode_.assign(reinterpret_cast<const char*>(byte_code_ptr),
                     bytecode_size_);
  }

  void TearDown() override { LiteRtDestroyCompiledResult(compiled_result_); }

  PluginPtr plugin_;
  LiteRtCompiledResult compiled_result_;
  std::string bytecode_;
  size_t bytecode_size_;
};

TEST_F(AppleMlxDispatchTest, GetVendorId) {
  const char* vendor_id;
  LITERT_ASSERT_OK(LiteRtDispatchGetVendorId(&vendor_id));
  EXPECT_STREQ(vendor_id, "Apple");
}

TEST_F(AppleMlxDispatchTest, GetBuildId) {
  const char* build_id;
  LITERT_ASSERT_OK(LiteRtDispatchGetBuildId(&build_id));
  EXPECT_STREQ(build_id, "Apple MLX Dispatch 1.0");
}

TEST_F(AppleMlxDispatchTest, DeviceContextLifecycle) {
  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                                     /*options=*/nullptr,
                                                     &device_context));
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextDestroy(device_context));
}

TEST_F(AppleMlxDispatchTest, InvocationContextLifecycle) {
  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                                     /*options=*/nullptr,
                                                     &device_context));
  absl::Cleanup device_cleanup = [&device_context] {
    LiteRtDispatchDeviceContextDestroy(device_context);
  };

  LiteRtMemBuffer exec_bytecode_buffer = {
      /*.fd=*/-1,
      /*.base_addr=*/bytecode_.data(),
      /*.offset=*/0,
      /*.size=*/bytecode_size_,
  };

  LiteRtDispatchInvocationContext invocation_context;
  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreate(
      LrtGetRuntimeContext(), device_context,
      kLiteRtDispatchExecutableTypeMlModel, &exec_bytecode_buffer,
      "mlx_partition_0", 1, 1, &invocation_context));

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextDestroy(invocation_context));
}

TEST_F(AppleMlxDispatchTest, RegisterTensorBuffer) {
  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                                     /*options=*/nullptr,
                                                     &device_context));
  absl::Cleanup device_cleanup = [&device_context] {
    LiteRtDispatchDeviceContextDestroy(device_context);
  };

  // Create a dummy host memory tensor buffer
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      SimpleBuffer::Create<float>({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer, buffer.SpawnTensorBuffer());

  LiteRtTensorBufferHandle handle;
  LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
      device_context, tensor_buffer.Get(), &handle));
  EXPECT_GT(handle, 0);

  LITERT_ASSERT_OK(
      LiteRtDispatchUnregisterTensorBuffer(device_context, handle));
}

TEST_F(AppleMlxDispatchTest, InvokeStub) {
  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(LrtGetRuntimeContext(),
                                                     /*options=*/nullptr,
                                                     &device_context));
  absl::Cleanup device_cleanup = [&device_context] {
    LiteRtDispatchDeviceContextDestroy(device_context);
  };

  LiteRtMemBuffer exec_bytecode_buffer = {
      /*.fd=*/-1,
      /*.base_addr=*/bytecode_.data(),
      /*.offset=*/0,
      /*.size=*/bytecode_size_,
  };

  LiteRtDispatchInvocationContext invocation_context;
  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreate(
      LrtGetRuntimeContext(), device_context,
      kLiteRtDispatchExecutableTypeMlModel, &exec_bytecode_buffer,
      "mlx_partition_0", 1, 1, &invocation_context));
  absl::Cleanup invocation_cleanup = [&invocation_context] {
    LiteRtDispatchInvocationContextDestroy(invocation_context);
  };

  // Create input/output buffers
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input,
      SimpleBuffer::Create<float>({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_tb, input.SpawnTensorBuffer());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output, SimpleBuffer::Create<float>({1, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tb, output.SpawnTensorBuffer());

  LiteRtTensorBufferHandle input_handle;
  LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
      device_context, input_tb.Get(), &input_handle));
  LiteRtTensorBufferHandle output_handle;
  LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
      device_context, output_tb.Get(), &output_handle));

  LITERT_ASSERT_OK(
      LiteRtDispatchAttachInput(invocation_context, 0, input_handle));
  LITERT_ASSERT_OK(
      LiteRtDispatchAttachOutput(invocation_context, 0, output_handle));

  // This will call the stub on Linux, which should just log and return OK.
  LITERT_ASSERT_OK(LiteRtDispatchInvoke(invocation_context));
}

}  // namespace
}  // namespace litert
