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

#include <gtest/gtest.h>
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_dispatch_delegate.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "tflite/c/c_api_types.h"
#include "tflite/interpreter.h"

class LiteRtDispatchDeviceContextT {};

namespace litert {
namespace {

int DeviceContextCreateCount = 0;
int DeviceContextDestroyCount = 0;

LiteRtStatus Initialize(const LiteRtRuntimeContext* runtime_context,
                        LiteRtEnvironment environment, LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "DeviceContextTest";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  *build_id = "DeviceContextTest";
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(const LiteRtRuntimeContext* runtime_context,
                                 LiteRtOptions options,
                                 LiteRtDispatchDeviceContext* device_context) {
  ++DeviceContextCreateCount;
  *device_context = new LiteRtDispatchDeviceContextT();
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  ++DeviceContextDestroyCount;
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtDispatchInterface DeviceContextTestInterface = {
    /*.initialize=*/Initialize,
    /*.get_vendor_id=*/GetVendorId,
    /*.get_build_id=*/GetBuildId,
    /*.get_capabilities=*/GetCapabilities,
    /*.device_context_create=*/DeviceContextCreate,
    /*.device_context_destroy=*/DeviceContextDestroy,
    /*.get_input_requirements=*/nullptr,
    /*.get_output_requirements=*/nullptr,
    /*.register_tensor_buffer=*/nullptr,
    /*.unregister_tensor_buffer=*/nullptr,
    /*.invocation_context_create=*/nullptr,
    /*.invocation_context_destroy=*/nullptr,
    /*.invocation_context_set_scheduling_info=*/nullptr,
    /*.attach_input=*/nullptr,
    /*.attach_output=*/nullptr,
    /*.detach_input=*/nullptr,
    /*.detach_output=*/nullptr,
    /*.invoke=*/nullptr,
    /*.start_metrics_collection=*/nullptr,
    /*.stop_metrics_collection=*/nullptr,
    /*.get_num_metrics=*/nullptr,
    /*.get_metric=*/nullptr,
    /*.destroy_metrics=*/nullptr,
    /*.check_runtime_compatibility=*/CheckRuntimeCompatibility,
    /*.invocation_context_set_options=*/nullptr,
};

LiteRtStatus DummyCreateCustomTensorBuffer(
    LiteRtGpuDeviceId device_id, LiteRtGpuQueueId queue_id,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, size_t packed_bytes,
    HwMemoryInfoPtr* hw_memory_info) {
  static HwMemoryInfo info = {};
  *hw_memory_info = &info;
  return kLiteRtStatusOk;
}

LiteRtStatus DummyDestroyCustomTensorBuffer(HwMemoryInfoPtr hw_memory_info) {
  return kLiteRtStatusOk;
}

LiteRtCustomTensorBufferHandlersDef TestTensorBufferHandlers = {
    /*.abi_header=*/
    {
        /*.struct_size=*/sizeof(LiteRtCustomTensorBufferHandlersDef),
        /*.major_version=*/1,
        /*.minor_version=*/0,
        /*.reserved=*/0,
    },
    /*.create_func=*/DummyCreateCustomTensorBuffer,
    /*.destroy_func=*/DummyDestroyCustomTensorBuffer,
    /*.lock_func=*/nullptr,
    /*.unlock_func=*/nullptr,
    /*.clear_func=*/nullptr,
    /*.import_func=*/nullptr,
    /*.device_tag=*/kLiteRtEnvOptionTagDispatchLibraryDir,
    /*.queue_tag=*/kLiteRtEnvOptionTagDispatchLibraryDir,
    /*.num_supported_buffer_types=*/1,
    /*.supported_buffer_types=*/{kLiteRtTensorBufferTypeUserCustomBuffer},
};

LiteRtDispatchApi DeviceContextTestApi = {
    /*.abi_header=*/
    {
        /*.struct_size=*/sizeof(LiteRtDispatchApi),
        /*.major_version=*/1,
        /*.minor_version=*/0,
        /*.reserved=*/0,
    },
    /*.version=*/
    {/*.major=*/LITERT_API_VERSION_MAJOR,
     /*.minor=*/LITERT_API_VERSION_MINOR,
     /*.patch=*/LITERT_API_VERSION_PATCH},
    /*.interface=*/&DeviceContextTestInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
    /*.tensor_buffer_handlers_def=*/&TestTensorBufferHandlers,
};

LiteRtStatus GetDeviceContextTestApi(LiteRtDispatchApi* api) {
  *api = DeviceContextTestApi;
  return kLiteRtStatusOk;
}

class StaticLinkedDispatchApiScope {
 public:
  explicit StaticLinkedDispatchApiScope(
      LiteRtStatus (*get_api)(LiteRtDispatchApi*))
      : previous_get_api_(LiteRtStaticLinkedDispatchGetApi) {
    LiteRtStaticLinkedDispatchGetApi = get_api;
  }

  ~StaticLinkedDispatchApiScope() {
    LiteRtStaticLinkedDispatchGetApi = previous_get_api_;
  }

 private:
  LiteRtStatus (*previous_get_api_)(LiteRtDispatchApi*);
};

TEST(DispatchDelegateDeviceContextTest,
     MultiSignatureModelUsesSingleDeviceContext) {
  // This test has its own binary because the dispatch API is cached
  // process-wide after initialization.
  StaticLinkedDispatchApiScope static_dispatch_api(GetDeviceContextTestApi);
  DeviceContextCreateCount = 0;
  DeviceContextDestroyCount = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto c_options,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));

  {
    // The dispatch delegate must be declared before the TFL interpreter so it
    // outlives delegate kernels owned by the interpreter.
    DispatchDelegatePtr dispatch_delegate = {nullptr, nullptr};

    const std::string multi_signature_model_path =
        litert::testing::GetTfliteFilePath("testdata/multi_signatures.bin");
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto flatbuffer, litert::internal::FlatbufferWrapper::CreateFromTflFile(
                             multi_signature_model_path));
    LITERT_ASSERT_OK_AND_ASSIGN(
        litert::testing::TflRuntime::Ptr runtime,
        litert::testing::TflRuntime::CreateFromFlatBuffer(
            std::move(flatbuffer)));
    tflite::Interpreter& interpreter = runtime->Interpreter();

    auto signature_defs = interpreter.signature_keys();
    ASSERT_EQ(signature_defs.size(), 2);

    dispatch_delegate = CreateDispatchDelegatePtr(env.Get(), c_options.get());
    ASSERT_EQ(interpreter.ModifyGraphWithDelegate(dispatch_delegate.get()),
              kTfLiteOk);

    EXPECT_EQ(DeviceContextCreateCount, 1);
    EXPECT_EQ(DeviceContextDestroyCount, 0);
  }

  EXPECT_EQ(DeviceContextCreateCount, 1);
  EXPECT_EQ(DeviceContextDestroyCount, 1);
}

TEST(DispatchDelegateDeviceContextTest,
     MultipleEnvironmentsRegisterTensorBufferHandlers) {
  StaticLinkedDispatchApiScope static_dispatch_api(GetDeviceContextTestApi);

  LiteRtLayout layout = {};
  layout.rank = 1;
  layout.dimensions[0] = 16;

  LiteRtRankedTensorType tensor_type = {};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout = layout;

  LITERT_ASSERT_OK_AND_ASSIGN(auto env1, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options1, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto c_options1,
      internal::LiteRtOptionsPtrBuilder::Build(options1, env1.GetHolder()));

  ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env1.Get(),
                                     c_options1.get()),
            kLiteRtStatusOk);

  LiteRtTensorBuffer buffer1 = nullptr;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                env1.Get(), kLiteRtTensorBufferTypeUserCustomBuffer,
                &tensor_type, /*buffer_size=*/64, &buffer1),
            kLiteRtStatusOk);
  EXPECT_NE(buffer1, nullptr);
  LiteRtDestroyTensorBuffer(buffer1);

  // Second environment initialization: verify tensor buffer handlers are also
  // registered in the new environment despite process-global dispatch API
  // being already loaded.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env2, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options2, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto c_options2,
      internal::LiteRtOptionsPtrBuilder::Build(options2, env2.GetHolder()));

  ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env2.Get(),
                                     c_options2.get()),
            kLiteRtStatusOk);

  LiteRtTensorBuffer buffer2 = nullptr;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                env2.Get(), kLiteRtTensorBufferTypeUserCustomBuffer,
                &tensor_type, /*buffer_size=*/64, &buffer2),
            kLiteRtStatusOk);
  EXPECT_NE(buffer2, nullptr);
  LiteRtDestroyTensorBuffer(buffer2);
}

}  // namespace
}  // namespace litert
