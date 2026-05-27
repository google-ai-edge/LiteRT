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

#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
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

LiteRtDispatchApi DeviceContextTestApi = {
    /*.version=*/{/*.major=*/LITERT_API_VERSION_MAJOR,
                  /*.minor=*/LITERT_API_VERSION_MINOR,
                  /*.patch=*/LITERT_API_VERSION_PATCH},
    /*.interface=*/&DeviceContextTestInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
    /*.tensor_buffer_handlers_def=*/nullptr,
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

}  // namespace
}  // namespace litert
