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

#include <gtest/gtest.h>
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace litert {
namespace {

LiteRtStatus DummyCreate(LiteRtGpuDeviceId device_id, LiteRtGpuQueueId queue_id,
                         const LiteRtRankedTensorType* tensor_type,
                         LiteRtTensorBufferType buffer_type, size_t bytes,
                         size_t packed_bytes, HwMemoryInfoPtr* hw_memory_info) {
  return kLiteRtStatusOk;
}

LiteRtStatus DummyDestroy(HwMemoryInfoPtr hw_memory_info) {
  return kLiteRtStatusOk;
}

LiteRtStatus DummyLock(HwMemoryInfoPtr hw_memory_info,
                       LiteRtTensorBufferLockMode mode,
                       void** host_memory_ptr) {
  return kLiteRtStatusOk;
}

LiteRtStatus DummyUnlock(HwMemoryInfoPtr hw_memory_info) {
  return kLiteRtStatusOk;
}

LiteRtCustomTensorBufferHandlersDef CustomBufferHandlersDef = {
    /*.abi_header=*/
    {
        /*.struct_size=*/sizeof(LiteRtCustomTensorBufferHandlersDef),
        /*.major_version=*/1,
        /*.minor_version=*/0,
        /*.reserved=*/0,
    },
    /*.create_func=*/DummyCreate,
    /*.destroy_func=*/DummyDestroy,
    /*.lock_func=*/DummyLock,
    /*.unlock_func=*/DummyUnlock,
    /*.clear_func=*/nullptr,
    /*.import_func=*/nullptr,
    /*.device_tag=*/kLiteRtEnvOptionTagNull,
    /*.queue_tag=*/kLiteRtEnvOptionTagNull,
    /*.num_supported_buffer_types=*/1,
    /*.supported_buffer_types=*/{kLiteRtTensorBufferTypeUserCustomBuffer},
};

LiteRtStatus Initialize(const LiteRtRuntimeContext* runtime_context,
                        LiteRtEnvironment environment, LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "CustomBufferTest";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  *build_id = "CustomBufferTest";
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtDispatchInterface CustomBufferTestInterface = {
    /*.initialize=*/Initialize,
    /*.get_vendor_id=*/GetVendorId,
    /*.get_build_id=*/GetBuildId,
    /*.get_capabilities=*/GetCapabilities,
    /*.device_context_create=*/nullptr,
    /*.device_context_destroy=*/nullptr,
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

LiteRtDispatchApi CustomBufferTestApi = {
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
    /*.interface=*/&CustomBufferTestInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
    /*.tensor_buffer_handlers_def=*/&CustomBufferHandlersDef,
};

LiteRtStatus GetCustomBufferTestApi(LiteRtDispatchApi* api) {
  *api = CustomBufferTestApi;
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

TEST(DispatchDelegateCustomBufferTest,
     RegistersCustomBufferHandlersOnEveryEnvironment) {
  // This test has its own binary because the dispatch API is cached
  // process-wide after initialization.
  StaticLinkedDispatchApiScope static_dispatch_api(GetCustomBufferTestApi);

  // 1. First environment: initialize dispatch and verify custom handlers are
  // registered.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto env1, Environment::Create({}));
    ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env1.Get(),
                                       /*options=*/nullptr),
              kLiteRtStatusOk);

    auto handlers_or = env1.Get()->GetTensorBufferRegistry().GetCustomHandlers(
        kLiteRtTensorBufferTypeUserCustomBuffer);
    EXPECT_TRUE(handlers_or.HasValue());
  }

  // 2. Second environment: env1 was destroyed, so its TensorBufferRegistry was
  // freed. When LiteRtDispatchInitialize is called on env2, it MUST register
  // the handlers into env2's registry even though IsTheApiInitialized is
  // already true.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto env2, Environment::Create({}));
    ASSERT_EQ(LiteRtDispatchInitialize(LrtGetRuntimeContext(), env2.Get(),
                                       /*options=*/nullptr),
              kLiteRtStatusOk);

    auto handlers_or = env2.Get()->GetTensorBufferRegistry().GetCustomHandlers(
        kLiteRtTensorBufferTypeUserCustomBuffer);
    EXPECT_TRUE(handlers_or.HasValue());
  }
}

}  // namespace
}  // namespace litert
