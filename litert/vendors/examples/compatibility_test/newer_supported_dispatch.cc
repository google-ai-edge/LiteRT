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

#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/examples/compatibility_test/mock_dispatch_common.h"

namespace {

using namespace ::litert::compatibility;  // NOLINT

LiteRtStatus GetNewerVendorId(const char** vendor_id) {
  if (!vendor_id) return kLiteRtStatusErrorInvalidArgument;
  *vendor_id = "NewerSupportedDispatchVendor";
  return kLiteRtStatusOk;
}

static const LiteRtDispatchInterface_V1_2 kNewerSupportedDispatchInterface = {
    .v1 =
        {
            .abi_header =
                {
                    .struct_size = sizeof(LiteRtDispatchInterface_V1_2),
                    .major_version = 1,
                    .minor_version = 2,
                    .reserved = 0,
                },
            .initialize = MockDispatchInitialize,
            .get_vendor_id = GetNewerVendorId,
            .get_build_id = MockDispatchGetBuildId,
            .get_capabilities = MockDispatchGetCapabilities,
            .device_context_create = MockDispatchDeviceContextCreate,
            .device_context_destroy = MockDispatchDeviceContextDestroy,
            .get_input_requirements = MockDispatchGetInputRequirements,
            .get_output_requirements = MockDispatchGetOutputRequirements,
            .register_tensor_buffer = MockDispatchRegisterTensorBuffer,
            .unregister_tensor_buffer = MockDispatchUnregisterTensorBuffer,
            .invocation_context_create = MockDispatchInvocationContextCreate,
            .invocation_context_destroy = MockDispatchInvocationContextDestroy,
            .invocation_context_set_scheduling_info = nullptr,
            .attach_input = MockDispatchAttachInput,
            .attach_output = MockDispatchAttachOutput,
            .detach_input = MockDispatchDetachInput,
            .detach_output = MockDispatchDetachOutput,
            .invoke = MockDispatchInvoke,
            .start_metrics_collection = nullptr,
            .stop_metrics_collection = nullptr,
            .get_num_metrics = nullptr,
            .get_metric = nullptr,
            .destroy_metrics = nullptr,
            .check_runtime_compatibility =
                MockDispatchCheckRuntimeCompatibility,
            .invocation_context_set_options =
                MockDispatchInvocationContextSetOptions,
            .get_hooks = nullptr,
            .attach_edge_buffer = nullptr,
            .detach_edge_buffer = nullptr,
        },
    .future_extension = MockDispatchFutureExtension,
};

}  // namespace

extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtDispatchQueryInterface(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major == 1) {
    if (interface_id == kLiteRtInterfaceBasic) {
      *out_interface = const_cast<LiteRtDispatchInterface_V1_2*>(
          &kNewerSupportedDispatchInterface);
      return kLiteRtStatusOk;
    }
  }
  return kLiteRtStatusErrorUnsupported;
}
