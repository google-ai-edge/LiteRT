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

// Mock dispatch library with corrupted struct_size < sizeof(LiteRtAbiHeader).
extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtDispatchQueryInterface(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  static const LiteRtDispatchInterface_V1 corrupted_interface = {
      .abi_header =
          {
              // Corrupted: smaller than sizeof(LiteRtAbiHeader) = 8.
              .struct_size = 4,
              .major_version = 1,
              .minor_version = 0,
              .reserved = 0,
          },
  };
  *out_interface =
      const_cast<LiteRtDispatchInterface_V1*>(&corrupted_interface);
  return kLiteRtStatusOk;
}
