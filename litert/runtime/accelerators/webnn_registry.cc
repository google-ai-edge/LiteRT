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

#include "litert/runtime/accelerators/webnn_registry.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/core/environment.h"

extern "C" {

// Define a function pointer for the WebNN accelerator.
LiteRtStatus (*LiteRtRegisterStaticLinkedAcceleratorWebNn)(
    LiteRtEnvironmentT& environment) = nullptr;

}  // extern "C"

namespace litert::internal {

LiteRtStatus LiteRtRegisterWebNnAccelerator(LiteRtEnvironment environment) {
  if (LiteRtRegisterStaticLinkedAcceleratorWebNn != nullptr &&
      LiteRtRegisterStaticLinkedAcceleratorWebNn(*environment) ==
          kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "Statically linked WebNN accelerator registered.");
    return kLiteRtStatusOk;
  }

  LITERT_LOG(LITERT_VERBOSE, "WebNN accelerator is disabled.");
  return kLiteRtStatusErrorUnsupported;
}

}  // namespace litert::internal
