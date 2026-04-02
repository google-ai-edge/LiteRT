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

#include "litert/runtime/accelerators/cpu_registry.h"

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/runtime/accelerators/registration_helper.h"

extern "C" {

// Accelerator pointer initialized in xnnpack_accelerator.cc.
const LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorCpuDef = nullptr;

}  // extern "C"

namespace litert::internal {

LiteRtStatus LiteRtRegisterCpuAccelerator(LiteRtEnvironment environment) {
  if (LiteRtStaticLinkedAcceleratorCpuDef == nullptr) {
    LITERT_LOG(LITERT_VERBOSE, "CPU accelerator is disabled.");
    return kLiteRtStatusErrorUnsupported;
  }

  auto status = litert::internal::RegisterAcceleratorFromDef(
      environment, LiteRtStaticLinkedAcceleratorCpuDef);
  if (status == kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "CPU accelerator registered.");
  } else {
    LITERT_LOG(LITERT_WARNING,
               "CPU accelerator could not be loaded and registered: %s.",
               LiteRtGetStatusString(status));
  }
  return status;
}

}  // namespace litert::internal
