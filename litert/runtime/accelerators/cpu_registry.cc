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

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"

#if defined(LITERT_USE_XNNPACK) || defined(LITERT_HAS_YNNPACK)
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/runtime/accelerators/registration_helper.h"
#endif

#if defined(LITERT_HAS_YNNPACK)
#include "litert/runtime/accelerators/ynnpack/ynnpack_accelerator.h"
#endif

#if defined(LITERT_USE_XNNPACK)
extern "C" {
// Defined in xnnpack_accelerator.cc.
extern LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorCpuDef;
}
#endif

namespace litert::internal {

LiteRtStatus RegisterCpuAccelerator(LiteRtEnvironment environment) {
  // CompiledModel applies delegates in registration order. Register YNNPACK
  // first so XNNPACK can delegate only the remaining CPU nodes.
#if defined(LITERT_HAS_YNNPACK)
  if (LiteRtStaticLinkedAcceleratorYnnpackDef == nullptr) {
    LITERT_LOG(LITERT_WARNING, "YNNPACK CPU accelerator is disabled.");
    return kLiteRtStatusErrorUnsupported;
  }
  {
    auto status = litert::internal::RegisterAcceleratorFromDef(
        environment, LiteRtStaticLinkedAcceleratorYnnpackDef);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(
          LITERT_WARNING,
          "YNNPACK CPU accelerator could not be loaded and registered: %s.",
          LiteRtGetStatusString(status));
      return status;
    }
    LITERT_LOG(LITERT_INFO, "YNNPACK CPU accelerator registered.");
  }
#endif  // defined(LITERT_HAS_YNNPACK)

#if defined(LITERT_USE_XNNPACK)
  if (LiteRtStaticLinkedAcceleratorCpuDef == nullptr) {
    LITERT_LOG(LITERT_VERBOSE, "XNNPACK CPU accelerator is disabled.");
    return kLiteRtStatusErrorUnsupported;
  }
  {
    auto status = litert::internal::RegisterAcceleratorFromDef(
        environment, LiteRtStaticLinkedAcceleratorCpuDef);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(
          LITERT_WARNING,
          "XNNPACK CPU accelerator could not be loaded and registered: %s.",
          LiteRtGetStatusString(status));
      return status;
    }
    LITERT_LOG(LITERT_INFO, "XNNPACK CPU accelerator registered.");
  }
#endif  // defined(LITERT_USE_XNNPACK)

#if defined(LITERT_USE_XNNPACK) || defined(LITERT_HAS_YNNPACK)
  return kLiteRtStatusOk;
#else
  LITERT_LOG(LITERT_VERBOSE, "CPU accelerators are disabled.");
  return kLiteRtStatusErrorUnsupported;
#endif
}

}  // namespace litert::internal
