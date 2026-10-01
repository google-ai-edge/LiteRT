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
// Overridden by the strong definition in ynnpack_accelerator.cc when
// ynnpack_accelerator is linked into the binary.
#if defined(LITERT_HAS_YNNPACK)
extern const LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorYnnpackDef;
#elif !defined(_MSC_VER)
__attribute__((weak))
const LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorYnnpackDef = nullptr;
#else
const LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorYnnpackDef = nullptr;
#endif

#if defined(LITERT_USE_XNNPACK)
// Defined in xnnpack_accelerator.cc.
// TODO(gcarranza): Rename LiteRtStaticLinkedAcceleratorCpuDef to
// LiteRtStaticLinkedAcceleratorXnnpackDef once downstream callers are updated.
extern const LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorCpuDef;
#else
const LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorCpuDef = nullptr;
#endif
}  // extern "C"

namespace litert::internal {

// TODO(gcarranza): Remove weak attribute once downstream RegisterCpuAccelerator
// overrides (e.g. litert_lm_advanced_main.cc) are removed.
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
LiteRtStatus RegisterCpuAccelerator(LiteRtEnvironment environment) {
  bool cpu_accelerator_registered = false;

  // CompiledModel applies delegates in registration order. Register YNNPACK
  // first so XNNPACK can delegate only the remaining CPU nodes.
  if (LiteRtStaticLinkedAcceleratorYnnpackDef != nullptr) {
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
    cpu_accelerator_registered = true;
  }

  if (LiteRtStaticLinkedAcceleratorCpuDef != nullptr) {
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
    cpu_accelerator_registered = true;
  }

  if (!cpu_accelerator_registered) {
    LITERT_LOG(LITERT_VERBOSE, "CPU accelerators are disabled.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::internal
