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

#include "litert/runtime/accelerators/npu_registry.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"

#if !defined(LITERT_DISABLE_NPU)
#include "litert/runtime/accelerators/dispatch/dispatch_accelerator.h"
#endif  // !defined(LITERT_DISABLE_NPU)

namespace litert::internal {

LiteRtStatus LiteRtRegisterNpuAccelerator(LiteRtEnvironment environment) {
#if !defined(LITERT_DISABLE_NPU)
  auto npu_registration = ::LiteRtRegisterNpuAccelerator(environment);
  if (npu_registration == kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "NPU accelerator registered.");
  } else {
    LITERT_LOG(LITERT_WARNING,
               "NPU accelerator could not be loaded and registered: %s.",
               LiteRtGetStatusString(npu_registration));
  }
  return npu_registration;
#else
  LITERT_LOG(LITERT_VERBOSE, "NPU accelerator is disabled.");
  return kLiteRtStatusErrorUnsupported;
#endif
}

}  // namespace litert::internal
