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

#if !defined(LITERT_WINDOWS_OS)
#include <dlfcn.h>
#endif
#include <cstddef>

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/runtime/accelerators/builtin_accelerators.h"
#include "litert/runtime/accelerators/registration_helper.h"

namespace litert::internal {

namespace {
using GetBuiltinAcceleratorsFunc = const LiteRtAcceleratorDef* const* (*)();
using GetBuiltinAcceleratorsSizeFunc = size_t (*)();
}  // namespace

LiteRtStatus LiteRtRegisterCpuAccelerator(LiteRtEnvironment environment) {
  GetBuiltinAcceleratorsFunc get_accels = GetBuiltinCpuAccelerators;
  GetBuiltinAcceleratorsSizeFunc get_size = GetBuiltinCpuAcceleratorsSize;

#if !defined(LITERT_WINDOWS_OS) && !defined(__APPLE__) && \
    !defined(__EMSCRIPTEN__)
  // On Linux/Android, try to find the registry functions in the main executable
  // or other shared libraries if the runtime is in a shared library.
  void* symbol = dlsym(RTLD_DEFAULT, "GetBuiltinCpuAccelerators");
  if (symbol != nullptr) {
    get_accels = reinterpret_cast<GetBuiltinAcceleratorsFunc>(symbol);
  }
  symbol = dlsym(RTLD_DEFAULT, "GetBuiltinCpuAcceleratorsSize");
  if (symbol != nullptr) {
    get_size = reinterpret_cast<GetBuiltinAcceleratorsSizeFunc>(symbol);
  }
#endif

  if (get_accels == nullptr || get_size == nullptr) {
    return kLiteRtStatusErrorNotFound;
  }

  const LiteRtAcceleratorDef* const* builtin_accelerators = get_accels();
  size_t size = get_size();

  LiteRtStatus status = kLiteRtStatusOk;
  bool registered_any = false;
  for (size_t i = 0; i < size; ++i) {
    const LiteRtAcceleratorDef* accel_def = builtin_accelerators[i];
    if (accel_def == nullptr) {
      continue;
    }
    auto registration_status =
        litert::internal::RegisterAcceleratorFromDef(environment, accel_def);
    if (registration_status == kLiteRtStatusOk) {
      LITERT_LOG(LITERT_INFO, "CPU accelerator registered.");
      registered_any = true;
      break;
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "CPU accelerator could not be loaded and registered: %s.",
                 LiteRtGetStatusString(registration_status));
      status = registration_status;
    }
  }

  if (registered_any) {
    return kLiteRtStatusOk;
  }

  return status;
}

}  // namespace litert::internal
