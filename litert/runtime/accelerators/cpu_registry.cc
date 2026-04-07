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
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/runtime/accelerators/registration_helper.h"

#if defined(__EMSCRIPTEN__)
#include "absl/base/attributes.h"  // from @com_google_absl
extern "C" {
// For Emscripten, we use weak symbols to discover if an accelerator is linked.
// This allows the linker to resolve the pointer at link-time without relying on
// dlsym(RTLD_DEFAULT). On WASM environments using the Side Module (dynamic
// linking) pattern, dlsym(RTLD_DEFAULT) may not reliably search the main
// module's symbol table, whereas weak symbols are correctly resolved by the
// Emscripten linker during the final link of the WASM binary.
ABSL_ATTRIBUTE_WEAK const LiteRtAcceleratorDef*
    LiteRtStaticLinkedAcceleratorCpuDef;
}  // extern "C"
#endif

namespace litert::internal {

LiteRtStatus LiteRtRegisterCpuAccelerator(LiteRtEnvironment environment) {
  const LiteRtAcceleratorDef* static_cpu_def = nullptr;

#if defined(__EMSCRIPTEN__)
  static_cpu_def = LiteRtStaticLinkedAcceleratorCpuDef;
#else
  // For standard platforms (Linux, Windows, Mac), we use process-wide symbol
  // lookup via SharedLibrary. This avoids the use of static initializers
  // (which are forbidden in Chrome) while still allowing runtime selection
  // based on what code was linked into the binary.
  //
  // Note: On POSIX, this requires targeted linker flags (like
  // --export-dynamic-symbol) to ensure the discovery pointers are present in
  // the dynamic symbol table if they are in the main executable.
  auto self = SharedLibrary::Load(RtldFlags::kDefault);
  if (self.HasValue()) {
    auto sym = self->LookupSymbol<const LiteRtAcceleratorDef**>(
        "LiteRtStaticLinkedAcceleratorCpuDef");
    if (sym.HasValue()) {
      static_cpu_def = **sym;
    }
  }
#endif

  if (static_cpu_def == nullptr) {
    LITERT_LOG(LITERT_VERBOSE, "CPU accelerator is disabled.");
    return kLiteRtStatusErrorUnsupported;
  }

  auto status = litert::internal::RegisterAcceleratorFromDef(
      environment, static_cpu_def);
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
