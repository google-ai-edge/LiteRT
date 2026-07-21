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

#include "litert/c/internal/litert_runtime_builtin.h"

#include "litert/c/internal/litert_runtime_c_api.h"

// Header-selected builtin runtime implementation.
#if defined(LITERT_CC_RUNTIME_BUILTIN_NONE)
#if defined(__ANDROID__)
#include <string>
#include <utility>

#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#endif  // defined(__ANDROID__)
#else
#include "litert/c/internal/litert_runtime_api_export.h"
#endif

/// Gets the builtin LiteRT runtime implementation.
///
/// If `LITERT_CC_RUNTIME_BUILTIN_NONE` is defined:
/// - On Android, dynamically loads the LiteRT shared library ("libLiteRt.so")
/// and retrieves the builtin runtime API structure.
/// - On other platforms, returns `nullptr`.
///
/// If `LITERT_CC_RUNTIME_BUILTIN_NONE` is not defined:
/// - Returns the linked `kLiteRtRuntimeBuiltin` instance.
const LiteRtRuntimeCApiStruct* GetLiteRtRuntimeBuiltin() {
#if defined(LITERT_CC_RUNTIME_BUILTIN_NONE)
#if defined(__ANDROID__)
  static const LiteRtRuntimeCApiStruct* dynamic_runtime =
      []() -> const LiteRtRuntimeCApiStruct* {
    auto dynamic_lib = litert::SharedLibrary::Load(
        "libLiteRt.so", litert::RtldFlags::Default());
    if (!dynamic_lib) {
      LITERT_LOG(LITERT_ERROR, "Failed to load LiteRT runtime library: %s",
                 std::string(dynamic_lib.Error().Message()).c_str());
      return nullptr;
    }
    litert::Expected<const LiteRtRuntimeCApiStruct*> runtime_api_ptr =
        dynamic_lib->LookupSymbol<const LiteRtRuntimeCApiStruct*>(
            "kLiteRtRuntimeBuiltin");
    if (!runtime_api_ptr) {
      LITERT_LOG(LITERT_ERROR,
                 "Loaded runtime library but kLiteRtRuntimeBuiltin not found.");
      return nullptr;
    }
    LITERT_LOG(LITERT_INFO,
               "Loaded runtime library and found kLiteRtRuntimeBuiltin.");
    // Leak the library handle to keep it loaded.
    [[maybe_unused]] static auto* leaked_lib =
        new litert::SharedLibrary(std::move(*dynamic_lib));
    return *runtime_api_ptr;
  }();
  return dynamic_runtime;
#else   // !defined(__ANDROID__)
  return nullptr;
#endif  // defined(__ANDROID__)
#else   // !defined(LITERT_CC_RUNTIME_BUILTIN_NONE)
  return &kLiteRtRuntimeBuiltin;
#endif  // defined(LITERT_CC_RUNTIME_BUILTIN_NONE)
}
