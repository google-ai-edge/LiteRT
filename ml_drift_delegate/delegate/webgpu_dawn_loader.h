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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_WEBGPU_DAWN_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_WEBGPU_DAWN_LOADER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::ml_drift {

// Ensures that the Dawn WebGPU C API functions (`wgpu*`) are available.
//
// When built with dynamic Dawn loading (`LITERT_WEBGPU_DAWN_DYNAMIC_LOADER`),
// this first checks whether Dawn symbols are already available in the current
// process and otherwise attempts to dynamically load `libwebgpu_dawn.so` /
// `libwebgpu_dawn.dylib` / `webgpu_dawn.dll` (searching `runtime_library_dir`
// first when non-empty, followed by the default dynamic library search paths).
// Returns `absl::NotFoundError` if no Dawn runtime library could be opened, or
// `absl::FailedPreconditionError` if a library was opened but lacked required
// WebGPU symbols.
//
// When built with static Dawn or Emscripten, this returns `absl::OkStatus()`
// immediately.
//
// This function is thread-safe. The first successful load is cached for the
// lifetime of the process (subsequent calls return `absl::OkStatus()` without
// re-probing `runtime_library_dir`); failed calls are not cached so subsequent
// calls may retry.
absl::Status EnsureWebGpuDawnLoaded(absl::string_view runtime_library_dir = "");

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_WEBGPU_DAWN_LOADER_H_
