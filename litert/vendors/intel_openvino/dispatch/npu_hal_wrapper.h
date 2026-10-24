// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_NPU_HAL_WRAPPER_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_NPU_HAL_WRAPPER_H_

#if defined(__ANDROID__)
#include <dlfcn.h>

#include <cstdint>
#include <optional>

#include "openvino/runtime/properties.hpp"
#include "litert/c/internal/litert_logging.h"

// Declarations of the entry points exported by libnpu_hal_hook.so. These are
// resolved at runtime via dlsym; the declarations only exist so decltype can
// derive the matching function-pointer types.
extern "C" {
int npu_hal_submit_inference_async(void** ctx, void* infer_request,
                                   int32_t job_priority, int32_t original_uid);
void npu_hal_release_context(void* ctx);
typedef enum {
  NPU_HAL_MODEL_PRIORITY_LOW = 0,
  NPU_HAL_MODEL_PRIORITY_MEDIUM = 1,
  NPU_HAL_MODEL_PRIORITY_HIGH = 2,
} npu_hal_model_priority_e;
npu_hal_model_priority_e npu_hal_resolve_ov_model_priority(int32_t job_priority,
                                                          int32_t original_uid);
}

namespace litert::openvino {

// Caller job priority when none is supplied. [0, 1000], default 0 = no demotion
// (job priority can only demote). Reported to the HAL as-is.
inline constexpr int32_t kDefaultJobPriority = 0;

// Function symbols resolved from libnpu_hal_hook.so under a single dlopen.
struct NpuHalHooks {
  decltype(&npu_hal_submit_inference_async) submit_inference_async = nullptr;
  decltype(&npu_hal_release_context) release_context = nullptr;
  // Optional unified resolver. Absent only on a pre-resolver hook, in which
  // case no model-priority hint is applied.
  decltype(&npu_hal_resolve_ov_model_priority) resolve_ov_model_priority =
      nullptr;
  // Set when the library loaded but a required symbol could not be resolved.
  bool load_error = false;
};

// Loads libnpu_hal_hook.so exactly once and resolves all required symbols from
// that single handle, storing them in the returned struct. If the library
// loads but a required symbol is missing, `load_error` is set so callers can
// abort. The handle is intentionally kept open for the lifetime of the process.
inline const NpuHalHooks& GetNpuHalHooks() {
  static NpuHalHooks hooks = []() -> NpuHalHooks {
    NpuHalHooks resolved;
    // Loaded by soname, not by path: the hook is listed in
    // /vendor/etc/public.libraries.txt, so an app resolves it through the sphal
    // namespace, and a path would also hard-code the 64-bit lib directory.
    void* handle = dlopen("libnpu_hal_hook.so", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      LITERT_LOG(LITERT_WARNING,
                 "libnpu_hal_hook.so not available, NPU HAL priority "
                 "scheduling disabled: %s",
                 dlerror());
      return resolved;
    }
    // Clear any existing error before dlsym.
    dlerror();
    resolved.submit_inference_async =
        reinterpret_cast<decltype(&npu_hal_submit_inference_async)>(
            dlsym(handle, "npu_hal_submit_inference_async"));
    if (resolved.submit_inference_async == nullptr) {
      LITERT_LOG(LITERT_ERROR, "npu_hal_submit_inference_async not found: %s",
                 dlerror());
      resolved.load_error = true;
      return resolved;
    }
    resolved.release_context =
        reinterpret_cast<decltype(&npu_hal_release_context)>(
            dlsym(handle, "npu_hal_release_context"));
    if (resolved.release_context == nullptr) {
      LITERT_LOG(LITERT_ERROR, "npu_hal_release_context not found: %s",
                 dlerror());
      resolved.load_error = true;
      return resolved;
    }

    // Optional unified resolver.
    resolved.resolve_ov_model_priority =
        reinterpret_cast<decltype(&npu_hal_resolve_ov_model_priority)>(
            dlsym(handle, "npu_hal_resolve_ov_model_priority"));
    return resolved;
  }();
  return hooks;
}

// This process's OpenVINO model-priority hint for a given job priority and the
// app the work is attributed to (original_uid). Returns nullopt with a
// pre-resolver hook; in that case the caller leaves the model priority at
// OpenVINO's default.
inline std::optional<ov::hint::Priority> ResolveOvModelPriority(
    int32_t job_priority, int32_t original_uid) {
  const NpuHalHooks& hooks = GetNpuHalHooks();
  if (hooks.resolve_ov_model_priority == nullptr) {
    return std::nullopt;
  }
  switch (hooks.resolve_ov_model_priority(job_priority, original_uid)) {
    case NPU_HAL_MODEL_PRIORITY_HIGH:
      return ov::hint::Priority::HIGH;
    case NPU_HAL_MODEL_PRIORITY_MEDIUM:
      return ov::hint::Priority::MEDIUM;
    default:
      return ov::hint::Priority::LOW;
  }
}

}  // namespace litert::openvino
#endif  // defined(__ANDROID__)

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_NPU_HAL_WRAPPER_H_
