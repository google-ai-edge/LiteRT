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

#include "litert/c/internal/litert_logging.h"
#include "openvino/runtime/properties.hpp"

// Declarations of the entry points exported by libnpu_hal_hook.so. These are
// resolved at runtime via dlsym; the declarations only exist so decltype can
// derive the matching function-pointer types.
extern "C" {
int npu_hal_submit_inference_async(void** ctx, void* infer_request,
                                   int32_t job_priority, int32_t original_uid);
void npu_hal_release_context(void* ctx);
}

namespace litert::openvino {

// Job priority used when the caller supplies none but the NPU HAL is present.
// Middle of the [0, 1000] range -> MEDIUM scheduling priority.
inline constexpr int32_t kDefaultJobPriority = 500;

inline ov::hint::Priority ToOvModelPriority(int32_t job_priority) {
  // LiteRT priority is [0, 1000] where lower value means higher priority.
  if (job_priority <= 333) {
    return ov::hint::Priority::HIGH;
  }
  if (job_priority <= 666) {
    return ov::hint::Priority::MEDIUM;
  }
  return ov::hint::Priority::LOW;
}

// Function symbols resolved from libnpu_hal_hook.so under a single dlopen.
struct NpuHalHooks {
  decltype(&npu_hal_submit_inference_async) submit_inference_async = nullptr;
  decltype(&npu_hal_release_context) release_context = nullptr;
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
    void* handle =
        dlopen("/vendor/lib64/libnpu_hal_hook.so", RTLD_NOW | RTLD_GLOBAL);
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
    return resolved;
  }();
  return hooks;
}

}  // namespace litert::openvino
#endif  // defined(__ANDROID__)

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_NPU_HAL_WRAPPER_H_
