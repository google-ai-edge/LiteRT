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

#include <atomic>
#include <cstdint>

#include "openvino/runtime/properties.hpp"
#include "litert/c/internal/litert_logging.h"

// Declarations of the entry points exported by libnpu_hal_hook.so. These are
// resolved at runtime via dlsym; the declarations only exist so decltype can
// derive the matching function-pointer types.
extern "C" {
int npu_hal_submit_inference_async(void** ctx, void* infer_request,
                                   int32_t job_priority, int32_t original_uid);
void npu_hal_release_context(void* ctx);
int npu_hal_query_priority(int32_t uid, int32_t* out_priority);
typedef void (*npu_hal_priority_callback)(int32_t uid, int32_t new_priority,
                                          int has_direct_access);
int npu_hal_register_priority_callback(npu_hal_priority_callback callback);
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
  // Optional: absent on older hooks, in which case the caller-supplied job
  // priority is used unchanged.
  decltype(&npu_hal_query_priority) query_priority = nullptr;
  decltype(&npu_hal_register_priority_callback) register_priority_callback =
      nullptr;
  // Set when the library loaded but a required symbol could not be resolved.
  bool load_error = false;
};

// This process's priority as last reported by the NPU HAL. NPU Manager
// re-prioritises the process on foreground/background changes, so this is the
// authoritative value rather than anything sampled at load time.
inline std::atomic<int32_t>& HalPriorityRef() {
  static std::atomic<int32_t> value{kDefaultJobPriority};
  return value;
}

// Invoked on a binder thread by the HAL hook.
inline void HalPriorityListener(int32_t uid, int32_t new_priority,
                                int has_direct_access) {
  HalPriorityRef().store(new_priority, std::memory_order_release);
  LITERT_LOG(LITERT_INFO,
             "NPU HAL priority changed: uid=%d priority=%d direct_access=%d",
             uid, new_priority, has_direct_access);
}

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

    // Priority plumbing is optional so an older hook still works; without it
    // the caller-supplied job priority is used as-is.
    resolved.query_priority =
        reinterpret_cast<decltype(&npu_hal_query_priority)>(
            dlsym(handle, "npu_hal_query_priority"));
    resolved.register_priority_callback =
        reinterpret_cast<decltype(&npu_hal_register_priority_callback)>(
            dlsym(handle, "npu_hal_register_priority_callback"));

    if (resolved.query_priority != nullptr) {
      int32_t priority = kDefaultJobPriority;
      // -1 means "this process"; the hook substitutes getuid().
      if (resolved.query_priority(-1, &priority) == 0) {
        HalPriorityRef().store(priority, std::memory_order_release);
        LITERT_LOG(LITERT_INFO, "NPU HAL initial priority: %d", priority);
      }
    }
    if (resolved.register_priority_callback != nullptr) {
      if (resolved.register_priority_callback(&HalPriorityListener) != 0) {
        LITERT_LOG(LITERT_WARNING,
                   "NPU HAL priority listener unavailable; priority will not "
                   "follow foreground/background changes");
      }
    }
    return resolved;
  }();
  return hooks;
}

// This process's current NPU HAL priority. Falls back to kDefaultJobPriority
// when the hook is absent or exposes no priority API.
inline int32_t CurrentHalPriority() {
  GetNpuHalHooks();  // Forces the one-time query and listener registration.
  return HalPriorityRef().load(std::memory_order_acquire);
}

}  // namespace litert::openvino
#endif  // defined(__ANDROID__)

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_NPU_HAL_WRAPPER_H_
