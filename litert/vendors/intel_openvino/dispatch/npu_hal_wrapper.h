// Copyright (C) 2025-2026 Intel Corporation
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

#include <cstddef>
#include <cstdint>
#include <optional>

#include "litert/c/internal/litert_logging.h"
#include "openvino/runtime/properties.hpp"

// Declarations of the entry points exported by libnpu_hal_hook.so. They are
// resolved at runtime with dlsym; the declarations exist only so that decltype
// can derive the matching function-pointer types. Names and values mirror the
// hook's header.
extern "C" {

// Callbacks the hook uses to run an inference. The hook never sees an OpenVINO
// object: the caller closes over its own inference request, so the request is
// always run by this library's OpenVINO.
typedef struct {
  size_t struct_size;  // Set to sizeof(npu_hal_infer_ops_t).
  // Must not block. A failure is reported by `wait`.
  void (*start)(void* user);
  // 0 on success, non-zero on timeout or error.
  int (*wait)(void* user, uint64_t timeout_ns);
  // `cancel` and `get_output` are reserved and may be null.
  void (*cancel)(void* user);
  int (*get_output)(void* user, size_t idx, void** data, size_t* bytes);
} npu_hal_infer_ops_t;

// Results of npu_hal_run_gated. ERROR does not say whether `start` ran.
enum {
  NPU_HAL_SUBMIT_OK = 0,
  NPU_HAL_SUBMIT_ERROR = -1,
  NPU_HAL_SUBMIT_ACCESS_DENIED = -2,
};

// Runs one inference through `ops`, reporting it to the HAL.
int npu_hal_run_gated(int32_t job_priority, int32_t original_uid,
                      const npu_hal_infer_ops_t* ops, void* user,
                      void** out_ctx);

// Releases the context that npu_hal_run_gated hands back.
void npu_hal_release_context(void* ctx);

// Model-priority hint buckets returned by the resolver.
typedef enum {
  NPU_HAL_MODEL_PRIORITY_LOW = 0,
  NPU_HAL_MODEL_PRIORITY_MEDIUM = 1,
  NPU_HAL_MODEL_PRIORITY_HIGH = 2,
} npu_hal_model_priority_e;

// Resolves the model-priority hint for a job priority and the app the work is
// attributed to.
npu_hal_model_priority_e npu_hal_resolve_ov_model_priority(
    int32_t job_priority, int32_t original_uid);

}  // extern "C"

namespace litert::openvino {

// Caller job priority when none is supplied: 0, the highest in the [0, 1000]
// range. Reported to the HAL as-is.
inline constexpr int32_t kDefaultJobPriority = 0;

// Results of the `wait` callback handed to npu_hal_run_gated: 0 on success,
// non-zero on timeout or error.
inline constexpr int kGatedWaitOk = 0;
inline constexpr int kGatedWaitFailed = -1;

// Function symbols resolved from libnpu_hal_hook.so under a single dlopen.
struct NpuHalHooks {
  decltype(&npu_hal_run_gated) run_gated = nullptr;
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
    resolved.run_gated = reinterpret_cast<decltype(&npu_hal_run_gated)>(
        dlsym(handle, "npu_hal_run_gated"));
    if (resolved.run_gated == nullptr) {
      // A hook that predates the OpenVINO-free interface is not used: running
      // without the HAL is the same as having no hook.
      LITERT_LOG(LITERT_WARNING,
                 "libnpu_hal_hook.so has no npu_hal_run_gated, NPU HAL "
                 "scheduling disabled: %s",
                 dlerror());
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
    case NPU_HAL_MODEL_PRIORITY_LOW:
      return ov::hint::Priority::LOW;
    case NPU_HAL_MODEL_PRIORITY_MEDIUM:
    default:
      // An unknown value from a newer hook must not demote the model.
      return ov::hint::Priority::MEDIUM;
  }
}

}  // namespace litert::openvino
#endif  // defined(__ANDROID__)

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_NPU_HAL_WRAPPER_H_
