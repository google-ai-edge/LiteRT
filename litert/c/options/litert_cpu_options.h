// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_CPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_CPU_OPTIONS_H_

#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LrtCpuOptions LrtCpuOptions;

// Selects how CPU ops are executed.
typedef enum LiteRtCpuKernelMode {
  // Use the CPU delegate pipeline. This is the default CPU mode.
  // XNNPACK delegates CPU ops, and an enabled YNNPACK delegate runs first
  // when it was compiled into the runtime.
  kLiteRtCpuKernelModeDelegate = 0,
  // Legacy source-compatible alias for the delegate mode.
  kLiteRtCpuKernelModeXnnpack = kLiteRtCpuKernelModeDelegate,
  // Use LiteRT's built-in reference kernels instead of CPU delegates.
  kLiteRtCpuKernelModeReference = 1,
  // Use LiteRT's built-in optimized kernels instead of CPU delegates.
  kLiteRtCpuKernelModeBuiltin = 2,
} LiteRtCpuKernelMode;

// Creates a cpu options object.
// The caller is responsible for freeing the returned options using
// `LrtDestroyCpuOptions`.
LiteRtStatus LrtCreateCpuOptions(LrtCpuOptions** options);

// Destroys a cpu options object.
void LrtDestroyCpuOptions(LrtCpuOptions* options);

// Serializes cpu options and returns the components needed to create opaque
// options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions`.
LiteRtStatus LrtGetOpaqueCpuOptionsData(const LrtCpuOptions* options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*));

// Gets the identifier for CPU options stored in opaque options.
const char* LrtGetCpuOptionsIdentifier();

// Sets how LiteRT should execute CPU ops.
LiteRtStatus LrtSetCpuOptionsKernelMode(LrtCpuOptions* options,
                                        LiteRtCpuKernelMode mode);

// Gets the CPU kernel mode that was set.
LiteRtStatus LrtGetCpuOptionsKernelMode(const LrtCpuOptions* options,
                                        LiteRtCpuKernelMode* mode);

// Enables YNNPACK to delegate supported CPU ops before XNNPACK. Requires a
// build with `--define litert_enable_ynnpack=true`; otherwise XNNPACK is used.
LiteRtStatus LrtSetCpuOptionsEnableYNNPack(LrtCpuOptions* options,
                                           bool enable_ynnpack);

// Gets whether YNNPACK delegation was enabled.
LiteRtStatus LrtGetCpuOptionsEnableYNNPack(const LrtCpuOptions* options,
                                           bool* enable_ynnpack);

// Sets the number of CPU threads used by the CPU accelerator.
LiteRtStatus LrtSetCpuOptionsNumThread(LrtCpuOptions* options, int num_threads);

// Gets the number of CPU threads used by the CPU accelerator.
LiteRtStatus LrtGetCpuOptionsNumThread(const LrtCpuOptions* options,
                                       int* num_threads);

// Sets the XNNPack flags used by XNNPACK in delegate mode.
LiteRtStatus LrtSetCpuOptionsXNNPackFlags(LrtCpuOptions* options,
                                          uint32_t flags);

// Gets the XNNPack flags used by XNNPACK in delegate mode.
LiteRtStatus LrtGetCpuOptionsXNNPackFlags(const LrtCpuOptions* options,
                                          uint32_t* flags);

// Sets whether to hint at fully delegating to a single delegate.
LiteRtStatus LrtSetCpuOptionsHintFullyDelegatedToSingleDelegate(
    LrtCpuOptions* options, bool hint_fully_delegated_to_single_delegate);

// Sets the XNNPack weight cache file path used by XNNPACK in delegate
// mode.
// Weight cache file path and descriptor must not both be set.
// The `path` string is copied into the options object.
LiteRtStatus LrtSetCpuOptionsXnnPackWeightCachePath(LrtCpuOptions* options,
                                                    const char* path);

// Gets the XNNPack weight cache file path used by XNNPACK in delegate
// mode.
// The returned string pointer is owned by the options object and is valid
// as long as the options object is valid and the path is not modified.
LiteRtStatus LrtGetCpuOptionsXnnPackWeightCachePath(
    const LrtCpuOptions* options, const char** path);

// Sets the XNNPack weight cache file descriptor used by XNNPACK in
// delegate mode.
// Weight cache file path and descriptor must not both be set.
LiteRtStatus LrtSetCpuOptionsXnnPackWeightCacheFileDescriptor(
    LrtCpuOptions* options, int fd);

// Gets the XNNPack weight cache file descriptor used by XNNPACK in
// delegate mode.
LiteRtStatus LrtGetCpuOptionsXnnPackWeightCacheFileDescriptor(
    const LrtCpuOptions* options, int* fd);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_CPU_OPTIONS_H_
