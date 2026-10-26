// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_OPTIONS_LITERT_DARWINN_DEVICE_OPTIONS_H_
#define ODML_LITERT_LITERT_C_OPTIONS_LITERT_DARWINN_DEVICE_OPTIONS_H_

#include <stdbool.h>
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtDarwinnDeviceOptions);

// Creates DarwiNN device options and adds them to the opaque options list.
LiteRtStatus LiteRtCreateDarwinnDeviceOptions(LiteRtOpaqueOptions* options);

// Finds DarwiNN device options in the opaque options list.
LiteRtStatus LiteRtFindDarwinnDeviceOptions(
    LiteRtOpaqueOptions opaque_options,
    LiteRtDarwinnDeviceOptions* device_options);

// Gets the identifier for DarwiNN device options.
const char* LiteRtGetDarwinnDeviceOptionsIdentifier();

// Device selection setters/getters
LiteRtStatus LiteRtSetDarwinnDeviceType(LiteRtDarwinnDeviceOptions options,
                                        const char* device_type);
LiteRtStatus LiteRtGetDarwinnDeviceType(LiteRtDarwinnDeviceOptionsConst options,
                                        const char** device_type);

LiteRtStatus LiteRtSetDarwinnDevicePath(LiteRtDarwinnDeviceOptions options,
                                        const char* device_path);
LiteRtStatus LiteRtGetDarwinnDevicePath(LiteRtDarwinnDeviceOptionsConst options,
                                        const char** device_path);

// Compilation options setters/getters
LiteRtStatus LiteRtSetDarwinnEnableMultipleSubgraphs(
    LiteRtDarwinnDeviceOptions options, bool enable);
LiteRtStatus LiteRtGetDarwinnEnableMultipleSubgraphs(
    LiteRtDarwinnDeviceOptionsConst options, bool* enable);

LiteRtStatus LiteRtSetDarwinnCompileIfResize(LiteRtDarwinnDeviceOptions options,
                                             bool compile_if_resize);
LiteRtStatus LiteRtGetDarwinnCompileIfResize(
    LiteRtDarwinnDeviceOptionsConst options, bool* compile_if_resize);

LiteRtStatus LiteRtSetDarwinnAllowCpuFallback(
    LiteRtDarwinnDeviceOptions options, bool allow_cpu_fallback);
LiteRtStatus LiteRtGetDarwinnAllowCpuFallback(
    LiteRtDarwinnDeviceOptionsConst options, bool* allow_cpu_fallback);

LiteRtStatus LiteRtSetDarwinnSkipOpFilter(LiteRtDarwinnDeviceOptions options,
                                          bool skip_op_filter);
LiteRtStatus LiteRtGetDarwinnSkipOpFilter(
    LiteRtDarwinnDeviceOptionsConst options, bool* skip_op_filter);

LiteRtStatus LiteRtSetDarwinnNumInterpreters(LiteRtDarwinnDeviceOptions options,
                                             int num_interpreters);
LiteRtStatus LiteRtGetDarwinnNumInterpreters(
    LiteRtDarwinnDeviceOptionsConst options, int* num_interpreters);

// Memory configuration setters/getters
LiteRtStatus LiteRtSetDarwinnAvoidBounceBuffer(
    LiteRtDarwinnDeviceOptions options, bool avoid_bounce_buffer);
LiteRtStatus LiteRtGetDarwinnAvoidBounceBuffer(
    LiteRtDarwinnDeviceOptionsConst options, bool* avoid_bounce_buffer);

LiteRtStatus LiteRtSetDarwinnRegisterGraphDuringModify(
    LiteRtDarwinnDeviceOptions options, bool register_graph);
LiteRtStatus LiteRtGetDarwinnRegisterGraphDuringModify(
    LiteRtDarwinnDeviceOptionsConst options, bool* register_graph);

LiteRtStatus LiteRtSetDarwinnInKernelFence(LiteRtDarwinnDeviceOptions options,
                                           bool in_kernel_fence);
LiteRtStatus LiteRtGetDarwinnInKernelFence(
    LiteRtDarwinnDeviceOptionsConst options, bool* in_kernel_fence);

LiteRtStatus LiteRtSetDarwinnSkipIntermediateBufferAllocation(
    LiteRtDarwinnDeviceOptions options, bool skip_allocation);
LiteRtStatus LiteRtGetDarwinnSkipIntermediateBufferAllocation(
    LiteRtDarwinnDeviceOptionsConst options, bool* skip_allocation);

LiteRtStatus LiteRtSetDarwinnGraphBuffersDonatable(
    LiteRtDarwinnDeviceOptions options, bool donatable);
LiteRtStatus LiteRtGetDarwinnGraphBuffersDonatable(
    LiteRtDarwinnDeviceOptionsConst options, bool* donatable);

// Async/Tachyon configuration setters/getters
LiteRtStatus LiteRtSetDarwinnUseAsyncApi(LiteRtDarwinnDeviceOptions options,
                                         bool use_async_api);
LiteRtStatus LiteRtGetDarwinnUseAsyncApi(
    LiteRtDarwinnDeviceOptionsConst options, bool* use_async_api);

LiteRtStatus LiteRtSetDarwinnUseTachyon(LiteRtDarwinnDeviceOptions options,
                                        bool use_tachyon);
LiteRtStatus LiteRtGetDarwinnUseTachyon(LiteRtDarwinnDeviceOptionsConst options,
                                        bool* use_tachyon);

// Logging setters/getters
LiteRtStatus LiteRtSetDarwinnDisableLogInfo(LiteRtDarwinnDeviceOptions options,
                                            bool disable_log_info);
LiteRtStatus LiteRtGetDarwinnDisableLogInfo(
    LiteRtDarwinnDeviceOptionsConst options, bool* disable_log_info);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_OPTIONS_LITERT_DARWINN_DEVICE_OPTIONS_H_
