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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_OPENCL_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_OPENCL_TYPES_H_

#include <stdint.h>

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * Define LiteRT aliases for OpenCL types `cl_mem` and `cl_event`,
 * but ensure that they are always defined, even if OpenCL isn't supported.
 */
#if LITERT_HAS_OPENCL_SUPPORT
typedef cl_mem LiteRtClMem;
typedef cl_event LiteRtClEvent;
typedef cl_int LiteRtClInt;
#define LITE_RT_CL_SUCCESS CL_SUCCESS
#define LITE_RT_CL_COMPLETE CL_COMPLETE
#else
typedef struct LiteRtClMemStruct* LiteRtClMem;
typedef struct LiteRtClEventStruct* LiteRtClEvent;
typedef int32_t LiteRtClInt;
#define LITE_RT_CL_SUCCESS 0
#define LITE_RT_CL_COMPLETE 0x0
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_OPENCL_TYPES_H_
