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

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * Define LiteRT aliases for OpenCL types without requiring downstream users to
 * include the OpenCL SDK headers just to compile LiteRT public headers.
 *
 * The OpenCL C API models both `cl_mem` and `cl_event` as opaque pointers to
 * implementation-defined structs. Forward-declaring the same struct tags keeps
 * LiteRT's public ABI compatible with code that later includes `CL/cl.h`,
 * while avoiding a transitive dependency on the OpenCL headers for consumers
 * that never bind directly to raw OpenCL objects.
 */
typedef struct _cl_mem* LiteRtClMem;
typedef struct _cl_event* LiteRtClEvent;
typedef int32_t LiteRtClInt;

#define LITE_RT_CL_SUCCESS 0
#define LITE_RT_CL_COMPLETE 0x0

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_OPENCL_TYPES_H_
