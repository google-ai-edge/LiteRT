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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_WEBGPU_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_WEBGPU_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * Define LiteRT alias for WebGPU type WGPUBuffer,
 * but ensure that it is always defined, even if WebGPU isn't supported.
 */
#if LITERT_HAS_WEBGPU_SUPPORT
typedef struct WGPUBufferImpl LiteRtWGPUBufferImpl;
#else
typedef struct LiteRtWGPUBufferImpl LiteRtWGPUBufferImpl;
#endif  // LITERT_HAS_WEBGPU_SUPPORT
typedef LiteRtWGPUBufferImpl* LiteRtWGPUBuffer;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_WEBGPU_TYPES_H_
