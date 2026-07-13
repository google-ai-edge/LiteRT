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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_KV_CACHE_METAL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_KV_CACHE_METAL_H_

#include <stddef.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Copy KV cache buffers on GPU using Metal.
//
// This function performs GPU-side copy of KV cache buffers using
// MTLBlitCommandEncoder. It supports both "Reduce" (copying a specific index
// from a larger prefill buffer to a smaller decode buffer) and "Broadcast"
// (copying from a smaller prefill buffer to multiple batch locations in a
// larger decode buffer).
//
// @param src_buffer Pointer to the source MTLBuffer.
// @param dst_buffer Pointer to the destination MTLBuffer.
// @param src_index_to_copy_on_prefill Index to copy for Reduce operation. If
//        >= 0, it copies only the cache content of this index. If < 0, it
//        performs a Broadcast operation, copying the source buffer to all
//        batches in destination.
// @param decode_batch_size The batch size for decode (used in Broadcast).
// @param src_buffer_size The size of the source buffer in bytes.
// @param dst_buffer_size The size of the destination buffer in bytes.
// @param command_queue Pointer to the MTLCommandQueue to use for the copy.
//
// @return kLiteRtStatusOk on success, or an error code on failure.
// @note This function may be dynamically loaded via dlsym() in LiteRT-LM.
// Therefore, any changes to its name or signature should be made with caution.
LiteRtStatus LiteRtCopyKvCacheMetal(void* src_buffer, void* dst_buffer,
                                    int src_index_to_copy_on_prefill,
                                    int decode_batch_size,
                                    size_t src_buffer_size,
                                    size_t dst_buffer_size,
                                    void* command_queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_KV_CACHE_METAL_H_
