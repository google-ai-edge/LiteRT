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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_GREEDY_SAMPLER_C_API_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_GREEDY_SAMPLER_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque state for the optional NVIDIA dispatch k=1 sampler extension.
// A sampler is bound to its first CUDA device and is not thread-safe.
typedef void* LiteRtDispatchNvidiaGreedySampler;

LiteRtStatus LiteRtDispatchNvidiaGreedySamplerCreate(
    LiteRtDispatchNvidiaGreedySampler* sampler);

void LiteRtDispatchNvidiaGreedySamplerDestroy(
    LiteRtDispatchNvidiaGreedySampler sampler);

LiteRtStatus LiteRtDispatchNvidiaGreedySamplerSampleF32(
    LiteRtDispatchNvidiaGreedySampler sampler, LiteRtTensorBuffer logits,
    size_t count, int32_t* token_id);

// Samples each row of a dense FP16/FP32 NVIDIA CUDA buffer [1, rows, vocab_size].
// Canonical contiguous strides are accepted; padded layouts are unsupported.
// token_ids points to rows writable CPU int32 slots. The call waits for the
// logits event and completes one batched device-to-host transfer before return.
// Each row selects the first maximum; a NaN at index zero wins, other NaNs are
// ignored. Unsupported buffer types, element types, or layouts return
// kLiteRtStatusErrorUnsupported. No logits or caller-owned buffers are modified.
LiteRtStatus LiteRtDispatchNvidiaGreedySamplerSampleBatched(
    LiteRtDispatchNvidiaGreedySampler sampler, LiteRtTensorBuffer logits,
    size_t rows, size_t vocab_size, int32_t* token_ids);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_GREEDY_SAMPLER_C_API_H_
