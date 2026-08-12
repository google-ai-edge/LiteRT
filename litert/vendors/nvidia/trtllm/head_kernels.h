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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_HEAD_KERNELS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_HEAD_KERNELS_H_

#include <driver_types.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Converts count F32 values to BF16 using round-to-nearest-even.
cudaError_t LiteRtNvidiaLaunchF32ToBf16(const float* input, void* output,
                                        size_t count, cudaStream_t stream);

// Converts count BF16 logits to F32 while applying
// soft_cap * tanh(logit / soft_cap).
cudaError_t LiteRtNvidiaLaunchBf16SoftCapToF32(const void* input, float* output,
                                               size_t count, float soft_cap,
                                               cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_HEAD_KERNELS_H_
