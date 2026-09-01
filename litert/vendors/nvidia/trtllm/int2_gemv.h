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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT2_GEMV_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT2_GEMV_H_

#include <cstdint>

#include "driver_types.h"

// Computes one BF16 activation row times raw TFLite row-major signed INT2
// weights. Four consecutive two's-complement INT2 values occupy each byte,
// least-significant value first. Scales and output are BF16; accumulation is
// FP32.
extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int2PerChannelGemv(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int input_size, int output_size, cudaStream_t stream);

// Computes one BF16 activation row times raw TFLite row-major signed INT2 or
// INT4 weights with BF16 per-channel scales and FP32 accumulation.
extern "C" cudaError_t LiteRtNvidiaLaunchBf16SubbytePerChannelGemv(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int bit_width, int input_size, int output_size,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT2_GEMV_H_
