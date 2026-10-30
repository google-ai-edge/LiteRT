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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SUBBYTE_GEMV_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SUBBYTE_GEMV_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"

namespace litert::nvidia {

// BF16 activations [..., columns] times signed INT2 or INT4 weights
// [rows, columns] with BF16 per-channel scales:
//   inputs  activation BF16, packed weights INT8 (or a holder INT64, see
//           below), scales BF16
//   output  [..., rows] BF16
// One activation row (decode) runs as a GEMV with FP32 accumulation over the
// raw TFLite row-major weight bytes (trtllm/int2_gemv.h).
//
// `gemm`: many activation rows (prefill) run with INT4 weights as a GEMM on
// the tensor cores (trtllm/subbyte_gemm.h), for the shapes that supports. It
// reads the same row-major bytes as the GEMV.
//
// A `gate` (LiteRtNvidiaGemmGate) fuses the two projections of a gated
// feed-forward block in the GEMM: the weights and scales hold the gate
// projection followed by the up projection, rows / 2 channels each, and the
// output is gelu(gate) * up, [..., rows / 2].
//
// With `holder_granule` the weights input is not the packed weights but an
// INT64 "holder" constant that many plugins of the engine read: the packed
// weights are `weight_offset` bytes into the segment the holder stands for,
// which starts at the first multiple of `holder_granule` (a power of two) in
// the holder's device memory. The holder is one granule larger than the
// segment, so the segment fits wherever TensorRT places the holder, and an
// application can back it with CUDA virtual memory that it maps into several
// engines (bytecode.h, TensorRtWeightStore).
//
// `scale_block` 32 selects TFLite blockwise quantization (Gemma 4 26B-A4B):
// INT4 weights with one FP16 scale per block of 32 input dims, the scales
// [rows, columns / 32]. The scales input is then FP16, or, with
// `scale_offset` >= 0, the holder again: the scales are `scale_offset` bytes
// into its segment. The GEMV is trtllm/int4_block_gemv.h and the GEMM takes
// the scales per block.
//
// `norm_eps` >= 0 fuses an RMS norm of the activation rows in front of the
// product (blockwise weights only): a fourth input, the norm's FP32 scale
// vector [columns], and x * rsqrt(mean(x^2) + norm_eps) * scale *
// norm_multiplier replaces x. With a gate the blockwise GEMV also takes a
// few rows (gemm false): gelu(gate) * up in one launch.
nvinfer1::IPluginV3* CreateSubbyteGemvPlugin(
    int32_t bit_width, int32_t rows, int32_t columns, int32_t gate = 0,
    bool gemm = false, int64_t weight_offset = 0, int64_t holder_granule = 0,
    int32_t scale_block = 0, int64_t scale_offset = -1, float norm_eps = -1.0f,
    float norm_multiplier = 1.0f) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureSubbyteGemvPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SUBBYTE_GEMV_PLUGIN_H_
