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
//   inputs  activation BF16, packed weights INT8, scales BF16
//   output  [..., rows] BF16
// One activation row (decode) runs as a GEMV with FP32 accumulation over the
// raw TFLite row-major weight bytes (trtllm/int2_gemv.h).
//
// `tiled`: many activation rows (prefill) run with INT4 weights as a GEMM on
// the tensor cores (trtllm/subbyte_gemm.h), for the shapes that supports.
// The packed weights are then the raw bytes in the order of
// LiteRtNvidiaSubbyteGemmTileWeights.
//
// A `gate` (LiteRtNvidiaGemmGate) fuses the two projections of a gated
// feed-forward block in the GEMM: the weights and scales hold the gate
// projection followed by the up projection, rows / 2 channels each, and the
// output is gelu(gate) * up, [..., rows / 2].
nvinfer1::IPluginV3* CreateSubbyteGemvPlugin(int32_t bit_width, int32_t rows,
                                             int32_t columns, int32_t gate = 0,
                                             bool tiled = false) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureSubbyteGemvPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SUBBYTE_GEMV_PLUGIN_H_
