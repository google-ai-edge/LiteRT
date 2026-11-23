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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_GEMM_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_GEMM_H_

#include <cstddef>
#include <cstdint>

#include "driver_types.h"
#include "litert/vendors/nvidia/trtllm/quant_linear.h"

// The products of quant_linear.h for many activation rows at once (a prefill
// chunk), on the INT8 tensor cores: LiteRtNvidiaLaunchQuantLinear's rows
// layout with the same INT8 results, bit for bit.
//
// A block multiplies 128 activation rows by 128 weight rows, 32 inputs per
// MMA with INT8 operands and INT32 sums. The activations are staged as they
// are; the packed weights (the model file's bytes, as the row kernels read
// them) are widened to INT8 as a block multiplies: an INT4 weight q becomes
// 16 * q and a centered INT2 weight, whose value is q + 0.5, becomes
// 32 * (2 * q + 1), both a shift of the field into the top of its byte. A sum
// is therefore 16 or 32 times the integer the row kernels accumulate (the
// product sum for INT4 weights, twice it for centered INT2), exactly, and
// stays below 2^31 for up to 65,536 inputs. The requantization, the SiLU
// table and the regrouping of the outputs are the row kernels', operation
// for operation.

// Whether the GEMM runs `m` activation rows of `params`: the rows layout
// with INT8 results (no logits), m a multiple of 128, the outputs of a
// segment a multiple of 128, and the inputs a multiple of 128 up to 65,536.
extern "C" bool LiteRtNvidiaQuantGemmSupports(
    const LiteRtNvidiaQuantLinearParams* params, int32_t m);

// Whether the current device can run the kernel: compute capability 8.0 or
// newer with the shared memory a block needs (about 57 KB).
extern "C" bool LiteRtNvidiaQuantGemmAvailable();

// The bytes of workspace a launch needs: the INT8 activations of a float
// activation, which a first launch quantizes as the row kernels do. 0 for
// INT8 activations.
extern "C" size_t LiteRtNvidiaQuantGemmWorkspaceBytes(
    const LiteRtNvidiaQuantLinearParams* params, int32_t m);

// LiteRtNvidiaLaunchQuantLinear for a product LiteRtNvidiaQuantGemmSupports
// takes: the same arguments (there are no logits, so no row scales) and
// `workspace`. Activations or weights that do not start on 16 bytes, and a
// device that cannot run the kernel, go to the row kernels.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantGemm(
    const LiteRtNvidiaQuantLinearParams* params, int32_t m,
    const void* activation, const uint8_t* weights, const float* multipliers,
    const int8_t* lut, void* output, void* workspace, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_GEMM_H_
