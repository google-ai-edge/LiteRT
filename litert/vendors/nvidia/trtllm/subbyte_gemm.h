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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SUBBYTE_GEMM_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SUBBYTE_GEMM_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Tensor-core GEMM of many BF16 activation rows (prefill) with signed INT4
// weights (two two's-complement values per byte, least-significant first)
// and BF16 per-channel scales:
//   y[m][n] = scales[n] * sum_k activation[m][k] * weights[n][k]
// The output is BF16.
//
// A block multiplies 128 activation rows by 128 weight rows. It reads the
// activations, which a first launch scales per row by a power of two into
// [-1, 1] and converts to FP16, and converts the weights to FP16 while it
// multiplies. The weights are the raw TFLite row-major bytes [channels,
// input_size], the layout the decode GEMV (int2_gemv.h) reads too, so prefill
// and decode engines can share one copy of them: a block stages 64 input dims
// of its 128 weight rows at a time. Products accumulate in FP16 over 128 input
// dims at a time, which cannot overflow (activations of at most 1 times
// weights of at most 8), and in FP32 beyond; the error this adds stays below
// the rounding of a BF16 result.
enum LiteRtNvidiaGemmGate : int32_t {
  // output[m][n] = y[m][n] for output_size channels.
  kLiteRtNvidiaGemmGateNone = 0,
  // weights and scales hold 2 * output_size channels, the gate projection
  // followed by the up projection of a feed-forward block:
  // output[m][n] = gelu(y[m][n]) * y[m][output_size + n], with the tanh
  // approximation of GELU or the exact one.
  kLiteRtNvidiaGemmGateGeluTanh = 1,
  kLiteRtNvidiaGemmGateGeluErf = 2,
};

struct LiteRtNvidiaGemmShape {
  int32_t rows;
  int32_t input_size;
  int32_t output_size;
  int32_t gate;  // LiteRtNvidiaGemmGate
};

// rows a multiple of 128, input_size a multiple of 128, an even output_size.
extern "C" bool LiteRtNvidiaSubbyteGemmSupports(
    const LiteRtNvidiaGemmShape* shape);

// Whether the current device can run the kernel (compute capability 8.0 or
// newer with about 81 KB of shared memory per block).
extern "C" bool LiteRtNvidiaSubbyteGemmAvailable();

extern "C" size_t LiteRtNvidiaSubbyteGemmWorkspaceBytes(
    const LiteRtNvidiaGemmShape* shape);

// The bytes of the row-major weights the launch reads: [channels,
// input_size / 2], with a gate the gate projection followed by the up
// projection (2 * output_size channels).
extern "C" size_t LiteRtNvidiaSubbyteGemmWeightBytes(
    const LiteRtNvidiaGemmShape* shape);

// The blocks of a launch, one per 128 activation rows and column tile.
extern "C" int32_t LiteRtNvidiaSubbyteGemmBlocks(
    const LiteRtNvidiaGemmShape* shape);

// Whether the launch keeps the current device busy: it has at least three
// blocks per two multiprocessors. Part of the device idles during smaller
// products.
extern "C" bool LiteRtNvidiaSubbyteGemmFillsDevice(
    const LiteRtNvidiaGemmShape* shape);

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4Gemm(
    const LiteRtNvidiaGemmShape* shape, const void* activation,
    const uint8_t* weights, const void* scales, void* output, void* workspace,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SUBBYTE_GEMM_H_
